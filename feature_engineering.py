import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FEATURE ENGINEERING FOR AI-DRIVEN ACADEMIC PULSE MONITORING
# ============================================================

INPUT_FILE = "final_ml_dataset.csv"
OUTPUT_FILE = "feature_engineered_dataset.csv"

print("=" * 65)
print("FEATURE ENGINEERING - MULTI-MODAL LEARNING FEATURES")
print("=" * 65)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)
print(f"Original shape: {df.shape}")

# Keep target and course information for later modelling
if "performance_level" not in df.columns:
    raise ValueError("performance_level column not found.")

# Convert numerical columns safely
weekly_cols = [f"week_{i}" for i in range(1, 17) if f"week_{i}" in df.columns]

# ------------------------------------------------------------
# 2. REMOVE NON-INFORMATIVE / REDUNDANT INPUTS
# ------------------------------------------------------------
# home_exercises_attempted is constant zero according to EDA.
# Cumulative week features are retained for EDA/reference but are not
# used as separate engineered inputs because the raw weekly sequence
# contains the temporal information needed by LSTM.
remove_features = [
    "home_exercises_attempted",
    "week_1_6_engagement",
    "week_1_9_engagement",
    "week_1_12_engagement",
    "week_1_16_engagement",
]

remove_features = [c for c in remove_features if c in df.columns]
df = df.drop(columns=remove_features)
print("Removed non-informative / redundant columns:", remove_features)

# Helper to avoid division by zero
EPS = 1e-6

def safe_divide(a, b):
    return np.divide(a, np.maximum(b, EPS))

# ------------------------------------------------------------
# 3. MODALITY 1 - ENGAGEMENT FEATURES
# ------------------------------------------------------------
print("\n[1] ENGAGEMENT MODALITY")

if {"course_access_count", "active_days"}.issubset(df.columns):
    df["access_per_active_day"] = safe_divide(
        df["course_access_count"], df["active_days"]
    )

if {"page_view_count", "active_days"}.issubset(df.columns):
    df["page_views_per_active_day"] = safe_divide(
        df["page_view_count"], df["active_days"]
    )

if {"total_practice", "active_days"}.issubset(df.columns):
    df["practice_per_active_day"] = safe_divide(
        df["total_practice"], df["active_days"]
    )

# Overall engagement intensity from normalized component ranks.
engagement_base = [
    c for c in [
        "login_count",
        "course_access_count",
        "page_view_count",
        "total_practice",
    ] if c in df.columns
]

if engagement_base:
    ranks = df[engagement_base].rank(pct=True)
    df["engagement_intensity"] = ranks.mean(axis=1) * 100

# ------------------------------------------------------------
# 4. MODALITY 2 - ASSESSMENT / LEARNING FEATURES
# ------------------------------------------------------------
print("[2] ASSESSMENT / LEARNING MODALITY")

if {"quiz_average_score", "quiz_attempt_count"}.issubset(df.columns):
    # Score obtained per attempt, bounded to avoid extreme division values.
    df["quiz_efficiency"] = safe_divide(
        df["quiz_average_score"], df["quiz_attempt_count"]
    )

if {"exercises_attempted", "active_days"}.issubset(df.columns):
    df["exercise_attempts_per_active_day"] = safe_divide(
        df["exercises_attempted"], df["active_days"]
    )

if {"class_exercises_attempted", "exercises_attempted"}.issubset(df.columns):
    df["class_exercise_ratio"] = safe_divide(
        df["class_exercises_attempted"], df["exercises_attempted"]
    )

if {"module_completion_rate", "quiz_completion_rate"}.issubset(df.columns):
    df["learning_completion_index"] = (
        df["module_completion_rate"] + df["quiz_completion_rate"]
    ) / 2.0

# ------------------------------------------------------------
# 5. MODALITY 3 - STUDY BEHAVIOUR FEATURES
# ------------------------------------------------------------
print("[3] STUDY BEHAVIOUR MODALITY")

if {"time_spent_on_course_minutes", "active_days"}.issubset(df.columns):
    df["time_per_active_day"] = safe_divide(
        df["time_spent_on_course_minutes"], df["active_days"]
    )

if {"study_session_count", "active_days"}.issubset(df.columns):
    df["sessions_per_active_day"] = safe_divide(
        df["study_session_count"], df["active_days"]
    )

if {"login_count", "study_session_count"}.issubset(df.columns):
    df["sessions_per_login"] = safe_divide(
        df["study_session_count"], df["login_count"]
    )

if {"active_days", "study_frequency_per_week"}.issubset(df.columns):
    df["weekly_activity_density"] = safe_divide(
        df["active_days"], df["study_frequency_per_week"]
    )

# ------------------------------------------------------------
# 6. MODALITY 4 - TEMPORAL / WEEKLY FEATURES
# ------------------------------------------------------------
print("[4] TEMPORAL BEHAVIOUR MODALITY")

if len(weekly_cols) >= 2:
    weekly = df[weekly_cols].astype(float)

    df["weekly_mean"] = weekly.mean(axis=1)
    df["weekly_std"] = weekly.std(axis=1).fillna(0)
    df["weekly_max"] = weekly.max(axis=1)
    df["weekly_min"] = weekly.min(axis=1)
    df["weekly_range"] = df["weekly_max"] - df["weekly_min"]
    df["weekly_volatility"] = safe_divide(df["weekly_std"], df["weekly_mean"] + EPS)
    df["inactive_week_count"] = (weekly == 0).sum(axis=1)

    # First half vs second half activity
    first_half = weekly.iloc[:, :8].mean(axis=1)
    second_half = weekly.iloc[:, 8:16].mean(axis=1)
    df["first_half_weekly_mean"] = first_half
    df["second_half_weekly_mean"] = second_half
    df["temporal_change"] = second_half - first_half

    # Recent engagement gives more weight to the latest 4 weeks.
    df["recent_week_mean"] = weekly.iloc[:, -4:].mean(axis=1)

    # Linear trend across weeks: positive = increasing activity.
    x = np.arange(len(weekly_cols), dtype=float)
    x_centered = x - x.mean()
    denominator = np.sum(x_centered ** 2)
    df["weekly_trend"] = weekly.apply(
        lambda row: np.sum(x_centered * (row.values - row.values.mean())) / denominator,
        axis=1,
    )

# ------------------------------------------------------------
# 7. CLEAN ENGINEERED VALUES
# ------------------------------------------------------------
# Replace inf/nan caused by unusual source values.
df = df.replace([np.inf, -np.inf], np.nan)

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# ------------------------------------------------------------
# 8. SUMMARY
# ------------------------------------------------------------
print("\n" + "=" * 65)
print("ENGINEERED DATASET SUMMARY")
print("=" * 65)
print("Final shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

engineered_cols = [
    "access_per_active_day", "page_views_per_active_day",
    "practice_per_active_day", "engagement_intensity",
    "quiz_efficiency", "exercise_attempts_per_active_day",
    "class_exercise_ratio", "learning_completion_index",
    "time_per_active_day", "sessions_per_active_day",
    "sessions_per_login", "weekly_activity_density",
    "weekly_mean", "weekly_std", "weekly_max", "weekly_min",
    "weekly_range", "weekly_volatility", "inactive_week_count",
    "first_half_weekly_mean", "second_half_weekly_mean",
    "temporal_change", "recent_week_mean", "weekly_trend"
]
engineered_cols = [c for c in engineered_cols if c in df.columns]

print("\nNew engineered features:")
for c in engineered_cols:
    print(" -", c)

print("\nTarget distribution:")
print(df["performance_level"].value_counts())

# ------------------------------------------------------------
# 9. SAVE
# ------------------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")
print("=" * 65)
