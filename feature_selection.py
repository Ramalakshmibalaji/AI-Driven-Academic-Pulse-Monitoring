import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

print("=" * 70)
print("FEATURE SELECTION - MULTI-MODAL LEARNING FEATURES")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD ENGINEERED DATASET
# ------------------------------------------------------------

df = pd.read_csv("feature_engineered_dataset.csv")

print(f"Input shape: {df.shape}")

target = "performance_level"

# Keep the real 16-week sequence intact for LSTM.
week_cols = [f"week_{i}" for i in range(1, 17)]

# Course name is categorical and will be encoded only for selection.
categorical_cols = ["course_name"]

# Numeric candidate features
numeric_cols = [
    c for c in df.columns
    if c not in [target, "course_name"]
]

# Remove constant features
constant_cols = [
    c for c in numeric_cols
    if df[c].nunique(dropna=False) <= 1
]

if constant_cols:
    print("\nConstant features removed:")
    print(constant_cols)
    numeric_cols = [c for c in numeric_cols if c not in constant_cols]

# ------------------------------------------------------------
# 2. DEFINE MODALITIES
# ------------------------------------------------------------

modalities = {
    "Engagement": [
        "login_count",
        "course_access_count",
        "page_view_count",
        "total_practice",
        "access_per_active_day",
        "page_views_per_active_day",
        "practice_per_active_day",
        "engagement_intensity",
    ],

    "Assessment_Learning": [
        "class_exercises_attempted",
        "exercises_attempted",
        "quiz_attempt_count",
        "quiz_completion_rate",
        "quiz_average_score",
        "completed_modules",
        "module_completion_rate",
        "quiz_efficiency",
        "exercise_attempts_per_active_day",
        "class_exercise_ratio",
        "learning_completion_index",
    ],

    "Study_Behaviour": [
        "active_days",
        "login_frequency",
        "login_consistency",
        "study_session_count",
        "time_spent_on_course_minutes",
        "study_frequency_per_week",
        "study_interval_regularity",
        "average_access_hour",
        "time_per_active_day",
        "sessions_per_active_day",
        "sessions_per_login",
        "weekly_activity_density",
    ],

    "Temporal_Behaviour": week_cols + [
        "weekly_mean",
        "weekly_std",
        "weekly_max",
        "weekly_min",
        "weekly_range",
        "weekly_volatility",
        "inactive_week_count",
        "first_half_weekly_mean",
        "second_half_weekly_mean",
        "temporal_change",
        "recent_week_mean",
        "weekly_trend",
    ],
}

# Remove columns that are not actually present
for name in modalities:
    modalities[name] = [
        c for c in modalities[name]
        if c in df.columns
    ]

# ------------------------------------------------------------
# 3. CORRELATION-BASED FILTERING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[1] CORRELATION-BASED FEATURE FILTERING")
print("=" * 70)

corr_candidates = [
    c for c in numeric_cols
    if c in df.columns
]

corr_matrix = df[corr_candidates].corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop_corr = [
    column
    for column in upper.columns
    if any(upper[column] >= 0.90)
]

corr_features = [
    c for c in corr_candidates
    if c not in to_drop_corr
]

print(f"Original numeric features : {len(corr_candidates)}")
print(f"Highly correlated removed : {len(to_drop_corr)}")
print(f"Remaining features        : {len(corr_features)}")

print("\nRemoved by correlation:")
for c in to_drop_corr:
    print(" -", c)

# ------------------------------------------------------------
# 4. PREPARE DATA FOR MI + RANDOM FOREST
# ------------------------------------------------------------

X_numeric = df[corr_features].copy()

# Encode course name
course_encoded = pd.get_dummies(
    df["course_name"],
    prefix="course",
    dtype=float
)

X_selection = pd.concat(
    [X_numeric, course_encoded],
    axis=1
)

X_selection = X_selection.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target])

print("\nSelection matrix shape:", X_selection.shape)

# ------------------------------------------------------------
# 5. MUTUAL INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[2] MUTUAL INFORMATION FEATURE RANKING")
print("=" * 70)

mi_scores = mutual_info_classif(
    X_selection,
    y,
    random_state=42
)

mi_df = pd.DataFrame({
    "feature": X_selection.columns,
    "mutual_information": mi_scores
}).sort_values(
    "mutual_information",
    ascending=False
)

print("\nTop 20 Mutual Information features:")
print(mi_df.head(20).to_string(index=False))

# Keep top 50% for the next stage, with at least 10.
mi_keep_count = max(10, int(len(mi_df) * 0.50))
mi_features = mi_df.head(mi_keep_count)["feature"].tolist()

# ------------------------------------------------------------
# 6. RANDOM FOREST FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[3] RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_selection[mi_features],
    y
)

rf_df = pd.DataFrame({
    "feature": mi_features,
    "rf_importance": rf.feature_importances_
}).sort_values(
    "rf_importance",
    ascending=False
)

print("\nTop 20 Random Forest features:")
print(rf_df.head(20).to_string(index=False))

# Keep top 30 features or all if fewer.
rf_keep_count = min(30, len(rf_df))
rf_features = rf_df.head(rf_keep_count)["feature"].tolist()

# ------------------------------------------------------------
# 7. RFE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[4] RFE FEATURE SELECTION")
print("=" * 70)

# RFE is applied to the strongest RF candidates.
rfe_keep_count = min(20, len(rf_features))

rfe_estimator = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rfe = RFE(
    estimator=rfe_estimator,
    n_features_to_select=rfe_keep_count,
    step=0.20
)

rfe.fit(
    X_selection[rf_features],
    y
)

rfe_features = [
    feature
    for feature, selected in zip(rf_features, rfe.support_)
    if selected
]

print(f"RFE selected: {len(rfe_features)} features")

for feature in rfe_features:
    print(" -", feature)

# ------------------------------------------------------------
# 8. MODALITY-WISE SELECTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[5] MODALITY-WISE FINAL SELECTION")
print("=" * 70)

# We want every modality represented.
selected_static = []
modality_selection = {}

# Rank final RFE features using RF importance.
importance_lookup = dict(
    zip(rf_df["feature"], rf_df["rf_importance"])
)

# Select up to 4 features from each static modality.
# Temporal sequence weeks are preserved separately for LSTM.
for modality, features in modalities.items():

    if modality == "Temporal_Behaviour":
        continue

    available = [
        f for f in features
        if f in rfe_features
    ]

    available = sorted(
        available,
        key=lambda x: importance_lookup.get(x, 0),
        reverse=True
    )

    chosen = available[:4]

    # If fewer than 2 survived, add the strongest modality features
    # that survived correlation filtering.
    if len(chosen) < 2:
        fallback = [
            f for f in features
            if f in corr_features and f not in chosen
        ]

        fallback = sorted(
            fallback,
            key=lambda x: importance_lookup.get(x, 0),
            reverse=True
        )

        for f in fallback:
            if f not in chosen:
                chosen.append(f)
            if len(chosen) >= 2:
                break

    modality_selection[modality] = chosen
    selected_static.extend(chosen)

# ------------------------------------------------------------
# 9. TEMPORAL FEATURES FOR LSTM
# ------------------------------------------------------------

# IMPORTANT:
# Do NOT independently drop week_1...week_16.
# LSTM needs the complete ordered sequence.
temporal_derived = [
    "weekly_mean",
    "weekly_std",
    "weekly_range",
    "weekly_volatility",
    "inactive_week_count",
    "first_half_weekly_mean",
    "second_half_weekly_mean",
    "temporal_change",
    "recent_week_mean",
]

# Select the strongest derived temporal summary features
temporal_available = [
    f for f in temporal_derived
    if f in rfe_features
]

temporal_available = sorted(
    temporal_available,
    key=lambda x: importance_lookup.get(x, 0),
    reverse=True
)

temporal_static_selected = temporal_available[:3]

modality_selection["Temporal_Behaviour"] = (
    week_cols + temporal_static_selected
)

print("\nFinal modality selection:")

for modality, features in modality_selection.items():
    print(f"\n{modality} ({len(features)} features)")
    for feature in features:
        print(" -", feature)

# ------------------------------------------------------------
# 10. BUILD FINAL FUSION DATASET
# ------------------------------------------------------------

all_selected = []

for features in modality_selection.values():
    for feature in features:
        if feature not in all_selected:
            all_selected.append(feature)

final_columns = all_selected + [target]

fusion_df = df[final_columns].copy()

print("\n" + "=" * 70)
print("FINAL MULTI-MODAL FUSION DATASET")
print("=" * 70)

print("Final shape:", fusion_df.shape)
print("Total selected features:", len(all_selected))
print("Target:", target)

print("\nFinal selected features:")
for feature in all_selected:
    print(" -", feature)

# ------------------------------------------------------------
# 11. SAVE RESULTS
# ------------------------------------------------------------

fusion_df.to_csv(
    "multimodal_fusion_dataset.csv",
    index=False
)

mi_df.to_csv(
    "feature_ranking_mutual_information.csv",
    index=False
)

rf_df.to_csv(
    "feature_ranking_random_forest.csv",
    index=False
)

selection_rows = []

for modality, features in modality_selection.items():
    for feature in features:
        selection_rows.append({
            "modality": modality,
            "feature": feature
        })

pd.DataFrame(selection_rows).to_csv(
    "final_modality_features.csv",
    index=False
)

print("\nSaved files:")
print(" - multimodal_fusion_dataset.csv")
print(" - feature_ranking_mutual_information.csv")
print(" - feature_ranking_random_forest.csv")
print(" - final_modality_features.csv")

print("\n" + "=" * 70)
print("FEATURE SELECTION COMPLETED")
print("=" * 70)
