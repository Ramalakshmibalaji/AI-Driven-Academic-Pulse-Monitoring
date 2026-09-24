import pandas as pd

# Load final ML dataset
df = pd.read_csv("final_ml_dataset.csv")


# Features we may use for Academic Pulse
pulse_features = [
    "login_count",
    "active_days",
    "login_frequency",
    "login_consistency",
    "course_access_count",
    "module_completion_rate",
    "page_view_count",
    "total_practice",
    "exercises_attempted",
    "quiz_attempt_count",
    "quiz_completion_rate",
    "quiz_average_score",
    "study_session_count",
    "study_frequency_per_week",
    "week_1_6_engagement",
    "week_1_9_engagement",
    "week_1_12_engagement",
    "week_1_16_engagement",
    "completed_modules",
    "study_interval_regularity"
]


print("\n==========================================")
print("       ACADEMIC PULSE FEATURE RANGES")
print("==========================================")

for feature in pulse_features:

    print(f"\n{feature}")
    print("-" * len(feature))

    print("Minimum :", df[feature].min())
    print("Maximum :", df[feature].max())
    print("Mean    :", round(df[feature].mean(), 2))