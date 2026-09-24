import pandas as pd


# ==========================================
# 1. Load Student Prediction Dataset
# ==========================================

df = pd.read_csv("student_prediction_data.csv")


# ==========================================
# 2. Min-Max Normalization (0-100)
# ==========================================

def normalize(series):

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(50, index=series.index)

    return (
        (series - min_value)
        / (max_value - min_value)
    ) * 100


# ==========================================
# 3. Engagement Score
# ==========================================

engagement_features = [
    "login_count",
    "course_access_count",
    "page_view_count",
    "total_practice",
    "quiz_attempt_count"
]

engagement_normalized = pd.DataFrame()

for feature in engagement_features:

    engagement_normalized[feature] = normalize(
        df[feature]
    )

df["engagement_score"] = (
    engagement_normalized.mean(axis=1)
)


# ==========================================
# 4. Consistency Score
# ==========================================

consistency_features = [
    "active_days",
    "login_consistency",
    "study_interval_regularity"
]

consistency_normalized = pd.DataFrame()

for feature in consistency_features:

    consistency_normalized[feature] = normalize(
        df[feature]
    )

df["consistency_score"] = (
    consistency_normalized.mean(axis=1)
)


# ==========================================
# 5. Learning Progress Score
# ==========================================

progress_features = [
    "module_completion_rate",
    "quiz_completion_rate",
    "exercises_attempted",
    "completed_modules"
]

progress_normalized = pd.DataFrame()

for feature in progress_features:

    progress_normalized[feature] = normalize(
        df[feature]
    )

df["learning_progress_score"] = (
    progress_normalized.mean(axis=1)
)


# ==========================================
# 6. Study Pattern Score
# ==========================================

study_features = [
    "login_frequency",
    "study_session_count",
    "study_frequency_per_week",
    "week_1_16_engagement"
]

study_normalized = pd.DataFrame()

for feature in study_features:

    study_normalized[feature] = normalize(
        df[feature]
    )

df["study_pattern_score"] = (
    study_normalized.mean(axis=1)
)


# ==========================================
# 7. Overall Academic Pulse Score
# ==========================================

df["academic_pulse_score"] = (
    df["engagement_score"]
    + df["consistency_score"]
    + df["learning_progress_score"]
    + df["study_pattern_score"]
) / 4


# ==========================================
# 8. Academic Pulse Thresholds
# ==========================================

low_threshold = (
    df["academic_pulse_score"].quantile(0.25)
)

high_threshold = (
    df["academic_pulse_score"].quantile(0.75)
)


# ==========================================
# 9. Select Student
# ==========================================

userid = int(
    input("\nEnter Student ID: ").strip()
)

student = df[
    df["userid"] == userid
]


# ==========================================
# 10. Display Academic Pulse
# ==========================================

if student.empty:

    print("\nStudent ID not found.")

else:

    # --------------------------------------
    # Get Student Pulse Score
    # --------------------------------------

    pulse_score = student[
        "academic_pulse_score"
    ].iloc[0]


    # --------------------------------------
    # Determine Monitoring Status
    # --------------------------------------

    if pulse_score <= low_threshold:

        pulse_status = "Low"

    elif pulse_score <= high_threshold:

        pulse_status = "Moderate"

    else:

        pulse_status = "High"


    # ======================================
    # Display Student Information
    # ======================================

    print("\n" + "=" * 60)

    print(
        "          ACADEMIC PULSE MONITORING"
    )

    print("=" * 60)


    print(
        "Student ID :",
        userid
    )

    print(
        "Course     :",
        student["course_name"].iloc[0]
    )


    # ======================================
    # Academic Pulse Indicators
    # ======================================

    print(
        "\n------ Academic Pulse Indicators ------"
    )


    print(
        "Engagement Score       :",
        round(
            student[
                "engagement_score"
            ].iloc[0],
            2
        )
    )


    print(
        "Consistency Score      :",
        round(
            student[
                "consistency_score"
            ].iloc[0],
            2
        )
    )


    print(
        "Learning Progress      :",
        round(
            student[
                "learning_progress_score"
            ].iloc[0],
            2
        )
    )


    print(
        "Study Pattern Score    :",
        round(
            student[
                "study_pattern_score"
            ].iloc[0],
            2
        )
    )


    # ======================================
    # Overall Academic Pulse
    # ======================================

    print(
        "\nAcademic Pulse Score   :",
        round(
            pulse_score,
            2
        )
    )


    print(
        "Monitoring Status      :",
        pulse_status
    )


    # ======================================
    # Threshold Information
    # ======================================

    print(
        "\n------ Dataset-Based Thresholds ------"
    )


    print(
        "Low Threshold          :",
        round(
            low_threshold,
            2
        )
    )


    print(
        "High Threshold         :",
        round(
            high_threshold,
            2
        )
    )


    print("=" * 60)