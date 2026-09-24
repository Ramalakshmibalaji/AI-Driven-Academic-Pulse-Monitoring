import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model


# ============================================================
# 1. LOAD DATASETS
# ============================================================

# Student-level dataset used for searching by userid
student_df = pd.read_csv(
    "student_prediction_data.csv"
)

# Exact dataset used during ANN training
ann_df = pd.read_csv(
    "final_ml_dataset.csv"
)


# ============================================================
# 2. LOAD SAVED ANN MODEL
# ============================================================

model = load_model(
    "ann_student_performance_model.keras"
)


# ============================================================
# 3. LOAD SAVED PREPROCESSING OBJECTS
# ============================================================

scaler = joblib.load(
    "scaler.pkl"
)

course_encoder = joblib.load(
    "course_encoder.pkl"
)

label_encoder = joblib.load(
    "label_encoder.pkl"
)


# ============================================================
# 4. GET EXACT ANN FEATURE COLUMNS
# ============================================================

ann_feature_columns = [
    column
    for column in ann_df.columns
    if column != "performance_level"
]


# ============================================================
# 5. IDENTIFY ANN FEATURE TYPES
# ============================================================

ann_X = ann_df[
    ann_feature_columns
]

ann_categorical_columns = (
    ann_X.select_dtypes(
        include=["object"]
    ).columns.tolist()
)

ann_numerical_columns = (
    ann_X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()
)


# ============================================================
# 6. ACADEMIC PULSE NORMALIZATION
# ============================================================

def normalize(series):

    min_value = series.min()

    max_value = series.max()

    if max_value == min_value:

        return pd.Series(
            50,
            index=series.index
        )

    return (
        (series - min_value)
        / (max_value - min_value)
    ) * 100


# ============================================================
# 7. CALCULATE ACADEMIC PULSE
# ============================================================

# -----------------------------
# Engagement
# -----------------------------

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
        student_df[feature]
    )

student_df["engagement_score"] = (
    engagement_normalized.mean(
        axis=1
    )
)


# -----------------------------
# Consistency
# -----------------------------

consistency_features = [
    "active_days",
    "login_consistency",
    "study_interval_regularity"
]

consistency_normalized = pd.DataFrame()

for feature in consistency_features:

    consistency_normalized[feature] = normalize(
        student_df[feature]
    )

student_df["consistency_score"] = (
    consistency_normalized.mean(
        axis=1
    )
)


# -----------------------------
# Learning Progress
# -----------------------------

progress_features = [
    "module_completion_rate",
    "quiz_completion_rate",
    "exercises_attempted",
    "completed_modules"
]

progress_normalized = pd.DataFrame()

for feature in progress_features:

    progress_normalized[feature] = normalize(
        student_df[feature]
    )

student_df["learning_progress_score"] = (
    progress_normalized.mean(
        axis=1
    )
)


# -----------------------------
# Study Pattern
# -----------------------------

study_features = [
    "login_frequency",
    "study_session_count",
    "study_frequency_per_week",
    "week_1_16_engagement"
]

study_normalized = pd.DataFrame()

for feature in study_features:

    study_normalized[feature] = normalize(
        student_df[feature]
    )

student_df["study_pattern_score"] = (
    study_normalized.mean(
        axis=1
    )
)


# ============================================================
# 8. OVERALL ACADEMIC PULSE SCORE
# ============================================================

student_df["academic_pulse_score"] = (
    student_df["engagement_score"]
    + student_df["consistency_score"]
    + student_df["learning_progress_score"]
    + student_df["study_pattern_score"]
) / 4


# ============================================================
# 9. DATASET-BASED PULSE THRESHOLDS
# ============================================================

low_threshold = (
    student_df[
        "academic_pulse_score"
    ].quantile(0.25)
)

high_threshold = (
    student_df[
        "academic_pulse_score"
    ].quantile(0.75)
)


# ============================================================
# 10. SELECT STUDENT
# ============================================================

userid = int(
    input(
        "\nEnter Student ID: "
    ).strip()
)


student = student_df[
    student_df["userid"] == userid
].copy()


# ============================================================
# 11. CHECK STUDENT
# ============================================================

if student.empty:

    print(
        "\nStudent ID not found."
    )

else:

    # ========================================================
    # STUDENT INFORMATION
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "           ACADEMIC PULSE MONITORING SYSTEM"
    )

    print("=" * 70)

    print(
        "Student ID :",
        userid
    )

    print(
        "Course     :",
        student[
            "course_name"
        ].iloc[0]
    )


    # ========================================================
    # LEARNING BEHAVIOUR METRICS
    # ========================================================

    print(
        "\n------ Learning Behaviour Metrics ------"
    )

    print(
        "Active Days              :",
        student[
            "active_days"
        ].iloc[0]
    )

    print(
        "Login Count              :",
        student[
            "login_count"
        ].iloc[0]
    )

    print(
        "Login Frequency          :",
        round(
            student[
                "login_frequency"
            ].iloc[0],
            2
        )
    )

    print(
        "Login Consistency        :",
        round(
            student[
                "login_consistency"
            ].iloc[0],
            2
        )
    )

    print(
        "Course Access Count      :",
        student[
            "course_access_count"
        ].iloc[0]
    )

    print(
        "Module Completion Rate   :",
        round(
            student[
                "module_completion_rate"
            ].iloc[0],
            2
        ),
        "%"
    )

    print(
        "Page View Count          :",
        student[
            "page_view_count"
        ].iloc[0]
    )

    print(
        "Total Practice           :",
        student[
            "total_practice"
        ].iloc[0]
    )

    print(
        "Exercises Attempted      :",
        student[
            "exercises_attempted"
        ].iloc[0]
    )

    print(
        "Quiz Attempts            :",
        student[
            "quiz_attempt_count"
        ].iloc[0]
    )

    print(
        "Quiz Completion Rate     :",
        round(
            student[
                "quiz_completion_rate"
            ].iloc[0],
            2
        ),
        "%"
    )

    print(
        "Quiz Average Score       :",
        round(
            student[
                "quiz_average_score"
            ].iloc[0],
            2
        )
    )

    print(
        "Study Sessions           :",
        student[
            "study_session_count"
        ].iloc[0]
    )

    print(
        "Study Frequency / Week   :",
        round(
            student[
                "study_frequency_per_week"
            ].iloc[0],
            2
        )
    )

    print(
        "Time Spent (minutes)     :",
        round(
            student[
                "time_spent_on_course_minutes"
            ].iloc[0],
            2
        )
    )

    print(
        "Study Regularity         :",
        round(
            student[
                "study_interval_regularity"
            ].iloc[0],
            2
        )
    )


    # ========================================================
    # ACADEMIC PULSE
    # ========================================================

    pulse_score = student[
        "academic_pulse_score"
    ].iloc[0]


    if pulse_score <= low_threshold:

        pulse_status = "Low"

    elif pulse_score <= high_threshold:

        pulse_status = "Moderate"

    else:

        pulse_status = "High"


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


    # ========================================================
    # WEEKLY MONITORING
    # ========================================================

    weekly_features = [
        f"week_{i}"
        for i in range(1, 17)
    ]


    weekly_values = student[
        weekly_features
    ].iloc[0].astype(float)


    # -----------------------------
    # Weekly Statistics
    # -----------------------------

    average_weekly_activity = (
        weekly_values.mean()
    )

    highest_week = (
        weekly_values.idxmax()
    )

    highest_value = (
        weekly_values.max()
    )

    lowest_week = (
        weekly_values.idxmin()
    )

    lowest_value = (
        weekly_values.min()
    )

    first_week = (
        weekly_values.iloc[0]
    )

    last_week = (
        weekly_values.iloc[-1]
    )


    # -----------------------------
    # Inactive Weeks
    # -----------------------------

    inactive_weeks = (
        weekly_values == 0
    ).sum()


    # -----------------------------
    # Recent Activity
    # -----------------------------

    recent_weeks = (
        weekly_values.iloc[-4:]
    )

    recent_average = (
        recent_weeks.mean()
    )

    recent_inactive_weeks = (
        recent_weeks == 0
    ).sum()


    if recent_inactive_weeks == 4:

        recent_activity = "Inactive"

    elif recent_inactive_weeks >= 2:

        recent_activity = "Low Activity"

    elif recent_average < average_weekly_activity:

        recent_activity = "Moderate Activity"

    else:

        recent_activity = "Active"


    # ========================================================
    # OVERALL WEEKLY TREND
    # ========================================================

    first_half_average = (
        weekly_values.iloc[:8].mean()
    )

    second_half_average = (
        weekly_values.iloc[8:].mean()
    )

    activity_range = (
        weekly_values.max()
        - weekly_values.min()
    )

    mean_activity = (
        weekly_values.mean()
    )


    if mean_activity == 0:

        trend = "No Activity"

    elif activity_range > mean_activity:

        trend = "Fluctuating"

    elif second_half_average > (
        first_half_average * 1.10
    ):

        trend = "Increasing"

    elif second_half_average < (
        first_half_average * 0.90
    ):

        trend = "Decreasing"

    else:

        trend = "Stable"


    # ========================================================
    # DISPLAY WEEKLY MONITORING
    # ========================================================

    print(
        "\n------ Weekly Learning Monitoring ------"
    )

    print(
        "Average Weekly Activity :",
        round(
            average_weekly_activity,
            2
        )
    )

    print(
        "Highest Activity        :",
        highest_week,
        "(",
        highest_value,
        ")"
    )

    print(
        "Lowest Activity         :",
        lowest_week,
        "(",
        lowest_value,
        ")"
    )

    print(
        "First Week Activity     :",
        first_week
    )

    print(
        "Last Week Activity      :",
        last_week
    )

    print(
        "Inactive Weeks          :",
        inactive_weeks
    )

    print(
        "Recent Activity         :",
        recent_activity
    )

    print(
        "Overall Weekly Trend    :",
        trend
    )


    # ========================================================
    # ANN PERFORMANCE PREDICTION
    # ========================================================

    print(
        "\n------ ANN Performance Prediction ------"
    )


    # IMPORTANT:
    # Select EXACTLY the same features and
    # same order used during ANN training.

    X_student = student[
        ann_feature_columns
    ].copy()


    # -----------------------------
    # Numerical Features
    # -----------------------------

    X_num = X_student[
        ann_numerical_columns
    ]


    X_num_scaled = scaler.transform(
        X_num
    )


    # -----------------------------
    # Categorical Features
    # -----------------------------

    X_cat = X_student[
        ann_categorical_columns
    ]


    X_cat_encoded = (
        course_encoder.transform(
            X_cat
        )
    )


    # -----------------------------
    # Combine Features
    # -----------------------------

    X_processed = np.hstack(
        [
            X_num_scaled,
            X_cat_encoded
        ]
    )


    # ========================================================
    # ANN PREDICTION
    # ========================================================

    probabilities = model.predict(
        X_processed,
        verbose=0
    )


    predicted_class = np.argmax(
        probabilities,
        axis=1
    )


    predicted_label = (
        label_encoder.inverse_transform(
            predicted_class
        )
    )


    prediction = predicted_label[0]


    confidence = (
        np.max(
            probabilities[0]
        ) * 100
    )


    print(
        "Predicted Performance    :",
        prediction
    )

    print(
        "Prediction Confidence    :",
        f"{confidence:.2f}%"
    )


    print(
        "\nPrediction Probability:"
    )


    for class_name, probability in zip(
        label_encoder.classes_,
        probabilities[0]
    ):

        print(
            f"{class_name:<10}: "
            f"{probability * 100:.2f}%"
        )


    # ========================================================
    # FINAL MONITORING SUMMARY
    # ========================================================

    print(
        "\n------ Final Monitoring Summary ------"
    )

    print(
        "Academic Pulse           :",
        round(
            pulse_score,
            2
        ),
        "(",
        pulse_status,
        ")"
    )

    print(
        "Weekly Trend             :",
        trend
    )

    print(
        "Recent Activity          :",
        recent_activity
    )

    print(
        "Inactive Weeks           :",
        inactive_weeks
    )

    print(
        "ANN Prediction           :",
        prediction
    )

    print(
        "ANN Confidence           :",
        f"{confidence:.2f}%"
    )

    print("=" * 70)