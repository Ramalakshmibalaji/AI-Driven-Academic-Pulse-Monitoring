import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# ============================================================
# FILE PATHS
# ============================================================

DATA_FILE = "student_prediction_data.csv"
MODEL_FILE = "hybrid_multimodal_lstm_transformer.keras"
STATIC_SCALER_FILE = "hybrid_static_scaler.pkl"
TEMPORAL_SCALER_FILE = "hybrid_temporal_scaler.pkl"
LABEL_ENCODER_FILE = "hybrid_label_encoder.pkl"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\n==========================================")
print(" AI-DRIVEN ACADEMIC PULSE MONITORING")
print(" Student Performance Prediction")
print("==========================================")

print("\nDataset loaded successfully.")
print("Total records:", len(df))


# ============================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================

model = load_model(MODEL_FILE)

static_scaler = joblib.load(STATIC_SCALER_FILE)
temporal_scaler = joblib.load(TEMPORAL_SCALER_FILE)
label_encoder = joblib.load(LABEL_ENCODER_FILE)

print("Hybrid LSTM-Transformer model loaded.")


# ============================================================
# FEATURE GROUPS USED DURING FINAL MODEL TRAINING
# ============================================================

static_features = [
    "total_practice",
    "engagement_intensity",
    "quiz_average_score",
    "quiz_efficiency",
    "exercises_attempted",
    "class_exercises_attempted",
    "sessions_per_active_day",
    "study_interval_regularity"
]

temporal_features = [
    "week_1",
    "week_2",
    "week_3",
    "week_4",
    "week_5",
    "week_6",
    "week_7",
    "week_8",
    "week_9",
    "week_10",
    "week_11",
    "week_12",
    "week_13",
    "week_14",
    "week_15",
    "week_16"
]


# ============================================================
# GET STUDENT ID
# ============================================================

student_id = input("\nEnter Student ID: ").strip()


# ============================================================
# CREATE REQUIRED ENGINEERED FEATURES
# ============================================================

EPS = 1e-6


# ------------------------------------------------------------
# 1. ENGAGEMENT INTENSITY
# ------------------------------------------------------------

engagement_base = [
    "login_count",
    "course_access_count",
    "page_view_count",
    "total_practice"
]

available_engagement = [
    c for c in engagement_base
    if c in df.columns
]

if available_engagement:
    ranks = df[available_engagement].rank(pct=True)
    df["engagement_intensity"] = ranks.mean(axis=1) * 100
else:
    print("\nUnable to create engagement_intensity.")
    exit()


# ------------------------------------------------------------
# 2. QUIZ EFFICIENCY
# ------------------------------------------------------------

if {
    "quiz_average_score",
    "quiz_attempt_count"
}.issubset(df.columns):

    df["quiz_efficiency"] = (
        df["quiz_average_score"] /
        np.maximum(df["quiz_attempt_count"], EPS)
    )

else:
    print("\nUnable to create quiz_efficiency.")
    exit()


# ------------------------------------------------------------
# 3. SESSIONS PER ACTIVE DAY
# ------------------------------------------------------------

if {
    "study_session_count",
    "active_days"
}.issubset(df.columns):

    df["sessions_per_active_day"] = (
        df["study_session_count"] /
        np.maximum(df["active_days"], EPS)
    )

else:
    print("\nUnable to create sessions_per_active_day.")
    exit()


# ============================================================
# FIND STUDENT AFTER FEATURE CREATION
# ============================================================

try:
    student_id_numeric = int(student_id)
except ValueError:
    student_id_numeric = student_id


student_data = df[
    df["userid"] == student_id_numeric
]


if student_data.empty:
    print("\nStudent ID not found.")
    print("Please enter a valid Student ID.")
    exit()


# ============================================================
# SELECT STUDENT
# ============================================================

student = student_data.iloc[0]


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

required_features = (
    static_features +
    temporal_features
)

missing_features = [
    feature
    for feature in required_features
    if feature not in df.columns
]

if missing_features:

    print("\nMissing features in dataset:")
    print(missing_features)

    exit()


# ============================================================
# PREPARE STATIC FEATURES
# ============================================================

X_static = pd.DataFrame(
    [student[static_features].astype(float).values],
    columns=static_features
)

X_static_scaled = static_scaler.transform(X_static)



# ============================================================
# PREPARE TEMPORAL FEATURES
# ============================================================

X_temporal = pd.DataFrame(
    [student[temporal_features].astype(float).values],
    columns=temporal_features
)

X_temporal_scaled = temporal_scaler.transform(X_temporal)

X_temporal_input = X_temporal_scaled.reshape(1, 16, 1)



# LSTM input:
# samples = 1
# time steps = 16 weeks
# features = 1

X_temporal_input = X_temporal_scaled.reshape(
    1,
    16,
    1
)


# ============================================================
# MODEL PREDICTION
# ============================================================

prediction_probability = model.predict(
    {
        "static_input": X_static_scaled,
        "temporal_input": X_temporal_input
    },
    verbose=0
)


# ============================================================
# GET PREDICTED CLASS
# ============================================================

predicted_index = np.argmax(
    prediction_probability,
    axis=1
)[0]


predicted_class = label_encoder.inverse_transform(
    [predicted_index]
)[0]


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n==========================================")
print("         PREDICTION RESULT")
print("==========================================")

print("Student ID            :", student_id)
print("Predicted Performance :", predicted_class)


# ============================================================
# DISPLAY PROBABILITIES
# ============================================================

print("\nPrediction Probabilities:")

for class_name, probability in zip(
    label_encoder.classes_,
    prediction_probability[0]
):

    print(
        f"{class_name:10s}: "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# DISPLAY STUDENT INFORMATION
# ============================================================

if "course_name" in df.columns:

    print(
        "Course                :",
        student["course_name"]
    )


print("==========================================")

# ============================================================
# SAVE PREDICTION RESULT
# ============================================================

prediction_result = pd.DataFrame([{
    "student_id": student_id,
    "course_name": student["course_name"] if "course_name" in df.columns else "N/A",
    "predicted_performance": predicted_class,
    "high_probability": round(
        prediction_probability[0][
            list(label_encoder.classes_).index("High")
        ] * 100, 2
    ),
    "low_probability": round(
        prediction_probability[0][
            list(label_encoder.classes_).index("Low")
        ] * 100, 2
    ),
    "medium_probability": round(
        prediction_probability[0][
            list(label_encoder.classes_).index("Medium")
        ] * 100, 2
    )
}])

OUTPUT_FILE = "student_prediction_results.csv"

prediction_result.to_csv(
    OUTPUT_FILE,
    mode="a",
    header=not __import__("os").path.exists(OUTPUT_FILE),
    index=False
)

print("\nPrediction result saved to:", OUTPUT_FILE)