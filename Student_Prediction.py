import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# ==========================================
# 1. Load Student Prediction Dataset
# ==========================================

df = pd.read_csv("student_prediction_data.csv")


# ==========================================
# 2. Load Saved ANN Model
# ==========================================

model = load_model(
    "ann_student_performance_model.keras"
)


# ==========================================
# 3. Load Saved Preprocessing Objects
# ==========================================

scaler = joblib.load("scaler.pkl")
course_encoder = joblib.load("course_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# ==========================================
# 4. Prepare Model Input
# ==========================================

X = df.drop(
    columns=["userid", "performance_level"]
)


# ==========================================
# 5. Identify Columns
# ==========================================

categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    exclude=["object", "category"]
).columns.tolist()


# ==========================================
# 6. Student Prediction Function
# ==========================================

def predict_student(userid):

    # --------------------------------------
    # Find selected student
    # --------------------------------------

    student_data = df[
        df["userid"] == userid
    ].copy()

    if student_data.empty:
        print("\nStudent ID not found.")
        return


    # ======================================
    # Prepare Model Input
    # ======================================

    X_student = student_data.drop(
        columns=["userid", "performance_level"]
    )


    # ======================================
    # Numerical Preprocessing
    # ======================================

    X_num = X_student[numerical_cols]

    X_num_scaled = scaler.transform(
        X_num
    )


    # ======================================
    # Categorical Preprocessing
    # ======================================

    X_cat = X_student[categorical_cols]

    X_cat_encoded = course_encoder.transform(
        X_cat
    )


    # ======================================
    # Combine Features
    # ======================================

    X_processed = np.hstack(
        [
            X_num_scaled,
            X_cat_encoded
        ]
    )


    # ======================================
    # ANN Prediction
    # ======================================

    probabilities = model.predict(
        X_processed,
        verbose=0
    )

    predicted_class = np.argmax(
        probabilities,
        axis=1
    )

    predicted_label = label_encoder.inverse_transform(
        predicted_class
    )


    # ======================================
    # Display Student Information
    # ======================================

    print("\n" + "=" * 60)
    print("       ACADEMIC PULSE MONITORING SYSTEM")
    print("=" * 60)

    print(
        "Student ID :",
        userid
    )

    print(
        "Course     :",
        student_data["course_name"].iloc[0]
    )


    # ======================================
    # Learning Behaviour Metrics
    # ======================================

    print("\n------ Learning Behaviour Metrics ------")

    print(
        "Active Days              :",
        student_data["active_days"].iloc[0]
    )

    print(
        "Login Count              :",
        student_data["login_count"].iloc[0]
    )

    print(
        "Login Frequency          :",
        round(
            student_data["login_frequency"].iloc[0],
            2
        )
    )

    print(
        "Login Consistency        :",
        round(
            student_data["login_consistency"].iloc[0],
            2
        )
    )

    print(
        "Course Access Count      :",
        student_data["course_access_count"].iloc[0]
    )

    print(
        "Module Completion Rate   :",
        round(
            student_data["module_completion_rate"].iloc[0],
            2
        ),
        "%"
    )

    print(
        "Page View Count          :",
        student_data["page_view_count"].iloc[0]
    )

    print(
        "Total Practice           :",
        student_data["total_practice"].iloc[0]
    )

    print(
        "Exercises Attempted      :",
        student_data["exercises_attempted"].iloc[0]
    )

    print(
        "Quiz Attempts            :",
        student_data["quiz_attempt_count"].iloc[0]
    )

    print(
        "Quiz Completion Rate     :",
        round(
            student_data["quiz_completion_rate"].iloc[0],
            2
        ),
        "%"
    )

    print(
        "Quiz Average Score       :",
        round(
            student_data["quiz_average_score"].iloc[0],
            2
        )
    )

    print(
        "Study Sessions           :",
        student_data["study_session_count"].iloc[0]
    )

    print(
        "Study Frequency / Week   :",
        round(
            student_data["study_frequency_per_week"].iloc[0],
            2
        )
    )

    print(
        "Time Spent (minutes)     :",
        round(
            student_data[
                "time_spent_on_course_minutes"
            ].iloc[0],
            2
        )
    )

    print(
        "Study Regularity         :",
        round(
            student_data[
                "study_interval_regularity"
            ].iloc[0],
            2
        )
    )


    # ======================================
    # ANN Prediction Result
    # ======================================

    print("\n------ ANN Prediction ------")

    print(
        "Predicted Performance    :",
        predicted_label[0]
    )

    print("\nPrediction Probability:")

    for class_name, probability in zip(
        label_encoder.classes_,
        probabilities[0]
    ):
        print(
            f"{class_name:<10}: "
            f"{probability * 100:.2f}%"
        )

    print("=" * 60)


# ==========================================
# 7. Check Available Student IDs
# ==========================================

print(
    "\nNumber of student records:",
    len(df)
)

print("\nSample Student IDs:")

print(
    df["userid"]
    .drop_duplicates()
    .head(10)
    .to_string(index=False)
)


# ==========================================
# 8. Test Prediction
# ==========================================

userid = int(
    input(
        "\nEnter Student ID: "
    ).strip()
)

predict_student(userid)