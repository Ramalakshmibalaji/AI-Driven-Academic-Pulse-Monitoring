import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.models import load_model


# ============================================================
# FINAL MODEL EVALUATION
# HYBRID MULTI-MODAL LSTM-TRANSFORMER
# ============================================================

DATA_FILE = "multimodal_fusion_dataset.csv"

MODEL_FILE = "hybrid_multimodal_lstm_transformer.keras"

STATIC_SCALER_FILE = "hybrid_static_scaler.pkl"

TEMPORAL_SCALER_FILE = "hybrid_temporal_scaler.pkl"

LABEL_ENCODER_FILE = "hybrid_label_encoder.pkl"


print("=" * 72)
print("FINAL MODEL EVALUATION - HYBRID LSTM-TRANSFORMER")
print("=" * 72)


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

required_files = [
    DATA_FILE,
    MODEL_FILE,
    STATIC_SCALER_FILE,
    TEMPORAL_SCALER_FILE,
    LABEL_ENCODER_FILE
]

for file_name in required_files:

    if not os.path.exists(file_name):

        raise FileNotFoundError(
            f"\nRequired file not found: {file_name}\n"
            f"Please make sure the file is present in the project folder."
        )


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)


target_col = "performance_level"


if target_col not in df.columns:

    raise ValueError(
        f"Target column '{target_col}' not found."
    )


# ============================================================
# 3. DEFINE EXACT MODEL FEATURES
# ============================================================

# ------------------------------------------------------------
# STATIC / FUSED FEATURES
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# TEMPORAL FEATURES
# ------------------------------------------------------------

temporal_features = [

    f"week_{i}"

    for i in range(1, 17)

]


# ============================================================
# 4. CHECK FEATURES
# ============================================================

required_features = (

    static_features
    +
    temporal_features
    +
    [target_col]

)


missing_features = [

    feature

    for feature in required_features

    if feature not in df.columns

]


if missing_features:

    raise ValueError(

        "The following required features are missing:\n"

        +
        "\n".join(
            f" - {feature}"
            for feature in missing_features
        )

    )


print("\n" + "=" * 72)
print("MODEL FEATURES")
print("=" * 72)


print(
    "\nStatic features:",
    len(static_features)
)


for feature in static_features:

    print(
        " -",
        feature
    )


print(
    "\nTemporal features:",
    len(temporal_features)
)


for feature in temporal_features:

    print(
        " -",
        feature
    )


# ============================================================
# 5. CREATE INPUT DATA
# ============================================================

X_static = df[
    static_features
].copy()


X_temporal = df[
    temporal_features
].copy()


y = df[
    target_col
].copy()


# ============================================================
# 6. LOAD LABEL ENCODER
# ============================================================

print("\nLoading label encoder...")


label_encoder = joblib.load(
    LABEL_ENCODER_FILE
)


y_encoded = label_encoder.transform(
    y
)


class_names = label_encoder.classes_


print("\nTarget classes:")


for index, class_name in enumerate(class_names):

    print(
        f" {index} = {class_name}"
    )


# ============================================================
# 7. REPRODUCE TRAIN / VALIDATION / TEST SPLIT
# ============================================================

(
    X_static_train_val,
    X_static_test,

    X_temporal_train_val,
    X_temporal_test,

    y_train_val,
    y_test

) = train_test_split(

    X_static,

    X_temporal,

    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded

)


(
    X_static_train,
    X_static_val,

    X_temporal_train,
    X_temporal_val,

    y_train,
    y_val

) = train_test_split(

    X_static_train_val,

    X_temporal_train_val,

    y_train_val,

    test_size=0.20,

    random_state=42,

    stratify=y_train_val

)


print("\n" + "=" * 72)
print("DATA SPLIT")
print("=" * 72)


print(
    "Train      :",
    len(y_train)
)


print(
    "Validation :",
    len(y_val)
)


print(
    "Test       :",
    len(y_test)
)


# ============================================================
# 8. LOAD TRAINING SCALERS
# ============================================================

print("\nLoading scalers...")


static_scaler = joblib.load(
    STATIC_SCALER_FILE
)


temporal_scaler = joblib.load(
    TEMPORAL_SCALER_FILE
)


print("Scalers loaded successfully.")


# ============================================================
# 9. SCALE STATIC FEATURES
# ============================================================

X_test_static = static_scaler.transform(
    X_static_test
)


# ============================================================
# 10. SCALE TEMPORAL FEATURES
# ============================================================

# IMPORTANT:
#
# The temporal scaler was fitted using the 16 weekly
# features together.
#
# Therefore we pass all 16 weekly columns directly
# to StandardScaler.
#
# Input before scaling:
#
#     (457, 16)
#
# Output after scaling:
#
#     (457, 16)
#
# Then reshape into:
#
#     (457, 16, 1)
#
# for the LSTM temporal branch.

X_test_temporal_scaled = temporal_scaler.transform(
    X_temporal_test
)


# ============================================================
# 11. RESHAPE TEMPORAL DATA FOR LSTM
# ============================================================

X_test_temporal = X_test_temporal_scaled.reshape(

    -1,

    16,

    1

)


# ============================================================
# 12. VERIFY INPUT SHAPES
# ============================================================

print("\n" + "=" * 72)
print("MODEL INPUT SHAPES")
print("=" * 72)


print(
    "Static test input   :",
    X_test_static.shape
)


print(
    "Temporal test input :",
    X_test_temporal.shape
)


# Expected:
#
# Static:
#     (457, 8)
#
# Temporal:
#     (457, 16, 1)


if X_test_static.shape[1] != 8:

    raise ValueError(

        "Static input shape error. "
        f"Expected 8 features but received "
        f"{X_test_static.shape[1]}."

    )


if X_test_temporal.shape[1:] != (16, 1):

    raise ValueError(

        "Temporal input shape error. "
        f"Expected (samples, 16, 1) but received "
        f"{X_test_temporal.shape}."

    )


# ============================================================
# 13. LOAD TRAINED HYBRID MODEL
# ============================================================

print("\n" + "=" * 72)
print("LOADING HYBRID MODEL")
print("=" * 72)


model = load_model(
    MODEL_FILE
)


print(
    "Model loaded successfully."
)


# ============================================================
# 14. VERIFY MODEL INPUTS
# ============================================================

print("\n" + "=" * 72)
print("MODEL INPUT VERIFICATION")
print("=" * 72)


for input_layer in model.inputs:

    print(
        "Input name :",
        input_layer.name
    )

    print(
        "Input shape:",
        input_layer.shape
    )


# ============================================================
# 15. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")


# IMPORTANT:
#
# Hybrid model has TWO inputs:
#
# 1. static_input
# 2. temporal_input
#
# Therefore inputs must be passed as a dictionary.

y_prob = model.predict(

    {

        "static_input":
            X_test_static,

        "temporal_input":
            X_test_temporal

    },

    verbose=0

)


print(
    "Predictions generated successfully."
)


# ============================================================
# 16. CONVERT PROBABILITIES TO CLASS LABELS
# ============================================================

if (

    y_prob.ndim == 2

    and

    y_prob.shape[1] > 1

):

    y_pred = np.argmax(

        y_prob,

        axis=1

    )

else:

    y_pred = (

        y_prob.ravel() >= 0.5

    ).astype(int)


# ============================================================
# 17. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


precision = precision_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


recall = recall_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


f1 = f1_score(

    y_test,

    y_pred,

    average="weighted",

    zero_division=0

)


macro_f1 = f1_score(

    y_test,

    y_pred,

    average="macro",

    zero_division=0

)


# ============================================================
# 18. MULTI-CLASS ROC-AUC
# ============================================================

roc_auc = None


if (

    y_prob.ndim == 2

    and

    y_prob.shape[1] == len(class_names)

):

    y_test_bin = label_binarize(

        y_test,

        classes=np.arange(
            len(class_names)
        )

    )


    roc_auc = roc_auc_score(

        y_test_bin,

        y_prob,

        average="weighted",

        multi_class="ovr"

    )


# ============================================================
# 19. PRINT FINAL TEST PERFORMANCE
# ============================================================

print("\n" + "=" * 72)
print("FINAL TEST PERFORMANCE")
print("=" * 72)


print(

    f"Accuracy        : "
    f"{accuracy * 100:.2f}%"

)


print(

    f"Precision       : "
    f"{precision * 100:.2f}%"

)


print(

    f"Recall          : "
    f"{recall * 100:.2f}%"

)


print(

    f"F1 Score        : "
    f"{f1 * 100:.2f}%"

)


print(

    f"Macro F1        : "
    f"{macro_f1 * 100:.2f}%"

)


if roc_auc is not None:

    print(

        f"Weighted ROC-AUC: "
        f"{roc_auc:.4f}"

    )


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 72)
print("CLASSIFICATION REPORT")
print("=" * 72)


report = classification_report(

    y_test,

    y_pred,

    target_names=class_names,

    zero_division=0

)


print(report)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred

)


print("=" * 72)
print("CONFUSION MATRIX")
print("=" * 72)


print(cm)


# ------------------------------------------------------------
# Plot confusion matrix
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 6)
)


plt.imshow(

    cm,

    interpolation="nearest"

)


plt.title(

    "Hybrid Multi-Modal LSTM-Transformer Confusion Matrix"

)


plt.xlabel(

    "Predicted Class"

)


plt.ylabel(

    "Actual Class"

)


plt.xticks(

    range(len(class_names)),

    class_names

)


plt.yticks(

    range(len(class_names)),

    class_names

)


for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(

            j,

            i,

            cm[i, j],

            ha="center",

            va="center"

        )


plt.colorbar()


plt.tight_layout()


plt.savefig(

    "hybrid_lstm_transformer_confusion_matrix.png",

    dpi=300

)


plt.show()


# ============================================================
# 22. ROC CURVES
# ============================================================

if (

    y_prob.ndim == 2

    and

    y_prob.shape[1] == len(class_names)

):

    y_test_bin = label_binarize(

        y_test,

        classes=np.arange(
            len(class_names)
        )

    )


    plt.figure(

        figsize=(8, 6)

    )


    for i, class_name in enumerate(
        class_names
    ):

        fpr, tpr, _ = roc_curve(

            y_test_bin[:, i],

            y_prob[:, i]

        )


        class_auc = auc(

            fpr,

            tpr

        )


        plt.plot(

            fpr,

            tpr,

            label=(
                f"{class_name} "
                f"(AUC = {class_auc:.3f})"
            )

        )


    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--"

    )


    plt.title(

        "ROC Curves - Hybrid LSTM-Transformer"

    )


    plt.xlabel(

        "False Positive Rate"

    )


    plt.ylabel(

        "True Positive Rate"

    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        "hybrid_lstm_transformer_roc_curve.png",

        dpi=300

    )


    plt.show()


# ============================================================
# 23. SAVE FINAL METRICS
# ============================================================

metrics_df = pd.DataFrame(

    [

        {

            "Model":
                "Hybrid LSTM-Transformer",

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1_Score":
                f1,

            "Macro_F1":
                macro_f1,

            "Weighted_ROC_AUC":
                roc_auc

        }

    ]

)


metrics_df.to_csv(

    "final_model_evaluation.csv",

    index=False

)


# ============================================================
# 24. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(

    {

        "Actual":
            label_encoder.inverse_transform(
                y_test
            ),

        "Predicted":
            label_encoder.inverse_transform(
                y_pred
            )

    }

)


# ------------------------------------------------------------
# Add class probability columns
# ------------------------------------------------------------

if (

    y_prob.ndim == 2

    and

    y_prob.shape[1] == len(class_names)

):

    for i, class_name in enumerate(
        class_names
    ):

        prediction_df[
            f"Probability_{class_name}"
        ] = y_prob[:, i]


prediction_df.to_csv(

    "hybrid_lstm_transformer_test_predictions.csv",

    index=False

)


# ============================================================
# 25. SAVE CONFUSION MATRIX CSV
# ============================================================

cm_df = pd.DataFrame(

    cm,

    index=[

        f"Actual_{c}"

        for c in class_names

    ],

    columns=[

        f"Predicted_{c}"

        for c in class_names

    ]

)


cm_df.to_csv(

    "hybrid_lstm_transformer_confusion_matrix.csv"

)


# ============================================================
# 26. SAVE CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(

    y_test,

    y_pred,

    target_names=class_names,

    output_dict=True,

    zero_division=0

)


report_df = pd.DataFrame(
    report_dict
).transpose()


report_df.to_csv(

    "hybrid_lstm_transformer_classification_report.csv"

)


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("FILES SAVED")
print("=" * 72)


print(
    " - final_model_evaluation.csv"
)


print(
    " - hybrid_lstm_transformer_confusion_matrix.png"
)


print(
    " - hybrid_lstm_transformer_confusion_matrix.csv"
)


print(
    " - hybrid_lstm_transformer_roc_curve.png"
)


print(
    " - hybrid_lstm_transformer_test_predictions.csv"
)


print(
    " - hybrid_lstm_transformer_classification_report.csv"
)


print("\n" + "=" * 72)
print("FINAL MODEL EVALUATION COMPLETED")
print("=" * 72)