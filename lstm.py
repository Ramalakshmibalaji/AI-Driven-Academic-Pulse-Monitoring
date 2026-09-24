import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Concatenate,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# MULTI-MODAL LSTM - ACADEMIC PERFORMANCE PREDICTION
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("MULTI-MODAL LSTM - ACADEMIC PERFORMANCE PREDICTION")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv("multimodal_fusion_dataset.csv")

target = "performance_level"

week_cols = [f"week_{i}" for i in range(1, 17)]

engagement_cols = [
    "total_practice",
    "engagement_intensity"
]

assessment_cols = [
    "quiz_average_score",
    "quiz_efficiency",
    "exercises_attempted",
    "class_exercises_attempted"
]

study_cols = [
    "sessions_per_active_day",
    "study_interval_regularity"
]

static_cols = engagement_cols + assessment_cols + study_cols

print("\nDataset shape:", df.shape)
print("Static features:", len(static_cols))
print("Temporal features:", len(week_cols))

# ------------------------------------------------------------
# 2. PREPARE TARGET
# ------------------------------------------------------------

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(df[target])

print("\nTarget classes:")
for i, label in enumerate(label_encoder.classes_):
    print(i, "=", label)

# ------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ------------------------------------------------------------

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

# Train / validation split
train_idx, val_idx = train_test_split(
    train_idx,
    test_size=0.20,
    random_state=SEED,
    stratify=y[train_idx]
)

print("\nSplit:")
print("Train      :", len(train_idx))
print("Validation :", len(val_idx))
print("Test       :", len(test_idx))

# ------------------------------------------------------------
# 4. SCALE STATIC FEATURES
# ------------------------------------------------------------

static_scaler = StandardScaler()

X_static_train = static_scaler.fit_transform(
    df.loc[train_idx, static_cols]
)

X_static_val = static_scaler.transform(
    df.loc[val_idx, static_cols]
)

X_static_test = static_scaler.transform(
    df.loc[test_idx, static_cols]
)

# ------------------------------------------------------------
# 5. SCALE TEMPORAL FEATURES
# ------------------------------------------------------------

# Scale weekly engagement using ONLY training data.
temporal_scaler = StandardScaler()

X_temporal_train_2d = temporal_scaler.fit_transform(
    df.loc[train_idx, week_cols]
)

X_temporal_val_2d = temporal_scaler.transform(
    df.loc[val_idx, week_cols]
)

X_temporal_test_2d = temporal_scaler.transform(
    df.loc[test_idx, week_cols]
)

# Convert:
# samples x 16 weeks
# into:
# samples x 16 timesteps x 1 feature
X_temporal_train = X_temporal_train_2d.reshape(
    len(train_idx), 16, 1
)

X_temporal_val = X_temporal_val_2d.reshape(
    len(val_idx), 16, 1
)

X_temporal_test = X_temporal_test_2d.reshape(
    len(test_idx), 16, 1
)

print("\nInput shapes:")
print("Static train   :", X_static_train.shape)
print("Temporal train :", X_temporal_train.shape)

# ------------------------------------------------------------
# 6. HANDLE CLASS IMBALANCE
# ------------------------------------------------------------

class_counts = np.bincount(y[train_idx])

class_weights = {
    i: len(train_idx) / (len(class_counts) * count)
    for i, count in enumerate(class_counts)
    if count > 0
}

print("\nClass weights:")
print(class_weights)

# ------------------------------------------------------------
# 7. BUILD MULTI-MODAL LSTM
# ------------------------------------------------------------

# ---- Temporal branch ----

temporal_input = Input(
    shape=(16, 1),
    name="temporal_input"
)

x_temporal = LSTM(
    64,
    return_sequences=True,
    name="lstm_temporal"
)(temporal_input)

x_temporal = Dropout(0.30)(x_temporal)

x_temporal = LSTM(
    32,
    return_sequences=False,
    name="lstm_temporal_second"
)(x_temporal)

x_temporal = Dense(
    16,
    activation="relu",
    name="temporal_embedding"
)(x_temporal)


# ---- Static multimodal branch ----

static_input = Input(
    shape=(len(static_cols),),
    name="static_input"
)

x_static = Dense(
    32,
    activation="relu",
    name="static_dense"
)(static_input)

x_static = BatchNormalization()(x_static)

x_static = Dropout(0.30)(x_static)

x_static = Dense(
    16,
    activation="relu",
    name="static_embedding"
)(x_static)


# ---- Fusion ----

fused = Concatenate(
    name="multi_modal_fusion"
)([
    x_static,
    x_temporal
])

fused = Dense(
    32,
    activation="relu",
    name="fusion_dense"
)(fused)

fused = Dropout(0.30)(fused)

fused = Dense(
    16,
    activation="relu",
    name="fusion_representation"
)(fused)

output = Dense(
    len(label_encoder.classes_),
    activation="softmax",
    name="performance_output"
)(fused)


model = Model(
    inputs=[static_input, temporal_input],
    outputs=output
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n" + "=" * 70)
print("MODEL ARCHITECTURE")
print("=" * 70)

model.summary()

# ------------------------------------------------------------
# 8. CALLBACKS
# ------------------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

# ------------------------------------------------------------
# 9. TRAIN
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

history = model.fit(
    [X_static_train, X_temporal_train],
    y[train_idx],
    validation_data=(
        [X_static_val, X_temporal_val],
        y[val_idx]
    ),
    epochs=60,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)

# ------------------------------------------------------------
# 10. TEST PREDICTION
# ------------------------------------------------------------

probabilities = model.predict(
    [X_static_test, X_temporal_test],
    verbose=0
)

y_pred = np.argmax(
    probabilities,
    axis=1
)

accuracy = accuracy_score(
    y[test_idx],
    y_pred
)

precision, recall, f1, _ = precision_recall_fscore_support(
    y[test_idx],
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y[test_idx],
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# ------------------------------------------------------------
# 11. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y[test_idx],
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Multi-Modal LSTM Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_
)

plt.yticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_
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
    "lstm_confusion_matrix.png",
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 12. TRAINING CURVES
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Multi-Modal LSTM Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()

plt.savefig(
    "lstm_accuracy_curve.png",
    dpi=300
)

plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Multi-Modal LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    "lstm_loss_curve.png",
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 13. SAVE MODEL + PREPROCESSORS
# ------------------------------------------------------------

model.save("multimodal_lstm_model.keras")

import joblib

joblib.dump(
    static_scaler,
    "lstm_static_scaler.pkl"
)

joblib.dump(
    temporal_scaler,
    "lstm_temporal_scaler.pkl"
)

joblib.dump(
    label_encoder,
    "lstm_label_encoder.pkl"
)

print("\nSaved:")
print(" - multimodal_lstm_model.keras")
print(" - lstm_static_scaler.pkl")
print(" - lstm_temporal_scaler.pkl")
print(" - lstm_label_encoder.pkl")

print("\n" + "=" * 70)
print("MULTI-MODAL LSTM COMPLETED")
print("=" * 70)
