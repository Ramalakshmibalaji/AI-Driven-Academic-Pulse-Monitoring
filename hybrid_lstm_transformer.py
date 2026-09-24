import pandas as pd
import numpy as np
import tensorflow as tf
import random
import joblib
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
    Input, LSTM, Dense, Dropout, BatchNormalization,
    Concatenate, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# HYBRID MULTI-MODAL LSTM + TRANSFORMER
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 72)
print("HYBRID MULTI-MODAL LSTM + TRANSFORMER")
print("ACADEMIC PERFORMANCE PREDICTION")
print("=" * 72)

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
print("Temporal sequence:", len(week_cols), "weeks")

# ------------------------------------------------------------
# 2. TARGET
# ------------------------------------------------------------

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target])

print("\nTarget classes:")
for i, c in enumerate(label_encoder.classes_):
    print(f"{i} = {c}")

# ------------------------------------------------------------
# 3. TRAIN / VALIDATION / TEST
# ------------------------------------------------------------

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

train_idx, val_idx = train_test_split(
    train_idx,
    test_size=0.20,
    random_state=SEED,
    stratify=y[train_idx]
)

print("\nData split:")
print("Train      :", len(train_idx))
print("Validation :", len(val_idx))
print("Test       :", len(test_idx))

# ------------------------------------------------------------
# 4. STATIC FEATURE SCALING
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
# 5. TEMPORAL FEATURE SCALING
# ------------------------------------------------------------

temporal_scaler = StandardScaler()

X_week_train = temporal_scaler.fit_transform(
    df.loc[train_idx, week_cols]
)

X_week_val = temporal_scaler.transform(
    df.loc[val_idx, week_cols]
)

X_week_test = temporal_scaler.transform(
    df.loc[test_idx, week_cols]
)

# Genuine temporal representation:
# samples x 16 weeks x 1 feature

X_temporal_train = X_week_train.reshape(-1, 16, 1)
X_temporal_val = X_week_val.reshape(-1, 16, 1)
X_temporal_test = X_week_test.reshape(-1, 16, 1)

print("\nInput shapes:")
print("Static   :", X_static_train.shape)
print("Temporal :", X_temporal_train.shape)

# ------------------------------------------------------------
# 6. MODERATE CLASS WEIGHTS
# ------------------------------------------------------------

# Full inverse-frequency weighting from the first LSTM
# over-emphasized the minority class.
# Square-root weighting gives a more moderate correction.

counts = np.bincount(y[train_idx])
n = len(train_idx)
k = len(counts)

class_weights = {
    i: float(np.sqrt(n / (k * counts[i])))
    for i in range(k)
}

print("\nModerate class weights:")
print(class_weights)

# ------------------------------------------------------------
# 7. TEMPORAL BRANCH
# ------------------------------------------------------------

temporal_input = Input(
    shape=(16, 1),
    name="temporal_input"
)

# LSTM captures sequential learning behaviour.
x = LSTM(
    64,
    return_sequences=True,
    name="lstm_sequence"
)(temporal_input)

x = Dropout(0.25, name="temporal_dropout")(x)

# Transformer attention captures relationships between
# different weeks in the learning sequence.
attention = MultiHeadAttention(
    num_heads=4,
    key_dim=16,
    dropout=0.15,
    name="weekly_multi_head_attention"
)(
    query=x,
    value=x,
    key=x
)

# Residual connection + normalization
x = Add(name="attention_residual")([x, attention])
x = LayerNormalization(name="attention_layer_norm")(x)

# Feed-forward transformation
ff = Dense(
    64,
    activation="relu",
    name="transformer_feed_forward"
)(x)

ff = Dropout(0.20)(ff)

x = Add(name="feed_forward_residual")([x, ff])
x = LayerNormalization(name="feed_forward_norm")(x)

# Second LSTM summarizes the attended temporal representation.
x = LSTM(
    32,
    return_sequences=False,
    name="lstm_summary"
)(x)

x = Dense(
    16,
    activation="relu",
    name="temporal_embedding"
)(x)

# ------------------------------------------------------------
# 8. STATIC MULTI-MODAL BRANCH
# ------------------------------------------------------------

static_input = Input(
    shape=(len(static_cols),),
    name="static_input"
)

s = Dense(
    32,
    activation="relu",
    name="static_dense"
)(static_input)

s = BatchNormalization(name="static_batch_norm")(s)
s = Dropout(0.25, name="static_dropout")(s)

s = Dense(
    16,
    activation="relu",
    name="static_embedding"
)(s)

# ------------------------------------------------------------
# 9. MULTI-MODAL FEATURE FUSION
# ------------------------------------------------------------

fused = Concatenate(
    name="multi_modal_feature_fusion"
)([s, x])

fused = Dense(
    32,
    activation="relu",
    name="fusion_dense"
)(fused)

fused = Dropout(0.30, name="fusion_dropout")(fused)

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
    outputs=output,
    name="Hybrid_MultiModal_LSTM_Transformer"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n" + "=" * 72)
print("HYBRID MODEL ARCHITECTURE")
print("=" * 72)

model.summary()

# ------------------------------------------------------------
# 10. CALLBACKS
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
# 11. TRAIN
# ------------------------------------------------------------

print("\n" + "=" * 72)
print("TRAINING HYBRID MODEL")
print("=" * 72)

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
# 12. TEST
# ------------------------------------------------------------

probabilities = model.predict(
    [X_static_test, X_temporal_test],
    verbose=0
)

y_pred = np.argmax(probabilities, axis=1)

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

print("\n" + "=" * 72)
print("FINAL HYBRID TEST RESULTS")
print("=" * 72)

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
# 13. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y[test_idx],
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("Hybrid Multi-Modal LSTM-Transformer Confusion Matrix")
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
            j, i, cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()
plt.savefig("hybrid_lstm_transformer_confusion_matrix.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 14. TRAINING CURVES
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Hybrid LSTM-Transformer Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("hybrid_lstm_transformer_accuracy.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Hybrid LSTM-Transformer Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("hybrid_lstm_transformer_loss.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 15. SAVE MODEL AND PREPROCESSORS
# ------------------------------------------------------------

model.save("hybrid_multimodal_lstm_transformer.keras")

joblib.dump(
    static_scaler,
    "hybrid_static_scaler.pkl"
)

joblib.dump(
    temporal_scaler,
    "hybrid_temporal_scaler.pkl"
)

joblib.dump(
    label_encoder,
    "hybrid_label_encoder.pkl"
)

print("\nSaved:")
print(" - hybrid_multimodal_lstm_transformer.keras")
print(" - hybrid_static_scaler.pkl")
print(" - hybrid_temporal_scaler.pkl")
print(" - hybrid_label_encoder.pkl")

print("\n" + "=" * 72)
print("HYBRID MULTI-MODAL LSTM-TRANSFORMER COMPLETED")
print("=" * 72)

model.save("hybrid_lstm_transformer.keras")
print("Hybrid LSTM-Transformer model saved successfully.")
