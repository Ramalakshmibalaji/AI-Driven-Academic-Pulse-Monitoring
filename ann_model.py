import pandas as pd
import numpy as np
import joblib
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("final_ml_dataset.csv")

print("Dataset Shape:")
print(df.shape)


# ==========================================
# 2. Separate Features and Target
# ==========================================

X = df.drop("performance_level", axis=1)
y = df["performance_level"]


# ==========================================
# 3. Encode Target
# ==========================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nTarget Classes:")
print(label_encoder.classes_)


# ==========================================
# 4. Identify Feature Types
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_columns)

print("\nNumber of Numerical Features:")
print(len(numerical_columns))


# ==========================================
# 5. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# ==========================================
# 6. Scale Numerical Features
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test_scaled = scaler.transform(
    X_test[numerical_columns]
)


# ==========================================
# 7. One-Hot Encode Course Name
# ==========================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_course = encoder.fit_transform(
    X_train[categorical_columns]
)

X_test_course = encoder.transform(
    X_test[categorical_columns]
)


# ==========================================
# 8. Combine Features
# ==========================================

X_train_processed = np.hstack([
    X_train_scaled,
    X_train_course
])

X_test_processed = np.hstack([
    X_test_scaled,
    X_test_course
])

print("\nProcessed Training Shape:")
print(X_train_processed.shape)

print("\nProcessed Testing Shape:")
print(X_test_processed.shape)


# ==========================================
# 9. Create Training and Validation Sets
# ==========================================

X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train_processed,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\nBefore SMOTE - Training:")
print(pd.Series(y_train_main).value_counts())

print("\nValidation Data:")
print(pd.Series(y_val).value_counts())


# ==========================================
# 10. Apply SMOTE ONLY to Training Data
# ==========================================

smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_main,
    y_train_main
)

print("\nAfter SMOTE - Training:")
print(pd.Series(y_train_balanced).value_counts())

# ==========================================
# 10. ANN Model
# ==========================================

model = Sequential([
    Input(shape=(X_train_balanced.shape[1],)),

    Dense(
        32,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),

    Dropout(0.4),

    Dense(
        16,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),

    Dropout(0.3),

    Dense(
        3,
        activation="softmax"
    )
])


# ==========================================
# 11. Compile ANN
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 12. Display Model
# ==========================================

model.summary()


# ==========================================
# 13. Train ANN
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

history = model.fit(
    X_train_balanced,
    y_train_balanced,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

print("\nANN Training Completed Successfully.")


# ==========================================
# 14. Evaluate ANN
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test_processed,
    y_test,
    verbose=0
)

print("\nTest Loss:")
print(test_loss)

print("\nTest Accuracy:")
print(test_accuracy)


# ==========================================
# 15. Generate Predictions
# ==========================================

y_pred_prob = model.predict(
    X_test_processed,
    verbose=0
)

y_pred = np.argmax(
    y_pred_prob,
    axis=1
)


# ==========================================
# 16. Classification Report
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ==========================================
# 17. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("ANN Confusion Matrix")

plt.tight_layout()
# plt.show()

# ==========================================
# 18. Training and Validation Accuracy
# ==========================================

plt.figure(figsize=(7, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ANN Training vs Validation Accuracy")
plt.legend()

plt.tight_layout()
# plt.show()

# ==========================================
# 19. Training and Validation Loss
# ==========================================

plt.figure(figsize=(7, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN Training vs Validation Loss")
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 20. Save Trained ANN Model
# ==========================================

model.save("ann_student_performance_model.keras")

print("\nANN model saved successfully.")

# ==========================================
# 21. Save Preprocessing Objects
# ==========================================



joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "course_encoder.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nPreprocessing objects saved successfully.")

# ==========================================
# 22. Training vs Validation Accuracy
# ==========================================

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ANN Training vs Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()


# ==========================================
# 23. Training vs Validation Loss
# ==========================================

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN Training vs Validation Loss")
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# Reproducibility
# ==========================================

# Random seeds are set at the beginning of the script.


# ==========================================
# Overfitting Check
# ==========================================

final_train_accuracy = history.history["accuracy"][-1]
final_val_accuracy = history.history["val_accuracy"][-1]

final_train_loss = history.history["loss"][-1]
final_val_loss = history.history["val_loss"][-1]

print("\n===== Overfitting Check =====")

print("Final Training Accuracy:", final_train_accuracy)
print("Final Validation Accuracy:", final_val_accuracy)

print("Final Training Loss:", final_train_loss)
print("Final Validation Loss:", final_val_loss)

print(
    "Training-Validation Accuracy Gap:",
    final_train_accuracy - final_val_accuracy
)

# ==========================================
# 24. Class Bias Check
# ==========================================

print("\n===== Class Bias Check =====")

print("\nActual Test Class Distribution:")
print(
    pd.Series(
        y_test
    ).map(
        dict(enumerate(label_encoder.classes_))
    ).value_counts()
)

print("\nPredicted Test Class Distribution:")
print(
    pd.Series(
        y_pred
    ).map(
        dict(enumerate(label_encoder.classes_))
    ).value_counts()
)

# ==========================================
# 25. Save Final ANN Results
# ==========================================

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate final metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

# Create result table
results = pd.DataFrame({
    "Metric": [
        "Test Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1 Score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

# Save results
results.to_csv(
    "ann_final_results.csv",
    index=False
)

print("\n===== Final ANN Results =====")
print(results)

print("\nANN results saved successfully.")