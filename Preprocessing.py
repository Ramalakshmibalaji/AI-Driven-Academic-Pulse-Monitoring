import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE


# Load final ML dataset
df = pd.read_csv("final_ml_dataset.csv")

print("Original Shape:")
print(df.shape)

# Separate input features and target
X = df.drop("performance_level", axis=1)
y = df["performance_level"]

print("\nX Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Classes:")
print(y.unique())


label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nEncoded Target Classes:")
for original, encoded in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(original, "->", encoded)

print("\nEncoded Target Distribution:")
print(pd.Series(y_encoded).value_counts())

# Separate categorical and numerical features

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical Features:")
print(categorical_columns)

print("\nNumber of Numerical Features:")
print(len(numerical_columns))

print("\nNumber of Categorical Features:")
print(len(categorical_columns))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(pd.Series(y_train).value_counts())

print("\nTesting Target Distribution:")
print(pd.Series(y_test).value_counts())

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled = scaler.transform(X_test[numerical_columns])

print("\nScaled Training Shape:")
print(X_train_scaled.shape)

print("\nScaled Testing Shape:")
print(X_test_scaled.shape)


# One-hot encode course_name
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

print("\nEncoded Course Training Shape:")
print(X_train_course.shape)

print("\nEncoded Course Testing Shape:")
print(X_test_course.shape)

# Combine numerical and categorical features
X_train_processed = np.hstack([
    X_train_scaled,
    X_train_course
])

X_test_processed = np.hstack([
    X_test_scaled,
    X_test_course
])

print("\nFinal Processed Training Shape:")
print(X_train_processed.shape)

print("\nFinal Processed Testing Shape:")
print(X_test_processed.shape)


# Apply SMOTE only on training data
smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed,
    y_train
)

print("\nBefore SMOTE:")
print(pd.Series(y_train).value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

print("\nBalanced Training Shape:")
print(X_train_balanced.shape)

