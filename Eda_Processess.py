import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_file = "sql-20260914-231810.csv"

df = pd.read_csv(data_file)

# print(df.head())
# print("Dataset Shape:")
# print(df.shape)

# print("\nColumn Names:")
# print(df.columns.tolist())

# print("\nData Types:")
# print(df.dtypes)

# print("\nMissing Values:")
# print(df.isnull().sum())

# print("\nDuplicate Rows:")
# print(df.duplicated().sum())

# print("\nUnique Values:")
# print(df.nunique())

# print("\nDescriptive Statistics:")
# print(df.describe())

# print("\nSkewness:")
# print(df.select_dtypes(include='number').skew())



# numeric_columns = df.select_dtypes(include='number').columns

# for col in numeric_columns:
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)

#     IQR = Q3 - Q1

#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     outliers = df[
#         (df[col] < lower_bound) |
#         (df[col] > upper_bound)
#     ]

#     print(f"{col}: {len(outliers)} outliers")


# numeric_df = df.select_dtypes(include='number')

# correlation = numeric_df.corr()

# print("\nCorrelation Matrix:")
# print(correlation)

# plt.figure(figsize=(20, 16))

# sns.heatmap(
#     correlation,
#     cmap="coolwarm",
#     center=0
# )

# plt.title("Correlation Heatmap")
# plt.tight_layout()
# plt.show()

# print("\nPerformance-related Columns:")
# performance_columns = [
#     'mock_exam_score',
#     'portal_based_exam_score',
#     'model_exam_score',
#     'pba_score',
#     'cia_score',
#     'end_semester_score',
#     'in_between_assessment_grade',
#     'model_exam_assessment_score',
#     'pba_assessment_score',
#     'end_semester_assessment_score'
# ]

# print(df[performance_columns].describe())

# print("\nZero Values in Performance Columns:")

# for col in performance_columns:
#     zero_count = (df[col] == 0).sum()
#     non_zero_count = (df[col] != 0).sum()

#     print(f"{col}:")
#     print(f"  Zero     : {zero_count}")
#     print(f"  Non-Zero : {non_zero_count}")

# print("\nPerformance Column Distribution:")

# for col in performance_columns:
#     print(f"\n{col}")
#     print(df[col].value_counts().head(10))

# performance_base = [
#     'mock_exam_score',
#     'portal_based_exam_score',
#     'model_exam_score',
#     'pba_score',
#     'cia_score',
#     'end_semester_score'
# ]

# df['performance_nonzero_count'] = (
#     df[performance_base] > 0
# ).sum(axis=1)

# print("\nPerformance Non-Zero Count Distribution:")
# print(df['performance_nonzero_count'].value_counts().sort_index())

# print("\nPerformance Non-Zero Count by Course:")

# course_performance = (
#     df.groupby('course_name')['performance_nonzero_count']
#     .value_counts()
#     .unstack(fill_value=0)
# )

# print(course_performance)


# numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# print("\nNumerical Columns:")
# print(numeric_columns.tolist())

# print("\nTotal Numerical Columns:")
# print(len(numeric_columns))


# # Categorical Columns
# categorical_columns = df.select_dtypes(include=['object', 'str']).columns

# print("\nCategorical Columns:")
# print(categorical_columns.tolist())

# print("\nTotal Categorical Columns:")
# print(len(categorical_columns))

# print("\nCategorical Column Unique Values:")

# for col in categorical_columns:
#     print(f"\n{col}:")
#     print(df[col].nunique())
#     print(df[col].value_counts().head(10))

# print("\nCourse-wise Record Count:")

# course_counts = df['course_name'].value_counts()

# print(course_counts)

# print("\nCourse-wise Mean Analysis:")

# course_mean = df.groupby('course_name')[
#     [
#         'login_count',
#         'active_days',
#         'course_access_count',
#         'quiz_average_score',
#         'time_spent_on_course_minutes',
#         'completed_modules'
#     ]
# ].mean()

# print(course_mean)


# df[numeric_columns].hist(
#     figsize=(20, 25),
#     bins=30
# )

# plt.tight_layout()
# plt.show()

# important_cols = [
#     'login_count',
#     'active_days',
#     'course_access_count',
#     'page_view_count',
#     'total_practice',
#     'quiz_attempt_count',
#     'quiz_average_score',
#     'time_spent_on_course_minutes',
#     'study_frequency_per_week',
#     'completed_modules',
#     'study_interval_regularity'
# ]

# df[important_cols].hist(
#     figsize=(18, 15),
#     bins=30
# )

# plt.tight_layout()
# plt.show()

# print("\nMissing Values:")
# print(df.isnull().sum())

# print("\nMissing Value Percentage:")
# print((df.isnull().sum() / len(df)) * 100)

# print(df[df["login_frequency"].isnull()][
#     ["login_count", "active_days", "login_frequency", "login_consistency"]
# ])

df["login_frequency"] = df["login_frequency"].fillna(0)
df["login_consistency"] = df["login_consistency"].fillna(0)

print("\nMissing Values After Handling:")
print(df[["login_frequency", "login_consistency"]].isnull().sum())

# print("\nDuplicate Rows:")
# print(df.duplicated().sum())

# print("\nNegative Values Check:")
# print((df[numeric_columns] < 0).sum())

# print("\nRange Validation:")

# range_cols = [
#     "module_completion_rate",
#     "quiz_completion_rate",
#     "quiz_average_score",
#     "login_consistency",
#     "study_interval_regularity"
# ]

# for col in range_cols:
#     print(f"\n{col}")
#     print("Min:", df[col].min())
#     print("Max:", df[col].max())

# print("\nQuiz Scores Above 100:")
# print(df[df["quiz_average_score"] > 100][
#     ["course_name", "quiz_average_score"]
# ])

# print("\nCount of Quiz Scores Above 100:")
# print((df["quiz_average_score"] > 100).sum())

# performance_columns = [
#     'mock_exam_score',
#     'portal_based_exam_score',
#     'model_exam_score',
#     'pba_score',
#     'cia_score',
#     'end_semester_score',
#     'in_between_assessment_grade',
#     'model_exam_assessment_score',
#     'pba_assessment_score',
#     'end_semester_assessment_score'
# ]

# df['performance_nonzero_count'] = (df[performance_columns] > 0).sum(axis=1)

# target_distribution = df["performance_nonzero_count"].value_counts().sort_index()

# target_percentage = (
#     df["performance_nonzero_count"]
#     .value_counts(normalize=True)
#     .sort_index() * 100
# )
# # target_percentage=(
# #     df["performance_nonzero_count"]
# #     .value_counts(normalize=True)
# #     .sort_index()*100
# # )

# print("Target Distribution:")
# print(target_distribution)

# print("\nTarget Distribution Percentage:")
# print(target_percentage.round(2))



# course_counts = df["course_name"].value_counts().head(10)

# plt.figure(figsize=(10,6))

# course_counts.sort_values().plot(
#     kind="barh"
# )

# plt.title("Top 10 Courses by Record Count")
# plt.xlabel("Number of Records")
# plt.ylabel("Course Name")

# plt.tight_layout()
# plt.show()


# ============================================
# STEP 1: DUPLICATE FEATURE CHECK
# ============================================

print("\n" + "="*60)
print("DUPLICATE FEATURE CHECK")
print("="*60)

# Check for exact duplicate columns
duplicate_columns = []

for i in range(len(df.columns)):
    for j in range(i + 1, len(df.columns)):
        if df.iloc[:, i].equals(df.iloc[:, j]):
            duplicate_columns.append(
                (df.columns[i], df.columns[j])
            )

if duplicate_columns:
    print("\nExact duplicate columns found:")
    for col1, col2 in duplicate_columns:
        print(f"  {col1}  <-->  {col2}")
else:
    print("\nNo exact duplicate columns found.")


# ============================================
# STEP 2: HIGHLY CORRELATED FEATURE CHECK
# ============================================

print("\n" + "="*60)
print("HIGHLY CORRELATED FEATURE CHECK")
print("="*60)

# Select numerical columns
numeric_df = df.select_dtypes(include="number")

# Calculate correlation matrix
corr_matrix = numeric_df.corr()

# Find feature pairs with correlation >= 0.90
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):

        corr_value = corr_matrix.iloc[i, j]

        if abs(corr_value) >= 0.90:
            high_corr_pairs.append(
                (
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_value
                )
            )

if high_corr_pairs:

    print("\nHighly correlated feature pairs (|r| >= 0.90):")

    for col1, col2, corr_value in high_corr_pairs:
        print(
            f"{col1}  <-->  {col2}  :  {corr_value:.4f}"
        )

else:
    print("\nNo highly correlated feature pairs found.")

# ============================================
# STEP 3: CHECK SUSPICIOUS HIGH CORRELATIONS
# ============================================

print("\n" + "="*60)
print("SUSPICIOUS FEATURE COMPARISON")
print("="*60)

print("\n1. PBA SCORE vs PBA ASSESSMENT SCORE")
print(df[[
    "pba_score",
    "pba_assessment_score"
]].head(20))

print("\n2. CIA SCORE vs IN-BETWEEN ASSESSMENT GRADE")
print(df[[
    "cia_score",
    "in_between_assessment_grade"
]].head(20))

print("\n3. WEEK 12 vs WEEK 16 ENGAGEMENT")
print(df[[
    "week_1_12_engagement",
    "week_1_16_engagement"
]].head(20))

# ============================================
# STEP 4: IDENTIFIER / COURSE CHECK
# ============================================

print("\n" + "="*60)
print("IDENTIFIER AND COURSE CHECK")
print("="*60)

print("\nUnique values:")

print("course_id  :", df["course_id"].nunique())
print("course_name:", df["course_name"].nunique())
print("userid     :", df["userid"].nunique())
print("username   :", df["username"].nunique())

print("\nCourse-wise record count:")
print(df["course_name"].value_counts())

# ============================================
# STEP 5: ACADEMIC PERFORMANCE SCORE CHECK
# ============================================

performance_cols = [
    "mock_exam_score",
    "portal_based_exam_score",
    "model_exam_score",
    "pba_score",
    "cia_score",
    "end_semester_score",
    "in_between_assessment_grade"
]

print("\n" + "="*60)
print("ACADEMIC PERFORMANCE SCORE CHECK")
print("="*60)

print("\nBasic statistics:")
print(df[performance_cols].describe().T)

print("\nZero count:")
for col in performance_cols:
    print(f"{col}: {(df[col] == 0).sum()}")

print("\nNon-zero count:")
for col in performance_cols:
    print(f"{col}: {(df[col] != 0).sum()}")

# ============================================
# STEP 6: PERFORMANCE SCORES BY COURSE
# ============================================

# performance_cols = [
#     "mock_exam_score",
#     "portal_based_exam_score",
#     "model_exam_score",
#     "pba_score",
#     "cia_score",
#     "end_semester_score",
#     "in_between_assessment_grade"
# ]

# print("\n" + "="*60)
# print("PERFORMANCE SCORES BY COURSE")
# print("="*60)

# for col in performance_cols:

#     print(f"\n--- {col} ---")

#     result = df.groupby("course_name")[col].agg(
#         non_zero_count=lambda x: (x != 0).sum(),
#         max_score="max",
#         mean_score="mean"
#     ).sort_values("non_zero_count", ascending=False)

#     print(result.to_string())

# ============================================
# STEP 7: VALID PERFORMANCE RECORDS
# ============================================

performance_cols = [
    "mock_exam_score",
    "portal_based_exam_score",
    "model_exam_score",
    "pba_score",
    "cia_score",
    "end_semester_score",
    "in_between_assessment_grade"
]

df["performance_available"] = (
    df[performance_cols].sum(axis=1) > 0
)

print("\n" + "="*60)
print("VALID PERFORMANCE RECORD CHECK")
print("="*60)

print("\nPerformance available:")
print(df["performance_available"].value_counts())

print("\nCourse-wise performance availability:")

result = df.groupby("course_name")["performance_available"].agg(
    total_records="count",
    performance_available="sum"
)

print(result.to_string())

# ============================================
# STEP 8: COURSE-WISE NORMALIZED PERFORMANCE
# ============================================

performance_cols = [
    "mock_exam_score",
    "portal_based_exam_score",
    "model_exam_score",
    "pba_score",
    "cia_score",
    "end_semester_score",
    "in_between_assessment_grade"
]

# Keep only records where at least one performance score is available
model_df = df[df["performance_available"] == True].copy()

print("\n" + "="*60)
print("MODEL DATASET")
print("="*60)

print("Total records:", len(model_df))
print("Total courses:", model_df["course_name"].nunique())


# --------------------------------------------
# Normalize each performance score within course
# --------------------------------------------

normalized_cols = []

for col in performance_cols:

    normalized_col = col + "_normalized"

    model_df[normalized_col] = (
        model_df.groupby("course_name")[col]
        .transform(
            lambda x: (x / x.max() * 100) if x.max() > 0 else 0
        )
    )

    normalized_cols.append(normalized_col)


# --------------------------------------------
# Create unified performance score
# --------------------------------------------

model_df["performance_score"] = (
    model_df[normalized_cols]
    .replace(0, pd.NA)
    .mean(axis=1)
    .fillna(0)
)


print("\nPerformance Score Statistics:")
print(model_df["performance_score"].describe())


print("\nPerformance Score Sample:")
print(
    model_df[
        ["course_name", "userid", "performance_score"]
    ].head(20).to_string(index=False)
)

# ============================================
# STEP 8A: CHECK AND FIX PERFORMANCE SCORE TYPE
# ============================================

print("\n" + "="*60)
print("PERFORMANCE SCORE TYPE CHECK")
print("="*60)

print("\nData type:")
print(model_df["performance_score"].dtype)

print("\nConvert performance_score to numeric:")

model_df["performance_score"] = pd.to_numeric(
    model_df["performance_score"],
    errors="coerce"
)

print(model_df["performance_score"].dtype)

print("\nMissing values after conversion:")
print(model_df["performance_score"].isna().sum())

print("\nPerformance Score Statistics:")
print(model_df["performance_score"].describe())

print("\nMinimum Performance Score:",
      model_df["performance_score"].min())

print("Maximum Performance Score:",
      model_df["performance_score"].max())

print("Mean Performance Score:",
      model_df["performance_score"].mean())

print("Median Performance Score:",
      model_df["performance_score"].median())

print("\n" + "="*60)
print("PERFORMANCE CLASS DISTRIBUTION CHECK")
print("="*60)

# Proposed academic performance categories
model_df["performance_level"] = pd.cut(
    model_df["performance_score"],
    bins=[-1, 49.99, 74.99, 100],
    labels=["Low", "Medium", "High"]
)

print("\nPerformance Level Distribution:")
print(model_df["performance_level"].value_counts())

print("\nPerformance Level Percentage:")
print(
    model_df["performance_level"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nPerformance Level Distribution:")
print(
    model_df["performance_level"]
    .value_counts()
    .sort_index()
)

print("\n" + "="*60)
print("TARGET LEAKAGE CHECK")
print("="*60)

performance_columns = [
    "mock_exam_score",
    "portal_based_exam_score",
    "model_exam_score",
    "pba_score",
    "cia_score",
    "end_semester_score",
    "in_between_assessment_grade",
    "model_exam_assessment_score",
    "pba_assessment_score",
    "end_semester_assessment_score",
    "performance_score",
    "performance_level"
]

print("\nTarget-related columns:")
for col in performance_columns:
    if col in model_df.columns:
        print("-", col)

print("\nShape of model dataset:")
print(model_df.shape)


print("\n" + "="*60)
print("CREATING LEAKAGE-FREE ML DATASET")
print("="*60)

# Columns used to calculate the target
target_related_columns = [
    "mock_exam_score",
    "portal_based_exam_score",
    "model_exam_score",
    "pba_score",
    "cia_score",
    "end_semester_score",
    "in_between_assessment_grade",
    "model_exam_assessment_score",
    "pba_assessment_score",
    "end_semester_assessment_score",
    "performance_score"
]

# Remove target-related columns
ml_df = model_df.drop(
    columns=target_related_columns,
    errors="ignore"
).copy()

print("\nOriginal model_df shape:")
print(model_df.shape)

print("\nLeakage-free ML dataset shape:")
print(ml_df.shape)

print("\nRemaining columns:")
print(ml_df.columns.tolist())

print("\nTarget distribution:")
print(ml_df["performance_level"].value_counts())

print("\nTarget percentage:")
print(
    ml_df["performance_level"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n" + "="*60)
print("FINAL FEATURE CLEANING")
print("="*60)

# Remove identifiers and target-leakage / assessment-derived columns
remove_columns = [

    "course_id",
    "userid",
    "username",
    "firstname",
    "lastname",
    "student_name",

    "mock_exam_score_normalized",
    "portal_based_exam_score_normalized",
    "model_exam_score_normalized",
    "pba_score_normalized",
    "cia_score_normalized",
    "end_semester_score_normalized",
    "in_between_assessment_grade_normalized",

    "practice_submissions",
    "home_exercise_submissions",

    "performance_available"
    "home_exercise_submissions",
    "home_exercises_attempted",
    "practice_submissions",
    "model_exam_assessment_score",
    "end_semester_assessment_score",  

]

# Create final ML dataframe
final_ml_df = ml_df.drop(
    columns=remove_columns,
    errors="ignore"
).copy()

print("\nOriginal ML dataset shape:")
print(ml_df.shape)

print("\nFinal ML dataset shape:")
print(final_ml_df.shape)

print("\nFinal columns:")
print(final_ml_df.columns.tolist())

print("\nTarget distribution:")
print(final_ml_df["performance_level"].value_counts())

print("\nMissing values:")
print(final_ml_df.isnull().sum().sum())

print("\nDuplicate rows:")
print(final_ml_df.duplicated().sum())


# ==========================================
# Create Student Prediction Dataset
# ==========================================

student_prediction_df = ml_df[
    ["userid"] + [
        col for col in final_ml_df.columns
        if col != "performance_level"
    ] + ["performance_level"]
].copy()

student_prediction_df.to_csv(
    "student_prediction_data.csv",
    index=False
)

print("\nStudent prediction dataset saved successfully.")
print("Shape:", student_prediction_df.shape)

# Save final ML dataset

final_ml_df.to_csv(
    "final_ml_dataset.csv",
    index=False
)

print("\nFinal ML dataset saved successfully.")


print("\n" + "="*60)
print("STEP 9 - FEATURES AND TARGET SEPARATION")
print("="*60)

# Features
X = final_ml_df.drop(
    columns=["performance_level"]
).copy()

# Target
y = final_ml_df["performance_level"].copy()

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nFeature columns:")
print(X.columns.tolist())
print("\nPerformance Available:")
print(df["performance_available"].value_counts(dropna=False))


print("\nFeature data types:")
print(X.dtypes)

print("\nTarget classes:")
print(y.value_counts())

print("\nTarget percentages:")
print(
    y.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nCategorical columns:")
print(X.select_dtypes(include=["object", "category"]).columns.tolist())

print("\nNumerical columns:")
print(X.select_dtypes(include=["number"]).columns.tolist())


print("\n" + "="*60)
print("STEP 10 - TRAIN TEST SPLIT")
print("="*60)

from sklearn.model_selection import train_test_split

# Separate target
X = final_ml_df.drop(columns=["performance_level"]).copy()
y = final_ml_df["performance_level"].copy()

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:")
print(X_train.shape)

print("\nTesting set shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTraining target percentage:")
print(
    y_train.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nTesting target percentage:")
print(
    y_test.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n" + "="*60)
print("STEP 11 - COURSE NAME ENCODING")
print("="*60)

from sklearn.preprocessing import OneHotEncoder

# Create encoder
course_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Fit ONLY on training data
course_encoder.fit(
    X_train[["course_name"]]
)

# Transform train and test
X_train_course = course_encoder.transform(
    X_train[["course_name"]]
)

X_test_course = course_encoder.transform(
    X_test[["course_name"]]
)

print("\nNumber of course categories learned:")
print(len(course_encoder.categories_[0]))

print("\nTraining encoded shape:")
print(X_train_course.shape)

print("\nTesting encoded shape:")
print(X_test_course.shape)

print("\nCourse categories:")
print(course_encoder.categories_[0])

# ============================================================
# STEP 12 - DETAILED EDA VISUALIZATION
# ============================================================

print("\n" + "="*60)
print("STEP 12 - DETAILED EDA VISUALIZATION")
print("="*60)

import os

os.makedirs("eda_outputs", exist_ok=True)

# Use final leakage-free dataset
eda_df = final_ml_df.copy()

# ------------------------------------------------------------
# 1. Performance Level Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.countplot(data=eda_df, x="performance_level")
plt.title("Performance Level Distribution")
plt.xlabel("Performance Level")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("eda_outputs/01_performance_distribution.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 2. Performance Score Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(
    model_df["performance_score"],
    bins=30,
    kde=True
)
plt.title("Performance Score Distribution")
plt.xlabel("Performance Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("eda_outputs/02_performance_score_distribution.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 3. Correlation Heatmap - Behavioural Features
# ------------------------------------------------------------

behaviour_cols = [
    "login_count",
    "active_days",
    "login_frequency",
    "login_consistency",
    "course_access_count",
    "module_completion_rate",
    "page_view_count",
    "class_exercise_submissions",
    "class_exercises_attempted",
    "total_practice",
    "exercises_attempted",
    "quiz_attempt_count",
    "quiz_completion_rate",
    "quiz_average_score",
    "quiz_view_count",
    "study_session_count",
    "time_spent_on_course_minutes",
    "study_frequency_per_week",
    "completed_modules",
    "study_interval_regularity"
]

corr = eda_df[behaviour_cols].corr()

plt.figure(figsize=(16, 13))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap of Behavioural Features")
plt.tight_layout()
plt.savefig("eda_outputs/03_behaviour_correlation_heatmap.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 4. Engagement Features vs Performance
# ------------------------------------------------------------

engagement_cols = [
    "login_count",
    "course_access_count",
    "page_view_count",
    "total_practice",
    "quiz_attempt_count",
    "time_spent_on_course_minutes"
]

for col in engagement_cols:

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=eda_df,
        x="performance_level",
        y=col
    )

    plt.title(f"{col} vs Performance Level")
    plt.xlabel("Performance Level")
    plt.ylabel(col)

    plt.tight_layout()
    plt.savefig(
        f"eda_outputs/04_{col}_vs_performance.png",
        dpi=300
    )
    plt.show()


# ------------------------------------------------------------
# 5. Learning / Assessment Behaviour vs Performance
# ------------------------------------------------------------

learning_cols = [
    "module_completion_rate",
    "quiz_completion_rate",
    "quiz_average_score",
    "completed_modules",
    "exercises_attempted"
]

for col in learning_cols:

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=eda_df,
        x="performance_level",
        y=col
    )

    plt.title(f"{col} vs Performance Level")
    plt.xlabel("Performance Level")
    plt.ylabel(col)

    plt.tight_layout()
    plt.savefig(
        f"eda_outputs/05_{col}_vs_performance.png",
        dpi=300
    )
    plt.show()


# ------------------------------------------------------------
# 6. Weekly Engagement Trend
# ------------------------------------------------------------

week_cols = [f"week_{i}" for i in range(1, 17)]

weekly_mean = eda_df[week_cols].mean()

plt.figure(figsize=(12, 5))

plt.plot(
    range(1, 17),
    weekly_mean.values,
    marker="o"
)

plt.title("Average Weekly Student Engagement")
plt.xlabel("Week")
plt.ylabel("Average Engagement")

plt.xticks(range(1, 17))

plt.tight_layout()
plt.savefig(
    "eda_outputs/06_weekly_engagement_trend.png",
    dpi=300
)
plt.show()


# ------------------------------------------------------------
# 7. Weekly Engagement by Performance Level
# ------------------------------------------------------------

weekly_by_level = (
    eda_df.groupby("performance_level")[week_cols]
    .mean()
    .T
)

plt.figure(figsize=(12, 6))

for level in ["Low", "Medium", "High"]:
    if level in weekly_by_level.columns:
        plt.plot(
            range(1, 17),
            weekly_by_level[level],
            marker="o",
            label=level
        )

plt.title("Weekly Engagement Trend by Performance Level")
plt.xlabel("Week")
plt.ylabel("Average Engagement")
plt.xticks(range(1, 17))
plt.legend()

plt.tight_layout()
plt.savefig(
    "eda_outputs/07_weekly_engagement_by_performance.png",
    dpi=300
)
plt.show()


# ------------------------------------------------------------
# 8. Distribution of Important Behavioural Features
# ------------------------------------------------------------

important_cols = [
    "login_count",
    "active_days",
    "login_frequency",
    "login_consistency",
    "course_access_count",
    "page_view_count",
    "quiz_average_score",
    "study_session_count",
    "time_spent_on_course_minutes",
    "study_frequency_per_week",
    "study_interval_regularity"
]

for col in important_cols:

    plt.figure(figsize=(7, 5))

    sns.histplot(
        eda_df[col],
        bins=30,
        kde=True
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(
        f"eda_outputs/08_distribution_{col}.png",
        dpi=300
    )
    plt.show()


# ------------------------------------------------------------
# 9. Outlier Analysis
# ------------------------------------------------------------

print("\nOUTLIER ANALYSIS")

for col in important_cols:

    Q1 = eda_df[col].quantile(0.25)
    Q3 = eda_df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = (
        (eda_df[col] < lower) |
        (eda_df[col] > upper)
    ).sum()

    print(f"{col}: {count} outliers")


# ------------------------------------------------------------
# 10. Statistical Summary
# ------------------------------------------------------------

print("\nSTATISTICAL SUMMARY")

summary = eda_df[behaviour_cols].describe().T

summary["skewness"] = eda_df[behaviour_cols].skew()

print(summary)

summary.to_csv(
    "eda_outputs/09_statistical_summary.csv"
)


# ------------------------------------------------------------
# 11. Performance Level Behavioural Mean
# ------------------------------------------------------------

level_mean = (
    eda_df.groupby("performance_level")[important_cols]
    .mean()
)

print("\nBEHAVIOURAL FEATURES BY PERFORMANCE LEVEL")

print(level_mean.round(2))

level_mean.to_csv(
    "eda_outputs/10_behaviour_by_performance.csv"
)


print("\n" + "="*60)
print("DETAILED EDA COMPLETED")
print("="*60)
print("EDA outputs saved inside: eda_outputs/")

score_cols = [
    "mock_exam_score",
    "portal_based_exam_score",
    "model_exam_score",
    "pba_score",
    "cia_score",
    "end_semester_score"
]

print("\n===== RAW SCORE RANGE AUDIT =====")
for col in score_cols:
    if col in df.columns:
        print(f"\n{col}")
        print("Min :", df[col].min())
        print("Max :", df[col].max())
        print("Count > 100 :", (df[col] > 100).sum())