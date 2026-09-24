# ============================================================
# CLEANED PUBLICATION DATASET CREATION
# Project:
# AI-DRIVEN ACADEMIC PULSE MONITORING THROUGH
# MULTI-MODAL LEARNING FEATURE FUSION
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD RAW DATASET
# ============================================================

input_file = "sql-20260914-231810.csv"
output_file = "cleaned_publication_dataset.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("PUBLICATION DATASET CLEANING")
print("=" * 70)

print("\nOriginal Raw Dataset Shape:")
print(df.shape)

print("\nOriginal Number of Columns:")
print(len(df.columns))


# ============================================================
# 2. KEEP RAW DATA UNTOUCHED
# ============================================================

publication_df = df.copy()


# ============================================================
# 3. REMOVE PERSONAL / IDENTIFIABLE INFORMATION
# ============================================================

privacy_columns = [
    "userid",
    "username",
    "firstname",
    "lastname",
    "student_name"
]

publication_df = publication_df.drop(
    columns=privacy_columns,
    errors="ignore"
)

print("\nAfter removing personal identifiers:")
print(publication_df.shape)


# ============================================================
# 4. REMOVE COURSE IDENTIFIER
# ============================================================

# course_id can act as an institution/course-specific identifier.
# course_name is retained for analysis.

publication_df = publication_df.drop(
    columns=["course_id"],
    errors="ignore"
)


# ============================================================
# 5. REMOVE EXACT DUPLICATE / REDUNDANT FEATURES
# ============================================================

redundant_columns = [
    # All-zero / redundant home exercise features
    "home_exercise_submissions",
    "home_exercises_attempted",

    # Exact duplicate feature
    "practice_submissions",

    # Duplicate assessment representations
    "model_exam_assessment_score",
    "end_semester_assessment_score"
]

publication_df = publication_df.drop(
    columns=redundant_columns,
    errors="ignore"
)


# ============================================================
# 6. REMOVE NORMALIZED INTERMEDIATE FEATURES
# ============================================================

# These are intermediate features generated during modelling.
# Raw assessment values are retained only if required for
# publication analysis.

intermediate_columns = [
    "mock_exam_score_normalized",
    "portal_based_exam_score_normalized",
    "model_exam_score_normalized",
    "pba_score_normalized",
    "cia_score_normalized",
    "end_semester_score_normalized",
    "in_between_assessment_grade_normalized"
]

publication_df = publication_df.drop(
    columns=intermediate_columns,
    errors="ignore"
)


# ============================================================
# 7. REMOVE INTERNAL PROCESSING COLUMNS
# ============================================================

publication_df = publication_df.drop(
    columns=[
        "performance_nonzero_count"
    ],
    errors="ignore"
)


# ============================================================
# 8. HANDLE LOGIN-DERIVED MISSING VALUES
# ============================================================

# login_frequency and login_consistency become undefined
# when there is no recorded login activity.

for col in [
    "login_frequency",
    "login_consistency"
]:
    if col in publication_df.columns:
        publication_df[col] = publication_df[col].fillna(0)


# ============================================================
# 9. HANDLE OTHER MISSING VALUES
# ============================================================

# Numerical columns -> median
numeric_columns = publication_df.select_dtypes(
    include=np.number
).columns

for col in numeric_columns:
    if publication_df[col].isnull().any():
        publication_df[col] = publication_df[col].fillna(
            publication_df[col].median()
        )

# Categorical columns -> "Unknown"
categorical_columns = publication_df.select_dtypes(
    include=["object", "category"]
).columns

for col in categorical_columns:
    if publication_df[col].isnull().any():
        publication_df[col] = publication_df[col].fillna("Unknown")


# ============================================================
# 10. REMOVE COMPLETELY EMPTY / ALL-ZERO COLUMNS
# ============================================================

all_zero_columns = []

for col in publication_df.columns:

    if pd.api.types.is_numeric_dtype(publication_df[col]):

        if publication_df[col].fillna(0).eq(0).all():
            all_zero_columns.append(col)

if all_zero_columns:

    print("\nAll-zero columns removed:")
    print(all_zero_columns)

    publication_df = publication_df.drop(
        columns=all_zero_columns
    )

else:

    print("\nNo all-zero columns found.")


# ============================================================
# 11. REMOVE EXACT DUPLICATE COLUMNS
# ============================================================

duplicate_columns = []

columns = publication_df.columns

for i in range(len(columns)):

    for j in range(i + 1, len(columns)):

        col1 = columns[i]
        col2 = columns[j]

        if publication_df[col1].equals(
            publication_df[col2]
        ):

            duplicate_columns.append(col2)

if duplicate_columns:

    print("\nExact duplicate columns removed:")
    print(duplicate_columns)

    publication_df = publication_df.drop(
        columns=list(set(duplicate_columns))
    )

else:

    print("\nNo exact duplicate columns found.")


# ============================================================
# 12. REMOVE DUPLICATE ROWS
# ============================================================

duplicate_rows_before = publication_df.duplicated().sum()

print("\nDuplicate rows before cleaning:")
print(duplicate_rows_before)

if duplicate_rows_before > 0:

    publication_df = publication_df.drop_duplicates().reset_index(
        drop=True
    )

else:

    print("No duplicate rows removed.")


# ============================================================
# 13. CLEAN COLUMN NAMES
# ============================================================

publication_df.columns = (
    publication_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# ============================================================
# 14. FINAL DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL PUBLICATION DATASET QUALITY CHECK")
print("=" * 70)

print("\nFinal Shape:")
print(publication_df.shape)

print("\nFinal Number of Columns:")
print(len(publication_df.columns))

print("\nMissing Values:")
print(publication_df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(publication_df.duplicated().sum())

print("\nRemaining Personal Identifier Columns:")

remaining_identifiers = [
    col for col in [
        "userid",
        "username",
        "firstname",
        "lastname",
        "student_name"
    ]
    if col in publication_df.columns
]

print(remaining_identifiers)


# ============================================================
# 15. CHECK IMPORTANT COLUMNS
# ============================================================

print("\nImportant Publication Columns Present:")

important_columns = [
    "course_name",
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
    "average_access_hour",
    "study_session_count",
    "time_spent_on_course_minutes",
    "study_frequency_per_week",
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
    "week_16",
    "performance_score",
    "performance_level"
]

present_important = [
    col for col in important_columns
    if col in publication_df.columns
]

print(present_important)


# ============================================================
# 16. PRINT FINAL COLUMN LIST
# ============================================================

print("\n" + "=" * 70)
print("FINAL PUBLICATION DATASET COLUMNS")
print("=" * 70)

for i, col in enumerate(
    publication_df.columns,
    start=1
):
    print(f"{i:02d}. {col}")


# ============================================================
# 17. SAVE CLEANED PUBLICATION DATASET
# ============================================================

publication_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 18. FINAL CONFIRMATION
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("\nPublication dataset saved as:")
print(output_file)

print("\nFinal Shape:")
print(publication_df.shape)

print("\nMissing Values:")
print(publication_df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(publication_df.duplicated().sum())

print("\nFile creation completed successfully.")