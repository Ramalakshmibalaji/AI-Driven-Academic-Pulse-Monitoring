# ============================================================
# Publication Figures
# Figure 3 - Correlation Heatmap
# Figure 4 - Weekly Student Engagement Pattern
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# 1. LOAD CLEANED PUBLICATION DATASET
# ------------------------------------------------------------

file_path = "cleaned_publication_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# FIGURE 3 — CORRELATION HEATMAP
# ============================================================

correlation_cols = [
    "login_count",
    "active_days",
    "course_access_count",
    "module_completion_rate",
    "page_view_count",
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
    "study_interval_regularity"
]

# Keep only columns available in the dataset
available_corr_cols = [
    col for col in correlation_cols
    if col in df.columns
]

print("\nVariables used for Figure 3:")
print(available_corr_cols)

# Calculate correlation matrix
corr_matrix = df[available_corr_cols].corr()

# Create Figure 3
plt.figure(figsize=(20, 16))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.75}
)

plt.title(
    "Figure 3. Correlation Heatmap of Important Dataset Variables",
    fontsize=18,
    fontweight="bold",
    pad=20
)

plt.xticks(
    rotation=45,
    ha="right",
    fontsize=10
)

plt.yticks(
    rotation=0,
    fontsize=10
)

# Give extra space for long variable names
plt.subplots_adjust(
    left=0.30,
    right=0.96,
    bottom=0.25,
    top=0.91
)

# Save high-resolution figure
plt.savefig(
    "Figure_3_Correlation_Heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFigure 3 saved as:")
print("Figure_3_Correlation_Heatmap.png")


# ============================================================
# FIGURE 4 — WEEKLY STUDENT ENGAGEMENT PATTERN
# ============================================================

# Week 1 to Week 16 columns
weekly_cols = [
    f"week_{i}"
    for i in range(1, 17)
]

# Keep only columns available in the dataset
available_weekly_cols = [
    col for col in weekly_cols
    if col in df.columns
]

print("\nWeekly columns used for Figure 4:")
print(available_weekly_cols)

# Calculate mean engagement for each week
weekly_means = df[available_weekly_cols].mean()

# Extract week numbers
week_numbers = [
    int(col.replace("week_", ""))
    for col in available_weekly_cols
]


# Create Figure 4
plt.figure(figsize=(13, 7))

plt.plot(
    week_numbers,
    weekly_means.values,
    marker="o",
    markersize=6,
    linewidth=2
)

plt.title(
    "Figure 4. Weekly Student Engagement Pattern",
    fontsize=17,
    fontweight="bold",
    pad=15
)

plt.xlabel(
    "Week",
    fontsize=12
)

plt.ylabel(
    "Mean Weekly Engagement",
    fontsize=12
)

plt.xticks(
    range(1, 17),
    fontsize=10
)

plt.yticks(
    fontsize=10
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

# Save high-resolution figure
plt.savefig(
    "Figure_4_Weekly_Student_Engagement.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFigure 4 saved as:")
print("Figure_4_Weekly_Student_Engagement.png")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n==============================================")
print("PUBLICATION FIGURES CREATED SUCCESSFULLY")
print("==============================================")
print("1. Figure_3_Correlation_Heatmap.png")
print("2. Figure_4_Weekly_Student_Engagement.png")
print("Resolution: 300 DPI")
print("==============================================")