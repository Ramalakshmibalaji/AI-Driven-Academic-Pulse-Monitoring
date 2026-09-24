import pandas as pd


# ==========================================
# 1. Load Student Prediction Dataset
# ==========================================

df = pd.read_csv("student_prediction_data.csv")


# ==========================================
# 2. Select Weekly Features
# ==========================================

weekly_features = [
    f"week_{i}"
    for i in range(1, 17)
]


# ==========================================
# 3. Select Student
# ==========================================

userid = int(
    input("\nEnter Student ID: ").strip()
)

student = df[
    df["userid"] == userid
]


# ==========================================
# 4. Check Student
# ==========================================

if student.empty:

    print("\nStudent ID not found.")

else:

    # --------------------------------------
    # Extract Weekly Activity
    # --------------------------------------

    weekly_values = student[
        weekly_features
    ].iloc[0].astype(float)


    # ======================================
    # Display Weekly Monitoring
    # ======================================

    print("\n" + "=" * 60)

    print(
        "          WEEKLY LEARNING MONITORING"
    )

    print("=" * 60)

    print(
        "Student ID :",
        userid
    )

    print(
        "Course     :",
        student["course_name"].iloc[0]
    )

    print(
        "\n------ Weekly Learning Activity ------"
    )


    for week, value in weekly_values.items():

        print(
            f"{week:<10}: {value}"
        )


    # ======================================
    # Weekly Statistics
    # ======================================

    first_week = weekly_values.iloc[0]

    last_week = weekly_values.iloc[-1]

    average_weekly_activity = (
        weekly_values.mean()
    )

    highest_week = weekly_values.idxmax()

    highest_value = weekly_values.max()

    lowest_week = weekly_values.idxmin()

    lowest_value = weekly_values.min()


    # ======================================
    # Recent Activity Analysis
    # ======================================

    # Last 4 weeks are considered
    # for recent learning activity

    recent_weeks = weekly_values.iloc[-4:]

    recent_average = recent_weeks.mean()

    inactive_weeks = (
        weekly_values == 0
    ).sum()

    recent_inactive_weeks = (
        recent_weeks == 0
    ).sum()


    # ======================================
    # Determine Recent Activity Status
    # ======================================

    if recent_inactive_weeks == 4:

        recent_activity = "Inactive"

    elif recent_inactive_weeks >= 2:

        recent_activity = "Low Activity"

    elif recent_average < average_weekly_activity:

        recent_activity = "Moderate Activity"

    else:

        recent_activity = "Active"


    # ======================================
    # Display Trend Summary
    # ======================================

    print(
        "\n------ Weekly Trend Summary ------"
    )

    print(
        "Average Weekly Activity :",
        round(
            average_weekly_activity,
            2
        )
    )

    print(
        "Highest Activity        :",
        highest_week,
        "(",
        highest_value,
        ")"
    )

    print(
        "Lowest Activity         :",
        lowest_week,
        "(",
        lowest_value,
        ")"
    )

    print(
        "First Week Activity     :",
        first_week
    )

    print(
        "Last Week Activity      :",
        last_week
    )

    print(
        "Inactive Weeks          :",
        inactive_weeks
    )

    print(
        "Recent Activity         :",
        recent_activity
    )


    # ======================================
    # Improved Overall Trend Analysis
    # ======================================

    # First 8 weeks average

    first_half_average = (
        weekly_values.iloc[:8].mean()
    )

    # Last 8 weeks average

    second_half_average = (
        weekly_values.iloc[8:].mean()
    )

    # Overall activity range

    activity_range = (
        weekly_values.max()
        - weekly_values.min()
    )

    # Avoid division issues

    mean_activity = (
        weekly_values.mean()
    )


    # ======================================
    # Determine Overall Trend
    # ======================================

    if mean_activity == 0:

        trend = "No Activity"

    elif activity_range > mean_activity:

        trend = "Fluctuating"

    elif second_half_average > (
        first_half_average * 1.10
    ):

        trend = "Increasing"

    elif second_half_average < (
        first_half_average * 0.90
    ):

        trend = "Decreasing"

    else:

        trend = "Stable"


    # ======================================
    # Display Final Monitoring Status
    # ======================================

    print(
        "\nOverall Weekly Trend    :",
        trend
    )

    print("=" * 60)