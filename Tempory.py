import pandas as pd

df = pd.read_csv("sql-20260914-231810.csv")

print(df[[
    "login_count",
    "active_days",
    "login_frequency",
    "login_consistency"
]].loc[
    df["login_frequency"].isna()
].head(20))