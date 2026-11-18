
import pandas as pd

raw = pd.read_csv("data/raw/titanic.csv")
cleaned = pd.read_csv("data/cleaned/titanic_cleaned.csv")

report = pd.DataFrame({
    "Metric": [
        "Total Rows",
        "Total Columns",
        "Missing Values",
        "Duplicate Records"
    ],
    "Raw Dataset": [
        raw.shape[0],
        raw.shape[1],
        raw.isnull().sum().sum(),
        raw.duplicated().sum()
    ],
    "Cleaned Dataset": [
        cleaned.shape[0],
        cleaned.shape[1],
        cleaned.isnull().sum().sum(),
        cleaned.duplicated().sum()
    ]
})

print(report)

report.to_csv("data/cleaning_summary.csv", index=False)

print("\nSummary report saved successfully!")
