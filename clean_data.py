
import pandas as pd
import os

# Create output folder
os.makedirs("data/cleaned", exist_ok=True)

# Load raw dataset
df = pd.read_csv("data/raw/titanic.csv")

print("Original Dataset Shape:", df.shape)

# STEP 1: Remove duplicate records
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates().copy()

print("Duplicates Removed:", duplicate_count)

# STEP 2: Handle missing values

# Fill Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked with mode
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# Fill Cabin with Unknown
df["Cabin"] = df["Cabin"].fillna("Unknown")

# STEP 3: Standardize text values

df["Sex"] = df["Sex"].astype(str).str.strip().str.lower()

df["Embarked"] = (
    df["Embarked"].astype(str).str.strip().str.upper()
)

# STEP 4: Correct data types

df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")

df["Pclass"] = pd.to_numeric(
    df["Pclass"], errors="coerce"
).astype("Int64")

df["Survived"] = pd.to_numeric(
    df["Survived"], errors="coerce"
).astype("Int64")

# STEP 5: Save cleaned dataset
df.to_csv("data/cleaned/titanic_cleaned.csv", index=False)

# STEP 6: Final report
print("\n========== CLEANING REPORT ==========")

print("\nFinal Dataset Shape:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nFinal Data Types:")
print(df.dtypes)

print("\nRemaining Duplicate Records:", df.duplicated().sum())

print("\nCleaned Dataset Saved Successfully!")


# STEP 7: DATA VALIDATION

print("\n========== DATA VALIDATION ==========")

# Load original dataset again
raw_df = pd.read_csv("data/raw/titanic.csv")

# Load cleaned dataset
clean_df = pd.read_csv("data/cleaned/titanic_cleaned.csv")

# Compare dataset shapes
print("\nRAW DATASET SHAPE:", raw_df.shape)
print("CLEANED DATASET SHAPE:", clean_df.shape)

# Compare missing values
print("\nRAW MISSING VALUES:")
print(raw_df.isnull().sum())

print("\nCLEANED MISSING VALUES:")
print(clean_df.isnull().sum())

# Check duplicates
print("\nRAW DUPLICATES:", raw_df.duplicated().sum())
print("CLEANED DUPLICATES:", clean_df.duplicated().sum())

# Check data types
print("\nCLEANED DATA TYPES:")
print(clean_df.dtypes)

# Validation results
print("\n========== FINAL VALIDATION ==========")

if clean_df.isnull().sum().sum() == 0:
    print("PASS: No missing values found.")

if clean_df.duplicated().sum() == 0:
    print("PASS: No duplicate records found.")

if raw_df.shape[1] == clean_df.shape[1]:
    print("PASS: All columns preserved.")

if raw_df.shape[0] == clean_df.shape[0]:
    print("PASS: Row count preserved.")

print("\nVALIDATION COMPLETED!")
