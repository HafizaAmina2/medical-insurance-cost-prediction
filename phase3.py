
import pandas as pd

# Load the original dataset
df = pd.read_csv("insurance.csv")

# 1. Check missing values
print("1. MISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())

# 2. Check duplicate records
print("\n2. DUPLICATE RECORDS BEFORE CLEANING")
print(df.duplicated().sum())

# 3. Show original data
print("\n3. ORIGINAL DATA")
print(df.head())

# 4. Remove duplicate records
df_clean = df.drop_duplicates().copy()

# 5. Clean text values by removing extra spaces
# and making text lowercase
for col in df_clean.select_dtypes(include="object").columns:
    df_clean[col] = df_clean[col].str.strip().str.lower()

# 6. Convert numeric columns to numeric data types
numeric_cols = ["age", "bmi", "children", "charges"]

for col in numeric_cols:
    df_clean[col] = pd.to_numeric(
        df_clean[col], errors="coerce"
    )

# Remove rows where the target value is missing
df_clean = df_clean.dropna(subset=["charges"]).copy()

# Fill missing numerical input values with their median
for col in ["age", "bmi", "children"]:
    df_clean[col] = df_clean[col].fillna(
        df_clean[col].median()
    )

# Fill missing categorical input values with the most
# common value in each column
for col in ["sex", "smoker", "region"]:
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna(
            df_clean[col].mode()[0]
        )

# 7. Check the cleaned dataset
print("\n4. MISSING VALUES AFTER CLEANING")
print(df_clean.isnull().sum())

print("\n5. DUPLICATES AFTER CLEANING")
print(df_clean.duplicated().sum())

print("\n6. CLEANED DATA")
print(df_clean.head())

print("\n7. DATA TYPES AFTER CLEANING")
print(df_clean.dtypes)

# Save the cleaned dataset before encoding
df_clean.to_csv("insurance_cleaned.csv", index=False)

# 8. Encode categorical columns into numerical columns
df_ml = pd.get_dummies(
    df_clean,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

# 9. Display encoded data
print("\n8. ENCODED DATA")
print(df_ml.head())

print("\n9. FINAL DATASET SHAPE")
print(df_ml.shape)

print("\n10. FINAL COLUMN NAMES")
print(df_ml.columns.tolist())

# Save the final prepared dataset
df_ml.to_csv("insurance_preprocessed.csv", index=False)

print("\nPreprocessing completed successfully!")
