import pandas as pd

df = pd.read_csv("insurance.csv")

print("First Five Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nColumn Data Types:")
print(df.dtypes)
