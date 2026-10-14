
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("insurance_cleaned.csv")

# 1. Basic statistical analysis
print("1. DATASET OVERVIEW")
print("Dataset Shape:", df.shape)
print("\nStatistical Summary:")
print(df.describe())

# 2. Analyze categorical variables
print("\n2. CATEGORICAL VARIABLE ANALYSIS")

for col in ["sex", "smoker", "region"]:
    print(f"\nCounts for {col}:")
    print(df[col].value_counts())

# 3. Visualization: Age vs Insurance Charges
plt.figure(figsize=(8, 5))
plt.scatter(df["age"], df["charges"], alpha=0.6)
plt.title("Age vs Medical Insurance Charges")
plt.xlabel("Age")
plt.ylabel("Insurance Charges")
plt.tight_layout()
plt.savefig("eda_age_charges.png")
plt.show()

# 4. Visualization: BMI vs Insurance Charges
plt.figure(figsize=(8, 5))
plt.scatter(df["bmi"], df["charges"], alpha=0.6)
plt.title("BMI vs Medical Insurance Charges")
plt.xlabel("BMI")
plt.ylabel("Insurance Charges")
plt.tight_layout()
plt.savefig("eda_bmi_charges.png")
plt.show()

# 5. Visualization: Smoker vs Non-Smoker Charges
smoker_groups = [
    df[df["smoker"] == "no"]["charges"],
    df[df["smoker"] == "yes"]["charges"]
]

plt.figure(figsize=(8, 5))
plt.boxplot(smoker_groups, tick_labels=["Non-Smoker", "Smoker"])
plt.title("Insurance Charges: Smoker vs Non-Smoker")
plt.xlabel("Smoking Status")
plt.ylabel("Insurance Charges")
plt.tight_layout()
plt.savefig("eda_smoker_charges.png")
plt.show()

# 6. Correlation analysis of numerical variables
numeric_cols = ["age", "bmi", "children", "charges"]
correlation = df[numeric_cols].corr()

print("\n3. CORRELATION WITH INSURANCE CHARGES")
print(correlation["charges"].sort_values(ascending=False))

# 7. Correlation heatmap
plt.figure(figsize=(7, 5))
plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
plt.yticks(range(len(numeric_cols)), numeric_cols)

for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        plt.text(
            j, i, f"{correlation.iloc[i, j]:.2f}",
            ha="center", va="center"
        )

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png")
plt.show()

# 8. Automatically display important observations
print("\n4. IMPORTANT OBSERVATIONS")

print(
    "Average charges for non-smokers:",
    round(df[df["smoker"] == "no"]["charges"].mean(), 2)
)
print(
    "Average charges for smokers:",
    round(df[df["smoker"] == "yes"]["charges"].mean(), 2)
)

print(
    "Correlation between age and charges:",
    round(df["age"].corr(df["charges"]), 3)
)
print(
    "Correlation between BMI and charges:",
    round(df["bmi"].corr(df["charges"]), 3)
)
print(
    "Correlation between children and charges:",
    round(df["children"].corr(df["charges"]), 3)
)

print("\nEDA completed successfully!")
