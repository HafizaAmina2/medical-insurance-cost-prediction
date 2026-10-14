
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load cleaned dataset
df = pd.read_csv("insurance_cleaned.csv")

# Convert categorical columns into numeric columns
df = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

# Separate features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Use the same 80/20 split as Phase 5 and Phase 6
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and feature column names
joblib.dump(
    {
        "model": model,
        "feature_columns": X.columns.tolist()
    },
    "insurance_model.pkl"
)

print("Model trained and saved successfully!")
print("Saved file: insurance_model.pkl")
print("Features:", X.columns.tolist())
