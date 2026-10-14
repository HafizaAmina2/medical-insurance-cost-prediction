
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Load the cleaned dataset
df = pd.read_csv("insurance_cleaned.csv")

# 2. Convert categorical data into numeric data
df = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

# 3. Separate features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict insurance charges for test data
predictions = model.predict(X_test)

# 7. Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

# 8. Display evaluation results
print("MODEL EVALUATION RESULTS")
print("-------------------------")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 9. Compare actual and predicted charges
comparison = pd.DataFrame({
    "Actual Charges": y_test.values,
    "Predicted Charges": predictions.round(2)
})

print("\nACTUAL VS PREDICTED CHARGES")
print(comparison.head(10).round(2).to_string(index=False))

print("\nPhase 6 completed successfully!")
