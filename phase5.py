
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load the cleaned dataset
df = pd.read_csv("insurance_cleaned.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Convert categorical columns into numeric columns
df = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

# 3. Separate features (X) and target (y)
# X contains input variables
# y contains the value we want to predict: charges
X = df.drop("charges", axis=1)
y = df["charges"]

print("\nFeatures (X):")
print(X.columns.tolist())

print("\nTarget variable (y): charges")

# 4. Split data into training and testing sets
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining and testing data created!")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

# 5. Create the Multiple Linear Regression model
model = LinearRegression()

# 6. Train the model using training data
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# 7. Show model coefficients
print("\nModel Intercept:")
print(round(model.intercept_, 2))

print("\nFeature Coefficients:")
for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", round(coefficient, 2))

# 8. Make predictions on the testing data
predictions = model.predict(X_test)

print("\nFirst 5 Actual Charges:")
print(y_test.head().round(2).tolist())

print("\nFirst 5 Predicted Charges:")
print(pd.Series(predictions[:5]).round(2).tolist())

print("\nPhase 5 completed successfully!")
