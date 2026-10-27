
# Medical Insurance Cost Prediction

A machine learning project that estimates medical insurance charges using Multiple Linear Regression and a Streamlit web application.

## Project Overview

The goal of this project is to predict estimated medical insurance charges based on customer information such as age, BMI, gender, smoking status, number of children, and region.

The project demonstrates a complete machine learning workflow, from data exploration and preprocessing to model training, evaluation, and deployment.

## Objectives

- Explore and understand the insurance dataset.
- Clean duplicate records and check missing values.
- Convert categorical features into numerical features.
- Train a Multiple Linear Regression model.
- Evaluate the model using MAE, MSE, RMSE, and R² Score.
- Build an interactive application for making predictions.
- Deploy the application online.

## Dataset

The project uses the Medical Cost Personal Dataset available on Kaggle.

Dataset source:
https://www.kaggle.com/datasets/mirichoi0218/insurance

The dataset includes these columns:

| Feature | Description |
|---|---|
| age | Customer's age |
| sex | Customer's gender |
| bmi | Body Mass Index |
| children | Number of children/dependents |
| smoker | Smoking status |
| region | Residential region category |
| charges | Medical insurance charges; prediction target |

The original dataset contains 1,338 records. After removing one duplicate record, the cleaned dataset contains 1,337 records.

## Data Preprocessing

The preprocessing workflow included:

1. Loading the dataset using Pandas.
2. Checking missing values and duplicate records.
3. Removing the duplicate record.
4. Checking and converting data types where necessary.
5. Handling missing values if present.
6. Encoding categorical variables using one-hot encoding with `drop_first=True`.
7. Separating input features (X) from the target variable (y).

The target variable is `charges`. It is not included among the input features.

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the dataset and investigate relationships between customer characteristics and insurance charges.

The analysis included:

- Statistical summaries of numerical variables.
- Frequency counts of categorical variables.
- Age versus insurance charges.
- BMI versus insurance charges.
- Smoking status versus insurance charges.
- Correlation analysis of numerical variables.

Smoking status, age, and BMI are important variables to investigate when studying insurance charges. The strength of each relationship should be interpreted from the dataset analysis.

## Machine Learning Model

**Selected model:** Multiple Linear Regression

The model uses customer characteristics to estimate insurance charges.

### Training process

1. The cleaned dataset was loaded.
2. Categorical features were converted into numerical columns.
3. The features (X) and target (y) were separated.
4. The data was split into 80% training data and 20% testing data.
5. The Linear Regression model was trained using the training data.
6. Predictions were generated for the testing data.

The split used `random_state=42` for reproducibility.

## Model Evaluation Results

The model was evaluated on the test dataset.

| Metric | Result |
|---|---:|
| MAE | 4177.05 |
| MSE | 35478020.68 |
| RMSE | 5956.34 |
| R² Score | 0.8069 |

The R² Score of 0.8069 indicates that the model explains approximately 80.69% of the variation in charges in this test dataset.

MAE and RMSE measure prediction error in the same charge units as the target. Lower values generally indicate smaller errors. The model's predictions are estimates and are not exact for every customer.

## Prediction Application

A Streamlit application allows users to enter:

- Age
- BMI
- Gender
- Smoking status
- Number of children
- Region

After the user clicks **Predict Insurance Cost**, the trained model displays the estimated insurance charges.

### Live Application

Deployed Streamlit application URL:

https://medical-insurance-cost-prediction-9.streamlit.app/

## How to Run Locally

1. Install Python.
2. Open the project folder in VS Code.
3. Install the required packages:

   `python -m pip install -r requirements.txt`

4. Make sure `insurance_model.pkl` is in the project folder.
5. Start the application:

   `streamlit run app.py`

## Project Structure

```text
medical-insurance-cost-prediction/
├── app.py
├── phase2.py
├── phase3.py
├── phase4.py
├── phase5.py
├── phase6.py
├── phase7_train.py
├── insurance.csv
├── insurance_cleaned.csv
├── insurance_model.pkl
├── requirements.txt
└── README.md
```

## Limitations

- The model learns patterns from historical data and may not generalize to every person.
- Prediction errors are expected; estimated charges are not guaranteed real-world prices.
- The model's performance depends on the dataset and selected features.
- This project is for educational purposes and is not professional insurance, financial, or medical advice.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib
- GitHub

## Author

Created as a machine learning project for educational and internship purposes.
