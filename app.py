
import streamlit as st
import pandas as pd
import joblib

# Load the saved model
saved_data = joblib.load("insurance_model.pkl")
model = saved_data["model"]
feature_columns = saved_data["feature_columns"]

# Application title
st.title("Medical Insurance Cost Prediction")
st.write(
    "Enter the customer's details to estimate medical insurance charges."
)

# Input fields
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30,
    step=1
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

sex = st.selectbox(
    "Gender",
    ["female", "male"]
)

smoker = st.selectbox(
    "Smoking Status",
    ["no", "yes"]
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# Predict button
if st.button("Predict Insurance Cost"):

    # Create input data
    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region
    }])

    # Convert categorical values into numeric columns
    input_data = pd.get_dummies(
        input_data,
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=int
    )

    # Match the exact feature columns used during training
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict insurance charges
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(
        f"Estimated Insurance Charges: {prediction:,.2f}"
    )

    st.caption(
        "This is a model-based estimate, not an exact insurance quote."
    )
