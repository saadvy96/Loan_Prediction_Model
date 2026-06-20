import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("loan_approval_svm.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Page Config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Loan Approval Prediction System")
st.markdown("Enter applicant details to predict loan approval status.")

st.divider()

# User Inputs

dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=0
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0,
    value=500000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=1,
    value=1000000
)

loan_term = st.number_input(
    "Loan Term (Months)",
    min_value=1,
    value=12
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=700
)

residential_assets = st.number_input(
    "Residential Assets Value",
    min_value=0,
    value=0
)

commercial_assets = st.number_input(
    "Commercial Assets Value",
    min_value=0,
    value=0
)

luxury_assets = st.number_input(
    "Luxury Assets Value",
    min_value=0,
    value=0
)

bank_assets = st.number_input(
    "Bank Assets Value",
    min_value=0,
    value=0
)

# Prediction Button

if st.button("Predict Loan Status"):

    # Encoding
    education_encoded = 0 if education == "Graduate" else 1
    self_employed_encoded = 1 if self_employed == "Yes" else 0

    # Feature Engineering
    total_assets = (
        residential_assets
        + commercial_assets
        + luxury_assets
        + bank_assets
    )

    loan_income_ratio = loan_amount / income_annum if income_annum != 0 else 0

    asset_loan_ratio = (
        total_assets / loan_amount
        if loan_amount != 0
        else 0
    )

    # Create Input DataFrame
    input_data = pd.DataFrame([[
        dependents,
        education_encoded,
        self_employed_encoded,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets,
        commercial_assets,
        luxury_assets,
        bank_assets,
        loan_income_ratio,
        total_assets,
        asset_loan_ratio
    ]], columns=columns)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.divider()

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")