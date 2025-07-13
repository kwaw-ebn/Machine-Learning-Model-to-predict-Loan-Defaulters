import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import os

# === Load model and saved feature names ===
model, feature_names = joblib.load("loan_default_model.joblib")

# === UI Header ===
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("images/logo.JPG"):
        logo = Image.open("images/logo.JPG")
        st.image(logo, width=80)
with col2:
    st.title("🏦 Loan Default Prediction App")
    st.write("👨‍💻 Created by **Ebenezer Kwaw**")

st.markdown("---")

# === Collect user inputs ===
st.header("📋 Applicant Information")

credit_policy = st.selectbox("Credit Policy Met?", [1, 0])
int_rate = st.number_input("Interest Rate", min_value=0.0, max_value=1.0, format="%.3f")
installment = st.number_input("Installment Amount", min_value=0.0)
log_annual_inc = st.number_input("Log of Annual Income", min_value=0.0)
dti = st.number_input("Debt-to-Income Ratio", min_value=0.0)
fico = st.number_input("FICO Score", min_value=300, max_value=850)
revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0)
inq_last_6mths = st.number_input("Inquiries in Last 6 Months", min_value=0)

# === Map inputs to dictionary ===
user_inputs = {
    "credit.policy": credit_policy,
    "int.rate": int_rate,
    "installment": installment,
    "log.annual.inc": log_annual_inc,
    "dti": dti,
    "fico": fico,
    "revol.util": revol_util,
    "inq.last.6mths": inq_last_6mths
}

# === Predict ===
if st.button("🔍 Predict Loan Risk"):
    # Step 1: Create a full zeroed feature dictionary
    input_dict = {feature: 0 for feature in feature_names}

    # Step 2: Update with user's real inputs
    for key in user_inputs:
        if key in input_dict:
            input_dict[key] = user_inputs[key]

    # Step 3: Create input DataFrame
    input_df = pd.DataFrame([input_dict])

    # Step 4: Predict
    prediction = model.predict(input_df)

    st.subheader("📊 Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ High risk: Loan is likely to **default**.")
    else:
        st.success("✅ Low risk: Loan is likely to be **fully paid**.")
