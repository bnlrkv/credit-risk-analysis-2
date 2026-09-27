import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Loan Default Risk Calculator", layout="wide")

# Load model artifacts via joblib
@st.cache_resource
def load_model():
    return joblib.load('model_artifacts.joblib')

art = load_model()
model = art['model']
scaler = art['scaler']
medians = art['imputation_medians']
feature_cols = art['feature_cols']
job_map = art['job_map']
num_cols = art['num_cols']

st.title("🏦 Credit Risk & Default Prediction Calculator")
st.markdown("Enter applicant details below to evaluate the probability of default and generate an underwriting recommendation.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Loan Details")
    loan_amount = st.number_input("Requested Loan Amount ($)", value=250000, step=10000)
    term = st.selectbox("Term", ["Short Term", "Long Term"])
    
    # Cleaned list: exactly one "Other", "Personal Loan" added, duplicates removed
    purpose_options = [
        "Debt Consolidation",
        "Home Improvements",
        "Business Loan",
        "Buy a Car",
        "Personal Loan",
        "Other"
    ]
    purpose = st.selectbox("Loan Purpose", purpose_options)

with col2:
    st.subheader("Financial Profile")
    annual_income = st.number_input("Annual Income ($)", value=850000, step=25000)
    monthly_debt = st.number_input("Monthly Debt Payments ($)", value=12000, step=1000)
    credit_score = st.slider("Credit Score (FICO)", min_value=300, max_value=850, value=710)
    years_job = st.selectbox("Years in Current Job", list(job_map.keys()), index=5)
    home_ownership = st.selectbox("Home Ownership", ["Home Mortgage", "Rent", "Own Home"])

with col3:
    st.subheader("Credit Bureau History")
    credit_hist = st.number_input("Years of Credit History", value=12.0, step=0.5)
    open_acc = st.number_input("Number of Open Accounts", value=8, step=1)
    credit_bal = st.number_input("Current Credit Balance ($)", value=100000, step=5000)
    max_credit = st.number_input("Maximum Open Credit ($)", value=400000, step=10000)
    credit_prob = st.number_input("Number of Credit Problems", value=0, step=1)
    delinq_choice = st.radio("Prior Delinquency?", ["Never Delinquent", "Has Missed Payments"])
    delinq_months = st.number_input("Months Since Last Delinquent", value=12, step=1) if delinq_choice == "Has Missed Payments" else 0
    bankruptcies = st.number_input("Bankruptcies", value=0, step=1)
    tax_liens = st.number_input("Tax Liens", value=0, step=1)

if st.button("Evaluate Applicant Risk", type="primary", use_container_width=True):
    # Map "Personal Loan" or "Other" to the underlying training category
    model_purpose = "other" if purpose in ["Personal Loan", "Other"] else purpose

    app_data = {
        'Current Loan Amount': loan_amount,
        'Term': term,
        'Credit Score': credit_score,
        'Annual Income': annual_income,
        'Years in current job': job_map.get(years_job, 5),
        'Home Ownership': home_ownership,
        'Purpose': model_purpose,
        'Monthly Debt': monthly_debt,
        'Years of Credit History': credit_hist,
        'Has_Delinquency': 1 if delinq_choice == "Has Missed Payments" else 0,
        'Months since last delinquent': delinq_months if delinq_choice == "Has Missed Payments" else 0,
        'Number of Open Accounts': open_acc,
        'Number of Credit Problems': credit_prob,
        'Current Credit Balance': credit_bal,
        'Maximum Open Credit': max_credit,
        'Bankruptcies': bankruptcies,
        'Tax Liens': tax_liens
    }
    
    row = pd.DataFrame([app_data])
    
    # Dummy feature alignment
    for col in feature_cols:
        if col not in row.columns:
            if col.startswith('Term_'):
                val = col.split('Term_')[1]
                row[col] = 1 if term == val else 0
            elif col.startswith('Home Ownership_'):
                val = col.split('Home Ownership_')[1]
                row[col] = 1 if home_ownership == val else 0
            elif col.startswith('Purpose_Cleaned_'):
                val = col.split('Purpose_Cleaned_')[1]
                row[col] = 1 if model_purpose.lower() == val.lower() else 0
            else:
                row[col] = 0

    X_app = scaler.transform(row[feature_cols])
    prob = model.predict_proba(X_app)[0, 1]
    
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(label="Calculated Default Probability", value=f"{prob * 100:.2f}%")
        
    with res_col2:
        if prob < 0.38:
            st.success("Verdict: LOW RISK — APPROVE LOAN")
        elif prob < 0.55:
            st.warning("Verdict: MODERATE RISK — REQUIRE MANUAL REVIEW / COLLATERAL")
        else:
            st.error("Verdict: HIGH RISK — REJECT LOAN APPLICATION")
