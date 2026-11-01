import os

import requests
import streamlit as st

API_URL = os.getenv("CHURN_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Customer churn", layout="centered")
st.title("Customer churn prediction")
st.caption("Estimate churn risk for a telecom customer.")

with st.form("customer"):
    st.subheader("Customer")
    left, right = st.columns(2)
    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.checkbox("Senior citizen")
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
    with right:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])

    st.subheader("Services")
    left, right = st.columns(2)
    with left:
        phone = st.selectbox("Phone service", ["Yes", "No"])
        lines = st.selectbox("Multiple lines", ["No", "Yes"], disabled=phone == "No")
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online security", ["No", "Yes"], disabled=internet == "No")
        backup = st.selectbox("Online backup", ["No", "Yes"], disabled=internet == "No")
    with right:
        protection = st.selectbox("Device protection", ["No", "Yes"], disabled=internet == "No")
        tech_support = st.selectbox("Tech support", ["No", "Yes"], disabled=internet == "No")
        tv = st.selectbox("Streaming TV", ["No", "Yes"], disabled=internet == "No")
        movies = st.selectbox("Streaming movies", ["No", "Yes"], disabled=internet == "No")

    st.subheader("Billing")
    monthly = st.number_input("Monthly charges ($)", min_value=0.0, value=50.0, step=1.0)
    total = st.number_input("Total charges ($)", min_value=0.0, value=600.0, step=10.0)
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    ])
    submitted = st.form_submit_button("Predict churn")

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": lines if phone == "Yes" else "No phone service",
        "InternetService": internet,
        "OnlineSecurity": security if internet != "No" else "No internet service",
        "OnlineBackup": backup if internet != "No" else "No internet service",
        "DeviceProtection": protection if internet != "No" else "No internet service",
        "TechSupport": tech_support if internet != "No" else "No internet service",
        "StreamingTV": tv if internet != "No" else "No internet service",
        "StreamingMovies": movies if internet != "No" else "No internet service",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Could not get a prediction: {exc}")
    else:
        result = response.json()
        st.metric("Estimated churn probability", f"{result['churn_probability']:.1%}")
        st.write(f"Risk band: **{result['risk_level']}**")
        st.caption("Largest SHAP contributions (log-odds, not probability changes):")
        for item in result["top_factors"]:
            st.write(f"{item['feature']}: {item['direction']} ({item['impact']:+.3f})")
