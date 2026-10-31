import streamlit as st
import requests

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Customer Churn Prediction")
st.write("Enter customer data to see the churn probability and top reasons.")

# Create the form
with st.form("customer_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, 600.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["No", "Yes"])
        tech = st.selectbox("Tech Support", ["No", "Yes"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        
    # Submit button
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Form JSON (default values for missing fields)
        payload = {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": tenure, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": tech,
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total
        }
        
        # Send request to FastAPI
        try:
            response = requests.post("http://localhost:8000/predict", json=payload)
            result = response.json()
            
            st.divider()
            # Display result
            prob = result["churn_probability"]
            st.metric("Churn Probability", f"{prob*100:.1f}%", 
                      delta="High Risk" if prob > 0.6 else "Low Risk", 
                      delta_color="inverse" if prob > 0.6 else "normal")
            
            st.subheader("Top-3 Reasons:")
            for factor in result["top_factors"]:
                st.write(f"- **{factor['feature']}**: {factor['direction']} ({factor['impact']:+.2f})")
                
        except Exception as e:
            st.error(f"API Connection Error: {e}")