import streamlit as st
import pickle
import numpy as np

# Load model
with open("model/churn_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

st.title("📊 Customer Churn Prediction")

st.write("Enter customer details to predict churn")

tenure = st.number_input("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0)
contract = st.selectbox("Contract Type", [0, 1, 2])  # Encoded
payment_method = st.selectbox("Payment Method", [0, 1, 2, 3])

input_data = np.array([[tenure, monthly_charges, total_charges, contract, payment_method]])
input_scaled = scaler.transform(input_data)

if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    result = "Churn" if prediction == 1 else "No Churn"
    st.success(f"Prediction: {result}")
