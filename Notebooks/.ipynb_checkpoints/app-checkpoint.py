import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("healthcare_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🩺 AI Healthcare Risk Assessment")
st.write("Enter your health details to predict your risk level.")

# User inputs
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose Level", 50, 200, 100)
blood_pressure = st.slider("Blood Pressure", 60, 180, 120)
skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin Level", 0, 900, 80)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 18, 100, 30)

# Predict button
if st.button("Predict Risk"):
    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                            insulin, bmi, pedigree, age]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)

    if prediction[0] == 1:
        st.error("⚠ High Risk Detected! Consult a Doctor.")
    else:
        st.success("✅ Low Risk - Stay Healthy!")
