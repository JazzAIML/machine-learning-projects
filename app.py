import os
import streamlit as st
import numpy as np
import joblib

# 🛡 Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🛡 Load model and scaler using secure relative paths
MODEL_PATH = os.path.join(BASE_DIR, "healthcare_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# 🛡 Check if model files exist before loading
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("🔴 Model or Scaler file not found! Please check your repository.")
else:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

# 🔷 Streamlit UI
st.title("🩺 AI Healthcare Risk Assessment")
st.write("Enter your health details to predict your risk level.")

# 🔷 User inputs
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose Level", 50, 200, 100)
blood_pressure = st.slider("Blood Pressure", 60, 180, 120)
skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin Level", 0, 900, 80)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 18, 100, 30)

# 🔷 Predict button
if st.button("Predict Risk"):
    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                            insulin, bmi, pedigree, age]])

    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)

    if prediction[0] == 1:
        st.error("⚠ High Risk Detected! Consult a Doctor.")
    else:
        st.success("✅ Low Risk - Stay Healthy!")
