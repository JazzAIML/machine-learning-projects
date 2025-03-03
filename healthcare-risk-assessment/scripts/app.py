import os
import streamlit as st
import numpy as np
import joblib

# Set the base directory dynamically (where app.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the models directory
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Define model and scaler file paths
model_path = os.path.join(MODELS_DIR, "healthcare_model.pkl")
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

# Check if model and scaler files exist
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("⚠ Model or Scaler file not found! Please check your repository.")
else:
    # Load trained model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    st.success("✅ Model and Scaler Loaded Successfully!")

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
