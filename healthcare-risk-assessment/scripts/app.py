import streamlit as st
import pickle
import os
import numpy as np

# Define the paths to model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")

# Load the pretrained model
@st.cache_resource  # Caches the model to improve performance
def load_model():
    try:
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Ensure it's in the correct location.")
        return None

# Load the scaler (if applicable)
@st.cache_resource
def load_scaler():
    try:
        with open(SCALER_PATH, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        return scaler
    except FileNotFoundError:
        st.warning(f"Scaler file not found at {SCALER_PATH}. Proceeding without scaling.")
        return None

# Load model and scaler
model = load_model()
scaler = load_scaler()

# Streamlit UI
st.title("Healthcare Risk Assessment")

# Input form
age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
bmi = st.number_input("Enter BMI", min_value=10.0, max_value=50.0, value=22.5)
blood_pressure = st.number_input("Enter Blood Pressure", min_value=50, max_value=200, value=120)

if st.button("Predict Risk"):
    if model:
        # Prepare input data
        input_data = np.array([[age, bmi, blood_pressure]])

        # Apply scaler if available
        if scaler:
            input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Risk Score: {prediction:.2f}")
    else:
        st.error("Model not loaded. Please check the file paths.")

