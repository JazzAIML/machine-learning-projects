# import statements: These import necessary libraries and modules for the app.
# streamlit: Used to create the web application interface.
# pandas and numpy: For handling and processing data.
# pickle: To load the trained model and scaler.

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Saved Objects:
# Loads the previously trained model (model.pkl), scaler (scaler.pkl), and the feature list (features.pkl) from disk.
# The feature list (FEATURES) ensures that the input data for prediction is in the same order as when the model was trained.
# 🔹 Load model, scaler, and feature list

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.pkl"

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open(FEATURES_PATH, "rb") as features_file:
    FEATURES = pickle.load(features_file)

# Title: Sets the title of the app.
# Streamlit UI
st.title("Healthcare Readmission Prediction")

# Input Fields: Defines the UI elements where the user can input values for various 
# features (like age, gender, diagnosis, etc.). Each feature corresponds to the model's input.
# Create input fields for the user to enter values

age = st.number_input("Age", min_value=0, max_value=120, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])
diagnosis = st.selectbox("Diagnosis", ["COPD", "Diabetes", "Hypertension", "Asthma", "Heart Disease"])  # Example, change with actual conditions
previous_admissions = st.number_input("Previous Admissions", min_value=0, value=0)
lab_result = st.number_input("Lab Result", min_value=0.0, value=0.0)
insurance_type = st.selectbox("Insurance Type", ["Medicaid", "Medicare", "Private"])  # Example, change with actual types
length_of_stay = st.number_input("Length of Stay (in days)", min_value=1, value=5)

# Button: Creates a button that triggers the prediction when clicked.
# Create a button to make the prediction

if st.button("Predict Readmission"):
    # Prepare the input data in the same order as the model features
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender_Male': [1 if gender == "Male" else 0],
        'Gender_Female': [1 if gender == "Female" else 0],
        'Diagnosis_COPD': [1 if diagnosis == "COPD" else 0],
        'Diagnosis_Diabetes': [1 if diagnosis == "Diabetes" else 0],
        'Diagnosis_Hypertension': [1 if diagnosis == "Hypertension" else 0],
        'Diagnosis_Asthma': [1 if diagnosis == "Asthma" else 0],
        'Diagnosis_Heart Disease': [1 if diagnosis == "Heart Disease" else 0],
        'Previous_Admissions': [previous_admissions],
        'Lab_Result': [lab_result],
        'Insurance_Type_Medicaid': [1 if insurance_type == "Medicaid" else 0],
        'Insurance_Type_Medicare': [1 if insurance_type == "Medicare" else 0],
        'Insurance_Type_Private': [1 if insurance_type == "Private" else 0],
        'Length_of_Stay': [length_of_stay]
    })

    # Input Data:
    # This prepares a DataFrame with the input features. It handles one-hot encoding for categorical 
    # variables (e.g., Gender, Diagnosis, Insurance_Type), using 1 or # 0 to indicate the presence of a category.
    
    # Ensure input data matches the model's expected feature order
    input_data = input_data[FEATURES]  # Reorder the columns to match the training data

    # Scaling and Prediction:
    # The input data is scaled using the same scaler used during training.
    # The model predicts the readmission outcome based on the scaled input data.
    
    # Scale the input data using the saved scaler
    input_data_scaled = scaler.transform(input_data)

    # Predict the readmission using the trained model
    prediction = model.predict(input_data_scaled)

    # Display Prediction: Based on the model's prediction, 
    # a message is displayed to the user indicating whether the patient is likely to be readmitted or not.
    # Show the result
    if prediction == 1:
        st.write("The patient is likely to be readmitted.")
    else:
        st.write("The patient is not likely to be readmitted.")
