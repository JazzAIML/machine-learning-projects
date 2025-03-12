import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load model and scaler
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("🏥 Healthcare Readmission Prediction")
st.write("Predict whether a patient is likely to be readmitted based on various factors.")

# Sidebar inputs
st.sidebar.header("📊 Data Insights & Visualization")

# Create input fields for user input
age = st.number_input("Age", min_value=0, max_value=120, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])
diagnosis = st.selectbox("Diagnosis", ["COPD", "Diabetes", "Hypertension", "Asthma", "Heart Disease"])
previous_admissions = st.number_input("Previous Admissions", min_value=0, value=0)
lab_result = st.number_input("Lab Result", min_value=0.0, value=0.0)
insurance_type = st.selectbox("Insurance Type", ["Medicaid", "Medicare", "Private"])
length_of_stay = st.number_input("Length of Stay (in days)", min_value=1, value=5)

# Predict Readmission
if st.button("🚀 Predict Readmission"):
    # Prepare input data in the same order as training data
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender_Female': [1 if gender == "Female" else 0],
        'Gender_Male': [1 if gender == "Male" else 0],
        'Diagnosis_Asthma': [1 if diagnosis == "Asthma" else 0],
        'Diagnosis_COPD': [1 if diagnosis == "COPD" else 0],
        'Diagnosis_Diabetes': [1 if diagnosis == "Diabetes" else 0],
        'Diagnosis_Heart Disease': [1 if diagnosis == "Heart Disease" else 0],
        'Diagnosis_Hypertension': [1 if diagnosis == "Hypertension" else 0],
        'Previous_Admissions': [previous_admissions],
        'Lab_Result': [lab_result],
        'Insurance_Type_Medicaid': [1 if insurance_type == "Medicaid" else 0],
        'Insurance_Type_Medicare': [1 if insurance_type == "Medicare" else 0],
        'Insurance_Type_Private': [1 if insurance_type == "Private" else 0],
        'Length_of_Stay': [length_of_stay]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict the readmission
    prediction = model.predict(input_data_scaled)

    # Show result
    if prediction[0] == 1:
        st.error("⚠️ The patient is **likely to be readmitted**.")
    else:
        st.success("✅ The patient is **not likely to be readmitted**.")

# --- 📊 DATA VISUALIZATIONS ---
st.sidebar.subheader("📌 Data Visualizations")

# Load original dataset for visualization
DATA_PATH = "healthcare_dataset.csv"
df = pd.read_csv(DATA_PATH)
df = df.dropna()

# 1️⃣ Readmission Rate Distribution
st.sidebar.write("### 📈 Readmission Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x=df['Readmission'], palette="coolwarm", ax=ax1)
ax1.set_title("Readmission Distribution")
ax1.set_xlabel("Readmission (0 = No, 1 = Yes)")
ax1.set_ylabel("Count")
st.sidebar.pyplot(fig1)

# 2️⃣ Feature Importance Bar Chart
st.sidebar.write("### 🔥 Feature Importance")

# Ensure categorical encoding matches model
df = pd.get_dummies(df, columns=['Gender', 'Diagnosis', 'Insurance_Type'])
X = df.drop(columns=['Readmission', 'Patient_ID'])
y = df['Readmission']

# Get feature importance from model
feature_importance = model.feature_importances_
features = X.columns

# Create DataFrame for visualization
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots(figsize=(6,4))
sns.barplot(x=importance_df['Importance'], y=importance_df['Feature'], palette='viridis', ax=ax2)
ax2.set_title("Feature Importance for Readmission Prediction")
ax2.set_xlabel("Importance Score")
ax2.set_ylabel("Features")
st.sidebar.pyplot(fig2)

# 3️⃣ Age vs Readmission Scatter Plot
st.sidebar.write("### 🎯 Age vs Readmission")
fig3, ax3 = plt.subplots(figsize=(6,4))
sns.scatterplot(x=df['Age'], y=df['Readmission'], alpha=0.5, ax=ax3)
ax3.set_title("Age vs Readmission")
ax3.set_xlabel("Age")
ax3.set_ylabel("Readmission (0 = No, 1 = Yes)")
st.sidebar.pyplot(fig3)

st.sidebar.success("✅ Visualizations added successfully!")
