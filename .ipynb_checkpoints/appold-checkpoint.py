import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
@st.cache
def load_model():
    with open('HealthcareAI-Agent/your_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('HealthcareAI-Agent/your_data.csv')
    return df

# Function to predict with the model
def predict(input_data, model):
    # Assuming the model expects features to be scaled
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction

# Streamlit UI
st.title('Healthcare AI Agent')

# Load data and model
df = load_data()
model = load_model()

# Display dataset
st.subheader('Healthcare Dataset')
st.write(df.head())

# Input features for prediction (adjust according to your dataset and model)
st.subheader('Input Features for Prediction')
input_feature_1 = st.number_input('Feature 1', min_value=0)
input_feature_2 = st.number_input('Feature 2', min_value=0)
input_feature_3 = st.number_input('Feature 3', min_value=0)

input_data = pd.DataFrame([[input_feature_1, input_feature_2, input_feature_3]],
                          columns=['Feature 1', 'Feature 2', 'Feature 3'])

# Make a prediction
if st.button('Predict'):
    prediction = predict(input_data, model)
    st.subheader('Prediction Result')
    st.write(f"Prediction: {prediction[0]}")

