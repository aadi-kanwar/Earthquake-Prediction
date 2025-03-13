import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

model = load_model()

st.title("Earthquake Prediction App")
st.write("Enter the details to predict the earthquake severity.")

# Input fields
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.01)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.01)
stations = st.number_input("Number of Stations", min_value=1, value=5, step=1)

# Predict button
if st.button("Predict"):
    input_data = np.array([[latitude, longitude, stations]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Earthquake Severity: {prediction[0]}")
