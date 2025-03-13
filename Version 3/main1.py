import streamlit as st
import numpy as np
import pickle

# Load trained model
try:
    with open("earthquake_model.pkl1", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'earthquake_model.pkl' is in the correct directory.")
    st.stop()

# Streamlit UI
st.title("Earthquake Magnitude Prediction")
st.markdown("### Enter location coordinates and number of stations")

# User input fields
latitude = st.number_input("Enter Latitude", format="%.6f")
longitude = st.number_input("Enter Longitude", format="%.6f")
nst = st.number_input("Enter Number of Stations (Nst)", min_value=1, step=1)

# Predict button
if st.button("Predict Earthquake Magnitude"):
    input_data = np.array([[latitude, longitude, nst]])
    predicted_mag = model.predict(input_data)[0]
    st.success(f"Predicted Earthquake Magnitude: {round(predicted_mag, 2)}")