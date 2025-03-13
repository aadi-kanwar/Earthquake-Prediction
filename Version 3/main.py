import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import pickle

# Load trained model
try:
    with open("earthquake_model1.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'earthquake_model.pkl' is in the correct directory.")
    st.stop()

# Streamlit UI
st.title("Earthquake Magnitude Prediction")
st.markdown("### Select a location on the map and enter the number of stations")

# Create a Folium Map
m = folium.Map(location=[20, 0], zoom_start=5)
folium.Marker([20, 0], tooltip="Click to select").add_to(m)

# Show the map and let users select a point
map_data = st_folium(m, width=700, height=500)

# Get user-selected latitude and longitude
if map_data and map_data['last_clicked']:
    latitude = map_data['last_clicked']['lat']
    longitude = map_data['last_clicked']['lng']
else:
    latitude, longitude = 0.0, 0.0  # Default values

st.write(f"**Selected Latitude:** {latitude}")
st.write(f"**Selected Longitude:** {longitude}")

# Input for number of stations
nst = st.number_input("Enter Number of Stations (Nst)", min_value=1, step=1)

# Predict button
if st.button("Predict Earthquake Magnitude"):
    if latitude == 0.0 and longitude == 0.0:
        st.error("Please select a valid location on the map.")
    else:
        input_data = np.array([[latitude, longitude, nst]])
        predicted_mag = model.predict(input_data)[0]
        st.success(f"Predicted Earthquake Magnitude: {round(predicted_mag, 2)}")