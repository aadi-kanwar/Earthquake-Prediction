# import streamlit as st
# import folium
# from streamlit_folium import st_folium
# import numpy as np
# import joblib

# # Load trained model
# try:
#     model = joblib.load("earthquake_model1.pkl")
#     scaler = joblib.load("scaler.pkl")  # Load the same StandardScaler used in Jupyter Notebook
# except FileNotFoundError:
#     st.error("Model or scaler file not found. Please ensure both 'earthquake_model1.pkl' and 'scaler.pkl' are in the correct directory.")
#     st.stop()

# # Streamlit UI
# st.title("Earthquake Magnitude Prediction")
# st.markdown("### Select a location on the map and enter the number of stations")

# # Create a Folium Map
# m = folium.Map(location=[20, 0], zoom_start=5)
# folium.Marker([20, 0], tooltip="Click to select").add_to(m)

# # Show the map and let users select a point
# map_data = st_folium(m, width=700, height=500)

# # Get user-selected latitude and longitude
# if map_data and 'last_clicked' in map_data and map_data['last_clicked']:
#     latitude = map_data['last_clicked']['lat']
#     longitude = map_data['last_clicked']['lng']
# else:
#     latitude, longitude = None, None

# st.write(f"**Selected Latitude:** {latitude}")
# st.write(f"**Selected Longitude:** {longitude}")

# # Input for number of stations
# nst = st.number_input("Enter Number of Stations (Nst)", min_value=1, step=1)

# # Predict button
# if st.button("Predict Earthquake Magnitude"):
#     if latitude is None or longitude is None:
#         st.error("Please select a valid location on the map.")
#     else:
#         # Apply the same StandardScaler transformation
#         input_data = np.array([[latitude, longitude, nst]], dtype=np.float64)
#         input_scaled = scaler.transform(input_data)  # Scale input
#         predicted_mag = model.predict(input_scaled)[0]
#         st.success(f"Predicted Earthquake Magnitude: {round(predicted_mag, 2)}")

import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import joblib
import pandas as pd

# Load trained model
try:
    model = joblib.load("earthquake_model1.pkl")
    scaler = joblib.load("scaler.pkl")  # Load the same StandardScaler used in Jupyter Notebook
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure both 'earthquake_model1.pkl' and 'scaler.pkl' are in the correct directory.")
    st.stop()

# Streamlit UI
st.title("Earthquake Magnitude Prediction")
st.markdown("### Select a location on the map and enter the depth")

# Create a Folium Map
m = folium.Map(location=[20, 0], zoom_start=5)
folium.Marker([20, 0], tooltip="Click to select").add_to(m)

# Show the map and let users select a point
map_data = st_folium(m, width=700, height=500)

# Add custom CSS to reduce space
st.markdown(
    """
    <style>
    .stMarkdown {
        margin-top: -20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Get user-selected latitude and longitude
if map_data and 'last_clicked' in map_data and map_data['last_clicked']:
    latitude = map_data['last_clicked']['lat']
    longitude = map_data['last_clicked']['lng']
else:
    latitude, longitude = None, None

st.write(f"**Selected Latitude:** {latitude}")
st.write(f"**Selected Longitude:** {longitude}")

# Input for depth
depth = st.number_input("Enter Depth", min_value=-100.0, max_value=700.0, step=0.1)

# Predict button
if st.button("Predict Earthquake Magnitude"):
    if latitude is None or longitude is None:
        st.error("Please select a valid location on the map.")
    else:
        # Apply the same StandardScaler transformation
        input_data = pd.DataFrame([[latitude, longitude, depth]], columns=['latitude', 'longitude', 'depth'])
        input_scaled = scaler.transform(input_data)  # Scale input
        predicted_mag = model.predict(input_scaled)[0]
        st.success(f"Predicted Earthquake Magnitude: {round(predicted_mag, 2)}")