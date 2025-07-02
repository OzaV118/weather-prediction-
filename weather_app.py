import numpy as np
import pickle
import streamlit as st

# Load the trained model
try:
    weather_model = pickle.load(open("weather_model.sav", "rb"))
except FileNotFoundError:
    st.error("❌ Model file 'weather_model.sav' not found. Please make sure it is in the app directory.")
    st.stop()

# Encoding maps
cloud_cover_map = {"clear": 0, "partly cloudy": 1, "overcast": 2}
season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
location_map = {'inland': 0, 'coastal': 1, 'mountain': 2}
weather_map = {0: "Cloudy", 1: "Rainy", 2: "Sunny", 3: "Overcast", 4: "Snowy"}

# Page setup
st.set_page_config(page_title="Weather Prediction", layout="centered")
st.title('🌦️ Weather Type Prediction App')

# Default values
defaults = {
    "temperature": 25.0,
    "humidity": 50.0,
    "wind_speed": 10.0,
    "precipitation": 10.0,
    "atmospheric_pressure": 1013.0,
    "uv_index": 5.0,
    "visibility": 10.0,
    "cloud_cover": "clear",
    "season": "Summer",
    "location": "inland"
}

# Inputs
col1, col2 = st.columns(2)
with col1:
    temperature = st.number_input("Temperature (°C)", value=defaults["temperature"])
with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=defaults["humidity"])

col3, col4 = st.columns(2)
with col3:
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, value=defaults["wind_speed"])
with col4:
    precipitation = st.number_input("Precipitation (%)", min_value=0.0, max_value=100.0, value=defaults["precipitation"])

col5, col6 = st.columns(2)
with col5:
    atmospheric_pressure = st.number_input("Atmospheric Pressure (hPa)", value=defaults["atmospheric_pressure"])
with col6:
    uv_index = st.number_input("UV Index", min_value=0.0, value=defaults["uv_index"])

visibility = st.number_input("Visibility (km)", min_value=0.0, value=defaults["visibility"])

col7, col8, col9 = st.columns(3)
with col7:
    cloud_cover_input = st.selectbox("Cloud Cover", list(cloud_cover_map.keys()), index=0)
with col8:
    season_input = st.selectbox("Season", list(season_map.keys()), index=2)
with col9:
    location_input = st.selectbox("Location", list(location_map.keys()), index=0)

# Prediction function
def weather_prediction(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = weather_model.predict(input_array)
    return weather_map.get(prediction[0], f"Label {prediction[0]}")

# Predict button logic
if st.button("🔍 Predict Weather Type"):
    # Check if any inputs are different from defaults
    used_defaults = (
        temperature == defaults["temperature"]
        and humidity == defaults["humidity"]
        and wind_speed == defaults["wind_speed"]
        and precipitation == defaults["precipitation"]
        and atmospheric_pressure == defaults["atmospheric_pressure"]
        and uv_index == defaults["uv_index"]
        and visibility == defaults["visibility"]
        and cloud_cover_input == defaults["cloud_cover"]
        and season_input == defaults["season"]
        and location_input == defaults["location"]
    )

    if used_defaults:
        st.info("ℹ️ Using default values for prediction since no input was changed.")

    try:
        input_data = [
            float(temperature),
            float(humidity),
            float(wind_speed),
            float(precipitation),
            cloud_cover_map[cloud_cover_input],
            float(atmospheric_pressure),
            float(uv_index),
            season_map[season_input],
            float(visibility),
            location_map[location_input]
        ]
        result = weather_prediction(input_data)
        st.success(f"🌤️ Predicted Weather Type: **{result}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
