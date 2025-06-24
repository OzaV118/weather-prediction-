import numpy as np
import pickle
import streamlit as st

weather_model = pickle.load(open('weather_model.sav', 'rb'))

st.set_page_config(page_title="Weather Prediction", layout="centered")
st.title(' Weather Type Prediction App')

# dropdwonlist 
cloud_cover_map = {"clear": 0, "partly cloudy": 1, "overcast": 2}
season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
location_map = {'inland': 0, 'coastal': 1, 'mountain': 2}
weather_map = {0: "Cloudy", 1: "Rainy", 2: "Sunny"}

col1, col2 = st.columns(2)
with col1:
    temperature = st.text_input(" Temperature (°C)")
with col2:
    humidity = st.text_input(" Humidity (%)")

col3, col4 = st.columns(2)
with col3:
    wind_speed = st.text_input(" Wind Speed (km/h)")
with col4:
    precipitation = st.text_input(" Precipitation (%)")


col5, col6 = st.columns(2)
with col5:
    atmospheric_pressure = st.text_input(" Atmospheric Pressure (hPa)")
with col6:
    uv_index = st.text_input(" UV Index")


visibility = st.text_input(" Visibility (km)")


st.markdown("  Environmental Context")
col7, col8, col9 = st.columns(3)
with col7:
    cloud_cover_input = st.selectbox(' Cloud Cover', list(cloud_cover_map.keys()))
with col8:
    season_input = st.selectbox(' Season', list(season_map.keys()))
with col9:
    location_input = st.selectbox(' Location', list(location_map.keys()))

# Convert dropdown values to encoded values
cloud_cover_value = cloud_cover_map[cloud_cover_input]
season_value = season_map[season_input]
location_value = location_map[location_input]


st.markdown(" Make a Prediction")
if st.button('Predict Weather Type'):
        # Convert numeric inputs
    input_data = [
            float(temperature),
            float(humidity),
            float(wind_speed),
            float(precipitation),
            cloud_cover_value,
            float(atmospheric_pressure),
            float(uv_index),
            season_value,
            float(visibility),
            location_value
        ]

    input_array = np.array(input_data).reshape(1, -1)

        # Predict and convert to label
    weather_num = weather_model.predict(input_array)[0]
    weather_type = weather_map.get(weather_num, f"Label {weather_num}")

    st.success(f"Predicted Weather Type: {weather_type}")