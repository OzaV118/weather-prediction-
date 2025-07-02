import numpy as np
import pickle
import streamlit as st

# === Cached Model Loading ===
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("weather_model.sav", "rb"))
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'weather_model.sav' not found in the directory.")
        st.stop()

weather_model = load_model()

# === Label & Category Maps ===
cloud_cover_map = {"clear": 0, "partly cloudy": 1, "overcast": 2}
season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
location_map = {'inland': 0, 'coastal': 1, 'mountain': 2}
weather_map = {0: "Cloudy", 1: "Rainy", 2: "Sunny", 3: "Overcast", 4: "Snowy"}

# === Page Settings ===
st.set_page_config(page_title="🌦️ Smart Weather Predictor", layout="centered")
st.title("🌦️ Smart Weather Type Prediction")

st.markdown("📝 Adjust the parameters or use the defaults. Hit **Predict** to see results.")

# === Input Section ===
st.header("🔧 Weather Conditions")

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature (°C)", -30.0, 50.0, 25.0)
    wind_speed = st.slider("Wind Speed (km/h)", 0.0, 150.0, 10.0)
    atmospheric_pressure = st.slider("Pressure (hPa)", 900.0, 1100.0, 1013.0)
    visibility = st.slider("Visibility (km)", 0.0, 50.0, 10.0)

with col2:
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
    precipitation = st.slider("Precipitation (%)", 0.0, 100.0, 10.0)
    uv_index = st.slider("UV Index", 0.0, 15.0, 5.0)

# === Dropdowns ===
st.subheader("🌍 Environmental Factors")
col3, col4, col5 = st.columns(3)
with col3:
    cloud_cover_input = st.selectbox("☁️ Cloud Cover", list(cloud_cover_map.keys()))
with col4:
    season_input = st.selectbox("🌱 Season", list(season_map.keys()))
with col5:
    location_input = st.selectbox("📍 Location", list(location_map.keys()))

# === Prediction Logic ===
def predict_weather(data):
    arr = np.array(data).reshape(1, -1)
    prediction = weather_model.predict(arr)[0]

    try:
        confidence = weather_model.predict_proba(arr)[0]
        top_conf = np.max(confidence)
    except AttributeError:
        top_conf = None

    return prediction, top_conf

# === Predict Button ===
if st.button("🔍 Predict Weather Type"):
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

    label, conf = predict_weather(input_data)
    label_text = weather_map.get(label, f"Label {label}")

    st.success(f"🌤️ Predicted Weather Type: **{label_text}**")
    if conf:
        st.progress(int(conf * 100))
        st.caption(f"Model confidence: **{conf * 100:.2f}%**")

# === Future Placeholder ===
st.divider()
st.markdown("🧠 *Coming Soon: Explanation of what influenced this prediction using SHAP.*")
