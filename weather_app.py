# build by Oza V 
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# === Load model with caching ===
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("weather_model.sav", "rb"))
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'weather_model.sav' not found.")
        st.stop()

weather_model = load_model()

# === Encoding Maps ===
cloud_cover_map = {"clear": 0, "partly cloudy": 1, "overcast": 2}
season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
location_map = {'inland': 0, 'coastal': 1, 'mountain': 2}
weather_map = {0: "Cloudy", 1: "Rainy", 2: "Sunny", 3: "Overcast", 4: "Snowy"}

# === Page Config ===
st.set_page_config(page_title="🌤️ Weather Predictor", layout="centered")
st.title("🌦️ Smart Weather Type Predictor")

# === Sidebar Info ===
with st.sidebar:
    st.header("📘 About App")
    st.write("Predicts weather type based on environmental conditions using a pre-trained ML model.")
    uploaded_file = st.file_uploader("📁 Upload CSV for Bulk Prediction", type=["csv"])

# === Input Section ===
st.header("🔧 Enter Weather Conditions")

col1, col2 = st.columns(2)
with col1:
    temperature = st.text_input("🌡️ Temperature (°C)", "25.0")
    wind_speed = st.text_input("💨 Wind Speed (km/h)", "10.0")
    pressure = st.text_input("📈 Atmospheric Pressure (hPa)", "1013.0")
    visibility = st.text_input("🔭 Visibility (km)", "10.0")
with col2:
    humidity = st.text_input("💧 Humidity (%)", "50.0")
    precipitation = st.text_input("🌧️ Precipitation (%)", "10.0")
    uv_index = st.text_input("🌞 UV Index", "5.0")

col3, col4, col5 = st.columns(3)
with col3:
    cloud_cover = st.selectbox("☁️ Cloud Cover", list(cloud_cover_map.keys()))
with col4:
    season = st.selectbox("🗓️ Season", list(season_map.keys()))
with col5:
    location = st.selectbox("📍 Location", list(location_map.keys()))

# === Prediction Function ===
def predict_weather(input_data):
    arr = np.array(input_data).reshape(1, -1)
    prediction = weather_model.predict(arr)[0]
    try:
        confidence = np.max(weather_model.predict_proba(arr))
    except:
        confidence = None
    return prediction, confidence

# === Predict Button ===
if st.button("🔍 Predict Weather"):
    try:
        input_data = [
            float(temperature),
            float(humidity),
            float(wind_speed),
            float(precipitation),
            cloud_cover_map[cloud_cover],
            float(pressure),
            float(uv_index),
            season_map[season],
            float(visibility),
            location_map[location]
        ]

        label, conf = predict_weather(input_data)
        label_text = weather_map.get(label, f"Label {label}")

        st.success(f"🌤️ Prediction: **{label_text}**")
        if conf is not None:
            st.progress(int(conf * 100))
            st.caption(f"Confidence: **{conf*100:.2f}%**")

        with st.expander("📊 Input Summary"):
            st.json({
                "Temperature": temperature,
                "Humidity": humidity,
                "Wind Speed": wind_speed,
                "Precipitation": precipitation,
                "Cloud Cover": cloud_cover,
                "Pressure": pressure,
                "UV Index": uv_index,
                "Season": season,
                "Visibility": visibility,
                "Location": location
            })

        # Report Download
        df = pd.DataFrame([{
            "Prediction": label_text,
            "Confidence": f"{conf*100:.2f}%" if conf else "N/A",
            "Temperature": temperature,
            "Humidity": humidity,
            "Wind Speed": wind_speed,
            "Precipitation": precipitation,
            "Cloud Cover": cloud_cover,
            "Pressure": pressure,
            "UV Index": uv_index,
            "Season": season,
            "Visibility": visibility,
            "Location": location
        }])
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download Report", csv, "prediction_report.csv", mime="text/csv")

    except ValueError:
        st.warning("⚠️ Please enter valid numeric values in all input fields.")

# === Bulk Prediction ===
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df["Cloud Cover"] = df["Cloud Cover"].map(cloud_cover_map)
        df["Season"] = df["Season"].map(season_map)
        df["Location"] = df["Location"].map(location_map)

        features = [
            "Temperature", "Humidity", "Wind Speed", "Precipitation",
            "Cloud Cover", "Pressure", "UV Index", "Season", "Visibility", "Location"
        ]
        preds = weather_model.predict(df[features])
        try:
            probs = weather_model.predict_proba(df[features])
            confs = np.max(probs, axis=1)
        except:
            confs = [None] * len(preds)

        df["Prediction"] = [weather_map.get(p, p) for p in preds]
        df["Confidence"] = [f"{c*100:.2f}%" if c else "N/A" for c in confs]

        st.subheader("📋 Results")
        st.dataframe(df)

        result_csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Bulk Results", result_csv, "bulk_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing uploaded file: {e}")
