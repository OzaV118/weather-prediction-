import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import pickle

# ----------------------- Load saved models and scaler -----------------------
# Load machine learning models using pickle
diabetes_model = pickle.load(open('diabities_svm.sav', 'rb'))  # Diabetes prediction model
heart_diseases_model = pickle.load(open('heart_diseases_svm_regreestion.sav', 'rb'))  # Heart disease model
heart_diseases_scaler = pickle.load(open('heart_diseases_scaler.sav', 'rb'))  # Scaler for heart data
parkinsons_model = pickle.load(open('parkinson_svc.sav', 'rb'))  # Parkinson's disease model
# === Load model with caching ===
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("weather_model.sav", "rb"))
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'weather_model.sav' not found in the directory.")
        st.stop()

weather_model = load_model()

# ----------------------- Utility Prediction Function -----------------------
# Function to safely process inputs and predict using model
def predict(model, inputs, scaler=None):
    try:
        inputs = [float(i) for i in inputs]  # Convert all inputs to float
        inputs = np.array(inputs).reshape(1, -1)  # Reshape for model
        if scaler:
            inputs = scaler.transform(inputs)  # Apply scaler if needed
        prediction = model.predict(inputs)  # Make prediction
        return prediction[0]
    except ValueError:
        st.error("Please enter valid numeric values in all fields.")  # Error if conversion fails
        return None

# ----------------------- Sidebar Navigation -----------------------
# Sidebar menu for selecting disease prediction page
# === Encoding Maps ===
cloud_cover_map = {"clear": 0, "partly cloudy": 1, "overcast": 2}
season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
location_map = {'inland': 0, 'coastal': 1, 'mountain': 2}
weather_map = {0: "Cloudy", 1: "Rainy", 2: "Sunny", 3: "Overcast", 4: "Snowy"}

# === Page Configuration ===
st.set_page_config(page_title="🌦️ Smart Weather Predictor", layout="centered", page_icon="🌤️")
st.title("🌦️ Smart Weather Type Prediction")

# === Sidebar ===
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['house', 'activity', 'heart', 'person'],
        default_index=0
    )

# ----------------------- Home Page -----------------------
# Introduction page
if selected == 'Home':
    st.title("Welcome to the Multiple Disease Prediction System")
    st.write("""
        This application uses machine learning models to predict the presence of:
        - **Diabetes**
        - **Heart Disease**
        - **Parkinson’s Disease**
        
        Please select a test from the sidebar and enter the required medical data.
    """)

# ----------------------- Diabetes Prediction Page -----------------------
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction System')

    # Form for diabetes input
    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)

        with col1:
            Pregnancies = st.text_input('Number of Pregnancies', placeholder='e.g., 2')
            Glucose = st.text_input('Glucose Level (mg/dL)', placeholder='e.g., 120')
            BloodPressure = st.text_input('Blood Pressure (mm Hg)', placeholder='e.g., 70')
            SkinThickness = st.text_input('Skin Thickness (mm)', placeholder='e.g., 20')

        with col2:
            Insulin = st.text_input('Insulin Level (mu U/ml)', placeholder='e.g., 80')
            BMI = st.text_input('BMI (kg/m²)', placeholder='e.g., 32.5')
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', placeholder='e.g., 0.5')
            Age = st.text_input('Age (years)', placeholder='e.g., 45')

        submitted = st.form_submit_button('Get Diabetes Result')

        if submitted:
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DiabetesPedigreeFunction, Age]
            result = predict(diabetes_model, input_data)

            if result == 0:
                st.success('The person is not diabetic.')
            elif result == 1:
                st.error('The person is diabetic.')

# ----------------------- Heart Disease Prediction Page -----------------------
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction System')

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age (years)', placeholder='e.g., 55')
            chol = st.text_input('Cholesterol (mg/dL)', placeholder='e.g., 240')
            exang = st.text_input('Exercise Angina (1 = Yes, 0 = No)', placeholder='e.g., 0')
            thal = st.text_input('Thal (1=Normal, 2=Fixed, 3=Reversible)', placeholder='e.g., 2')

        with col2:
            sex = st.text_input('Gender (1=Male, 0=Female)', placeholder='e.g., 1')
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dL (1 = Yes, 0 = No)', placeholder='e.g., 0')
            oldpeak = st.text_input('ST Depression (Oldpeak)', placeholder='e.g., 1.4')
            slope = st.text_input('Slope of ST segment (0–2)', placeholder='e.g., 1')

        with col3:
            cp = st.text_input('Chest Pain Type (0–3)', placeholder='e.g., 3')
            trestbps = st.text_input('Resting Blood Pressure (mm Hg)', placeholder='e.g., 130')
            restecg = st.text_input('Resting ECG (0–2)', placeholder='e.g., 1')
            thalach = st.text_input('Max Heart Rate Achieved', placeholder='e.g., 160')
            ca = st.text_input('Number of Major Vessels (0–3)', placeholder='e.g., 0')

        submitted = st.form_submit_button('Get Heart Disease Result')

        if submitted:
            input_data = [age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]
            result = predict(heart_diseases_model, input_data, scaler=heart_diseases_scaler)

            if result == 0:
                st.success('The person does NOT have a heart disease.')
            elif result == 1:
                st.error('The person HAS a heart disease.')

# ----------------------- Parkinson's Prediction Page -----------------------
if selected == 'Parkinsons Prediction':
    st.title("Parkinson's Disease Prediction System")

    with st.form("parkinsons_form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            fo = st.text_input('MDVP:Fo(Hz)', placeholder='e.g., 119.99')
            fhi = st.text_input('MDVP:Fhi(Hz)', placeholder='e.g., 157.3')
            flo = st.text_input('MDVP:Flo(Hz)', placeholder='e.g., 74.99')
            Jitter_percent = st.text_input('MDVP:Jitter(%)', placeholder='e.g., 0.00554')

        with col2:
            Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', placeholder='e.g., 0.00005')
            RAP = st.text_input('MDVP:RAP', placeholder='e.g., 0.0023')
            PPQ = st.text_input('MDVP:PPQ', placeholder='e.g., 0.0020')
            DDP = st.text_input('Jitter:DDP', placeholder='e.g., 0.0069')

        with col3:
            Shimmer = st.text_input('MDVP:Shimmer', placeholder='e.g., 0.03')
            Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', placeholder='e.g., 0.25')
            APQ3 = st.text_input('Shimmer:APQ3', placeholder='e.g., 0.012')
            HNR = st.text_input('HNR', placeholder='e.g., 21.0')

        with col4:
            APQ5 = st.text_input('Shimmer:APQ5', placeholder='e.g., 0.015')
            APQ = st.text_input('MDVP:APQ', placeholder='e.g., 0.02')
            DDA = st.text_input('Shimmer:DDA', placeholder='e.g., 0.03')
            NHR = st.text_input('NHR', placeholder='e.g., 0.02')

        submitted = st.form_submit_button('Get Parkinsons Result')

        if submitted:
            input_data = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                          Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR]
            result = predict(parkinsons_model, input_data)

            if result == 0:
                st.success("The person does not have Parkinson’s disease.")
            elif result == 1:
                st.error("The person has Parkinson’s disease.")
    st.header("ℹ️ About")
    st.markdown("This app uses an AI model to predict the weather type.")
    st.markdown("Built by [Your Name](https://github.com/yourprofile)")
    uploaded_file = st.file_uploader("📁 Upload CSV for Bulk Prediction", type=["csv"])

# === Input Sliders ===
st.header("🔧 Enter Weather Parameters")

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

col3, col4, col5 = st.columns(3)
with col3:
    cloud_cover_input = st.selectbox("☁️ Cloud Cover", list(cloud_cover_map.keys()))
with col4:
    season_input = st.selectbox("🌱 Season", list(season_map.keys()))
with col5:
    location_input = st.selectbox("📍 Location", list(location_map.keys()))

# === Prediction Function ===
def predict_weather(data):
    arr = np.array(data).reshape(1, -1)
    prediction = weather_model.predict(arr)[0]
    try:
        confidence = weather_model.predict_proba(arr)[0]
        top_conf = np.max(confidence)
    except:
        top_conf = None
    return prediction, top_conf

# === Prediction Trigger ===
if st.button("🔍 Predict Weather Type"):
    input_data = [
        temperature,
        humidity,
        wind_speed,
        precipitation,
        cloud_cover_map[cloud_cover_input],
        atmospheric_pressure,
        uv_index,
        season_map[season_input],
        visibility,
        location_map[location_input]
    ]

    label, conf = predict_weather(input_data)
    label_text = weather_map.get(label, f"Label {label}")

    st.success(f"🌤️ Predicted Weather Type: **{label_text}**")
    if conf:
        st.progress(int(conf * 100))
        st.caption(f"Confidence: **{conf * 100:.2f}%**")

    # Live Input Summary
    with st.expander("📊 Input Summary"):
        st.write({
            "Temperature": temperature,
            "Humidity": humidity,
            "Wind Speed": wind_speed,
            "Precipitation": precipitation,
            "Cloud Cover": cloud_cover_input,
            "Pressure": atmospheric_pressure,
            "UV Index": uv_index,
            "Season": season_input,
            "Visibility": visibility,
            "Location": location_input
        })

    # CSV Report Download
    result_df = pd.DataFrame([{
        "Prediction": label_text,
        "Confidence": f"{conf * 100:.2f}%" if conf else "N/A",
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind Speed": wind_speed,
        "Precipitation": precipitation,
        "Cloud Cover": cloud_cover_input,
        "Pressure": atmospheric_pressure,
        "UV Index": uv_index,
        "Season": season_input,
        "Visibility": visibility,
        "Location": location_input
    }])
    csv = result_df.to_csv(index=False).encode()
    st.download_button("📥 Download Prediction Report", csv, "weather_prediction.csv", mime="text/csv")

# === Bulk CSV Prediction ===
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

        predictions = weather_model.predict(df[features])
        try:
            probs = weather_model.predict_proba(df[features])
            confs = np.max(probs, axis=1)
        except:
            confs = [None] * len(predictions)

        df["Prediction"] = [weather_map.get(p, p) for p in predictions]
        df["Confidence"] = [f"{c*100:.2f}%" if c else "N/A" for c in confs]

        st.subheader("📋 Bulk Prediction Results")
        st.dataframe(df)

        csv_bulk = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Bulk Results", csv_bulk, "bulk_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
