# /Kut/app.py
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---- Configuration ----
st.set_page_config(
    page_title="Kut Water AI", 
    page_icon="💧",
    layout="centered"
)

# ---- Model Loading ----
model = joblib.load("kut_dew_predictor.pkl")

# ---- Header ----
st.title("🌧️ Kut Water Harvesting Predictor")
st.caption("Optimize dew collection in Iraq's climate")

# ---- Sidebar Inputs ----
with st.sidebar:
    st.header("Weather Parameters")
    humidity = st.slider("Humidity (%)", 0, 100, 80)
    temp = st.slider("Temperature (°C)", 0, 50, 25)
    wind = st.slider("Wind Speed (km/h)", 0, 50, 10)

# Initialize yield_pred as None
yield_pred = None

# ---- Prediction ----
if st.button("Predict Yield", type="primary"):
    input_data = pd.DataFrame([[humidity, temp, wind]], 
                            columns=["relative_humidity_2m", "temperature_2m", "wind_speed_10m"])
    yield_pred = model.predict(input_data)[0]
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Yield", f"{yield_pred:.2f} L/m²/day")
    with col2:
        # Dynamic visualization instead of static image
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.barh(['Yield'], [yield_pred], color='#1f77b4')
        ax.set_xlim(0, 2.5)  # Adjust based on expected yield range
        st.pyplot(fig)
    
    # Success message with conditional emoji
    if yield_pred > 0.3:  # Lowered threshold
        st.success(f"💧 Good collection potential!")
    elif yield_pred > 0.1:
        st.warning(f"🌫️ Minimal yield")
    else:
        st.error(f"❌ Unlikely to collect dew")