# /Kut/app.py
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import time  # For loading animation

# ---- Configuration ----
st.set_page_config(
    page_title="Kut Water AI PRO", 
    page_icon="💧",
    layout="wide",  # Changed to wide for better mobile responsiveness
    initial_sidebar_state="expanded"  # Better UX
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #e6f7ff;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---- Model Loading ----
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_model():
    model_path = Path(__file__).parent / "kut_dew_predictor.pkl"
    try:
        with st.spinner("🧠 Loading AI model..."):
            time.sleep(1)  # Simulate loading
            model = joblib.load(model_path)
            
            # Verify model integrity
            test_input = pd.DataFrame([[80, 25, 10]], 
                                    columns=["relative_humidity_2m", "temperature_2m", "wind_speed_10m"])
            test_pred = model.predict(test_input)[0]
            
            if not isinstance(test_pred, float):
                raise ValueError("Model returned invalid prediction format")
                
        return model
        
    except Exception as e:
        st.error(f"⚠️ Model Error: {str(e)}")
        st.info("Please check if 'kut_dew_predictor.pkl' exists in the correct directory")
        return None

model = load_model()

# ---- Header Section ----
# ---- Header Section ----
st.title("🌧️ THIS IS MODIFIACATION IN VS CODE MACBOOK NOWWW\\\\\11")  # Changed for testing
st.write("TEST LINE - PLEASE DELETE ME!!!")  # Should be visible
st.markdown("""
    **Optimize dew collection in arid climates**  
    *Scientific model for water sustainability in Iraq*
""")



with st.expander("ℹ️ How It Works"):
    st.write("""
        This AI model predicts daily dew yield based on:
        - Relative Humidity (%)
        - Temperature (°C)
        - Wind Speed (km/h)  
        The predictions help optimize water collection in dry regions.
    """)

# ---- Sidebar Inputs ----
with st.sidebar:
    st.header("⚙️ Weather Parameters")
    
    # Improved input widgets
    humidity = st.slider(
        "Humidity (%)", 
        0, 100, 80,
        help="Relative humidity at 2m height"
    )
    
    temp = st.slider(
        "Temperature (°C)", 
        -10, 50, 25,
        help="Ambient temperature at 2m height"
    )
    
    wind = st.slider(
        "Wind Speed (km/h)", 
        0, 100, 10,
        help="Wind speed at 10m height"
    )
    
    if model:
        st.success("✅ Model Ready")
    else:
        st.error("❌ Model Not Loaded")

# ---- Prediction Section ----
if model:
    if st.button("🚀 Predict Yield", type="primary", help="Click to run prediction"):
        with st.spinner("🔮 Calculating..."):
            time.sleep(0.5)  # Simulate processing
            
            input_data = pd.DataFrame([[humidity, temp, wind]], 
                                    columns=["relative_humidity_2m", "temperature_2m", "wind_speed_10m"])
            
            try:
                yield_pred = model.predict(input_data)[0]
                
                # ---- Results Display ----
                st.balloons()  # Celebration effect
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(
                        label="Predicted Yield", 
                        value=f"{yield_pred:.2f} L/m²/day",
                        delta="Optimal" if yield_pred > 0.3 else None
                    )
                    
                    # Collection feedback
                    if yield_pred > 0.5:
                        st.success("💦 Excellent collection potential!")
                    elif yield_pred > 0.3:
                        st.success("💧 Good collection potential")
                    elif yield_pred > 0.1:
                        st.warning("🌫️ Marginal yield - consider alternatives")
                    else:
                        st.error("🏜️ Very low yield expected")
                
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.bar(['Dew Yield'], [yield_pred], color='#1f77b4')
                    ax.set_ylim(0, max(2.5, yield_pred * 1.5))
                    ax.set_ylabel('Liters per m²/day')
                    ax.set_title('Prediction Visualization')
                    st.pyplot(fig)
                
                # ---- Data Export ----
                st.download_button(
                    label="📥 Download Prediction Data",
                    data=input_data.to_csv(index=False),
                    file_name="dew_prediction_input.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.code(f"Debug Info:\n{input_data.to_dict()}", language='json')

# ---- Footer ----
st.markdown("---")
st.caption("""
    🛠️ Technical Details:  
    - Model Type: Random Forest Regressor  
    - Training Data: 2015-2022 Iraq Climate Data  
    - Last Updated: 2023
""")# Redeploy trigger Fri Apr  4 21:44:08 BST 2025
