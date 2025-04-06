# Kut Water Prediction App
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

import numpy as np
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ---- Configuration ----
st.set_page_config(
    page_title="Kut Water AI PRO", 
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .st-eb {
        display: block !important;
    }
    .stPlot {
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# ---- Model Loading ----
@st.cache_resource(ttl=3600)
def load_model():
    """Load and validate the predictive model."""
    try:
        model_path = Path(__file__).parent / "models/kut_dew_predictor.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        with st.spinner("🧠 Loading AI model..."):
            model = joblib.load(model_path)
            
            # Quick validation
            test_pred = model.predict([[80, 25, 10]])[0]
            if not isinstance(test_pred, (float, int, np.floating)):
                raise ValueError("Invalid prediction type")
                
            st.success("✅ Model loaded successfully!")
            return model
            
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {str(e)}")
        return None

model = load_model()

# ---- Header Section ----
st.title("🌧️ KUT Dew Prediction System")
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
        
        The prediction is shown in liters per square meter per day (L/m²/day).
    """)

# ---- Sidebar Inputs ----
with st.sidebar:
    st.header("⚙️ Weather Parameters")
    
    # Input widgets
    humidity = st.number_input(
        "Humidity (%)", 
        min_value=0, max_value=100, value=80, step=1,
        help="Relative humidity at 2m height"
    )
    
    temp = st.number_input(
        "Temperature (°C)", 
        min_value=-10.0, max_value=50.0, value=25.0, step=0.1,
        help="Ambient temperature at 2m height"
    )
    
    wind = st.number_input(
        "Wind Speed (km/h)", 
        min_value=0.0, max_value=100.0, value=10.0, step=0.1,
        help="Wind speed at 10m height"
    )
    
    st.markdown("---")
    if model:
        st.success("✅ Model Ready")
    else:
        st.error("❌ Model Not Loaded")

# ---- Prediction Section ----
if model:
    if st.button("🚀 Predict Yield", type="primary"):
        with st.spinner("🔮 Calculating..."):
            time.sleep(0.5)  # Simulate processing
            
            # Create input DataFrame
            input_data = pd.DataFrame(
                [[humidity, temp, wind]], 
                columns=["relative_humidity_2m", "temperature_2m", "wind_speed_10m"]
            )
            
            try:
                # Make prediction
                yield_pred = float(model.predict(input_data)[0])
                
                # Display results
                st.balloons()
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(
                        "Predicted Yield", 
                        f"{yield_pred:.2f} L/m²/day",
                        delta="Optimal" if yield_pred > 0.3 else None
                    )
                    
                    if yield_pred > 0.5:
                        st.success("💦 Excellent collection potential!")
                    elif yield_pred > 0.3:
                        st.info("💧 Good collection potential")
                    elif yield_pred > 0.1:
                        st.warning("🌫️ Marginal yield")
                    else:
                        st.error("🏜️ Very low yield")
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.bar(['Dew Yield'], [yield_pred], 
                                color='skyblue', 
                                edgecolor='navy',
                                linewidth=1)
                    ax.bar_label(bars, fmt='%.2f L/m²/day', padding=3)
                    ax.set_ylabel('Liters per square meter per day', fontsize=10)
                    ax.set_title('Predicted Dew Collection', fontsize=12, pad=10)
                    ax.set_ylim(0, max(yield_pred * 1.5, 1.0))
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)
                
                # Show input parameters
                st.subheader("Input Parameters Used:")
                param_col1, param_col2, param_col3 = st.columns(3)
                
                with param_col1:
                    st.metric("Humidity", f"{humidity}%")
                
                with param_col2:
                    st.metric("Temperature", f"{temp}°C")
                
                with param_col3:
                    st.metric("Wind Speed", f"{wind} km/h")
                
                # Download option
                st.download_button(
                    "📥 Download Prediction Data",
                    input_data.to_csv(index=False),
                    "prediction_input.csv",
                    help="Download the input parameters used for this prediction"
                )
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.error("Please check your input values and try again.")

# ---- Footer ----
st.markdown("---")
st.caption("""
    🛠️ Technical Details:  
    - Model Type: Random Forest Regressor  
    - Training Data: Iraq Climate Data  
    - Version: 1.0.0  
    - Prediction Unit: Liters per square meter per day (L/m²/day)
""")