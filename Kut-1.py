import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import numpy as np
import matplotlib.style as mplstyle
import os  # For file existence check
from sklearn.linear_model import LinearRegression  # For predictions

# Configuration
COORDINATES = (32.51, 45.77)  # Kut, Iraq
START_DATE = '2019-01-01'
OUTPUT_PATH = "/Users/ghofranalazawi/Downloads/kut_water_analysis_final.csv"

def download_weather_data():
    """Fetch weather data from Open-Meteo API"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': COORDINATES[0],
        'longitude': COORDINATES[1],
        'start_date': START_DATE,
        'end_date': datetime.now().strftime('%Y-%m-%d'),
        'hourly': ['temperature_2m', 'relative_humidity_2m', 
                  'dew_point_2m', 'wind_speed_10m', 'precipitation'],
        'timezone': 'auto'
    }
    
    print("🌤️ Downloading weather data...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    return pd.DataFrame(response.json()['hourly'])

def process_data(df):
    """Clean and process raw weather data"""
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Resample to daily data
    daily = df.resample('D').agg({
        'temperature_2m': ['max', 'min', 'mean'],
        'relative_humidity_2m': 'mean',
        'wind_speed_10m': 'max',
        'precipitation': 'sum',
        'dew_point_2m': 'mean'
    })
    
    # Flatten multi-index columns
    daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
    return daily

def calculate_yield(daily):
    """Calculate water collection potential"""
    # Dew yield (1L/m²/day with 75% efficiency)
    dew_cond = (
        (daily['relative_humidity_2m_mean'] > 80) &
        (daily['temperature_2m_max'] - daily['temperature_2m_min'] > 5) &
        (daily['wind_speed_10m_max'] < 15) &
        (daily['precipitation_sum'] == 0)
    )
    daily['dew_yield'] = np.where(dew_cond, 0.75, 0)
    
    # Fog yield (3L/m²/day with 60% efficiency)
    fog_cond = (
        (daily['relative_humidity_2m_mean'] > 85) & 
        daily['wind_speed_10m_max'].between(5, 35)
    )
    daily['fog_yield'] = np.where(fog_cond, 1.8, 0)
    
    return daily

def analyze_results(df):
    """Generate insights and visualizations"""
    # Annual statistics (using 'YE' instead of deprecated 'Y')
    annual = df.resample('YE')[['dew_yield', 'fog_yield']].sum()
    annual.index = annual.index.year
    annual['total'] = annual.sum(axis=1)
    
    # Monthly patterns
    monthly = df.groupby(df.index.month)[['dew_yield', 'fog_yield']].mean()
    
    # Optimal conditions
    optimal = {
        'dew': df[df['dew_yield'] > 0][['relative_humidity_2m_mean', 'temperature_2m_max', 'temperature_2m_min']].mean(),
        'fog': df[df['fog_yield'] > 0][['relative_humidity_2m_mean', 'wind_speed_10m_max']].mean()
    }
    
    return annual, monthly, optimal

def visualize(annual, monthly, optimal):
    """Create professional visualizations"""
    # Use available matplotlib style
    available_styles = plt.style.available
    use_style = 'ggplot' if 'ggplot' in available_styles else 'default'
    plt.style.use(use_style)
    
    # Figure 1: Annual Yield
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    annual[['dew_yield', 'fog_yield']].plot.bar(
        stacked=True, ax=ax1, 
        title='Annual Water Collection Potential in Kut',
        ylabel='Liters per m²'
    )
    
    # Figure 2: Monthly Patterns
    monthly.plot.bar(ax=ax2, stacked=True,
                   title='Average Monthly Yield',
                   ylabel='Liters per m²/day')
    plt.tight_layout()
    fig1.savefig('kut_annual_yield.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Optimal Conditions
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    conditions = pd.DataFrame({
        'Dew': [
            optimal['dew']['relative_humidity_2m_mean'],
            optimal['dew']['temperature_2m_max'] - optimal['dew']['temperature_2m_min']
        ],
        'Fog': [
            optimal['fog']['relative_humidity_2m_mean'],
            optimal['fog']['wind_speed_10m_max']
        ]
    }, index=['Humidity (%)', 'Temp Drop/Wind Speed'])
    
    conditions.T.plot.bar(ax=ax3, rot=0)
    ax3.set_title('Optimal Collection Conditions')
    ax3.set_ylabel('Value')
    ax3.legend(title='Parameter')
    plt.tight_layout()
    fig3.savefig('kut_optimal_conditions.png', dpi=300)
    plt.close()

def main():
    try:
        # Data pipeline
        raw_data = download_weather_data()
        processed = process_data(raw_data)
        results = calculate_yield(processed)
        annual, monthly, optimal = analyze_results(results)
        
        # Save and visualize
        results.to_csv(OUTPUT_PATH)
        print(f"💾 Results saved to {OUTPUT_PATH}")
        
        visualize(annual, monthly, optimal)
        print("📊 Visualizations saved as PNG files")
        
        # Print key insights
        print("\n💧 Annual Water Collection Potential (L/m²):")
        print(annual)
        
        print("\n🌦️ Best Collection Months:")
        print(f"Dew: Month {monthly['dew_yield'].idxmax()}")
        print(f"Fog: Month {monthly['fog_yield'].idxmax()}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Suppress urllib3 warning
    import warnings
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
    
    main()
    

def predict_future_yield(historical_data_path):
    """Predict water yield for future dates"""
    # ... [Insert the full prediction code from previous answer here] ...
    
if __name__ == "__main__":
    # ... [Keep your existing main() call] ...
    
    # Add prediction after main analysis
    if os.path.exists(OUTPUT_PATH):
        predict_future_yield(OUTPUT_PATH)
    else:
        print("Skipping predictions: No historical data found")