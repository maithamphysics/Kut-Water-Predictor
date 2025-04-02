# /Kut/data_collector.py
import requests
import pandas as pd

def fetch_kut_weather():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 32.51, 
        "longitude": 45.77,
        "hourly": ["relative_humidity_2m", "temperature_2m", "wind_speed_10m"],
        "start_date": "2020-01-01",
        "end_date": pd.Timestamp.now().strftime("%Y-%m-%d")
    }
    response = requests.get(url, params=params)
    return pd.DataFrame(response.json()["hourly"])

if __name__ == "__main__":
    df = fetch_kut_weather()
    df.to_csv("kut_weather_raw.csv", index=False)
    
# Add at the end of data_collector.py
print("✅ Raw data saved. Next run: python preprocess_data.py")