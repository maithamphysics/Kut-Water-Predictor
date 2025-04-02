# /Kut/preprocess_data.py
import pandas as pd

def preprocess(raw_path="kut_weather_raw.csv"):
    df = pd.read_csv(raw_path)
    
    # 1. Parse timestamps and filter Kut's humid hours (10PM-4AM)
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"].dt.hour.isin([22, 23, 0, 1, 2, 3])]  # Peak dew hours
    
    # 2. Calculate dew potential (your physics formula)
    df["dew_yield"] = (df["relative_humidity_2m"] > 80) & \
                      (df["temperature_2m"] < 25) & \
                      (df["wind_speed_10m"] < 15)
    df["dew_yield"] = df["dew_yield"].astype(int) * 0.75  # 0.75L/m² when conditions met
    
    # 3. Save processed data
    df.to_csv("kut_weather_processed.csv", index=False)
    return df

if __name__ == "__main__":
    preprocess()