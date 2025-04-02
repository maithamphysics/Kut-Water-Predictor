# /Kut/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def train():
    # Load preprocessed data
    df = pd.read_csv("kut_weather_processed.csv")
    
    # Prepare features (X) and target (y)
    X = df[["relative_humidity_2m", "temperature_2m", "wind_speed_10m"]]
    y = df["dew_yield"]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, "kut_dew_predictor.pkl")
    print("✅ Model trained & saved!")

if __name__ == "__main__":
    train()