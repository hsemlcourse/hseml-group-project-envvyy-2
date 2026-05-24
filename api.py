"""FastAPI сервер для предиктов Twitch."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent / "src"))
from config import PROCESSED_DATA_DIR

app = FastAPI(
    title="Twitch Popularity Prediction API",
    description="API для предсказания популярности Twitch стримеров",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None
encoders = None


class PredictionRequest(BaseModel):
    stream_time_minutes: float = Field(..., description="Время стрима в минутах")
    peak_viewers: int = Field(..., description="Пиковое количество зрителей")
    average_viewers: int = Field(..., description="Среднее количество зрителей")
    followers: int = Field(..., description="Количество подписчиков")
    followers_gained: int = Field(..., description="Полученные подписчики")
    views_gained: int = Field(..., description="Полученные просмотры")
    partnered: bool = Field(..., description="Является ли партнёром")
    mature: bool = Field(..., description="Возрастной контент")
    language: str = Field(..., description="Язык стрима")


class PredictionResponse(BaseModel):
    watch_time_minutes: float
    watch_time_hours: float
    model_version: str


def load_model_and_preprocessors():
    global model, scaler, encoders
    try:
        model = joblib.load(PROCESSED_DATA_DIR / "best_model.pkl")
        scaler = joblib.load(PROCESSED_DATA_DIR / "scaler.pkl")
        encoders = joblib.load(PROCESSED_DATA_DIR / "encoders.pkl")
        print(f"[API] Model loaded: {type(model).__name__}")
        return True
    except Exception as e:
        print(f"[API] Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    load_model_and_preprocessors()


@app.get("/")
async def root():
    return {"message": "Twitch API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create DataFrame
        data = {
            "Stream time(minutes)": [request.stream_time_minutes],
            "Peak viewers": [request.peak_viewers],
            "Average viewers": [request.average_viewers],
            "Followers": [request.followers],
            "Followers gained": [request.followers_gained],
            "Views gained": [request.views_gained],
            "Partnered": [1 if request.partnered else 0],
            "Mature": [1 if request.mature else 0],
            "Language": [request.language],
        }
        df = pd.DataFrame(data)
        
        # Encode Language
        if encoders and "Language" in encoders:
            try:
                df["Language"] = encoders["Language"].transform(df["Language"])
            except:
                df["Language"] = 3
        
        # Feature engineering
        df["avg_viewers"] = df["Average viewers"] / (df["Stream time(minutes)"] + 1)
        df["follower_viewer_ratio"] = df["Followers"] / (df["Average viewers"] + 1)
        df["peak_avg_ratio"] = df["Peak viewers"] / (df["Average viewers"] + 1)
        df["follower_per_hour"] = df["Followers"] / (df["Stream time(minutes)"] / 60 + 1)
        df["watch_per_stream"] = df["Views gained"] / (df["Stream time(minutes)"] + 1) / 1000000
        df["viewers_per_follower"] = df["Average viewers"] / (df["Followers"] + 1) * 1000
        df["log_Followers"] = np.log1p(df["Followers"])
        df["log_Average_viewers"] = np.log1p(df["Average viewers"])
        df["log_Peak_viewers"] = np.log1p(df["Peak viewers"])
        df["log_Watch_time(Minutes)"] = np.log1p(df["Stream time(minutes)"] * df["Average viewers"])
        
        # Expected columns
        expected_cols = [
            "Stream time(minutes)", "Peak viewers", "Average viewers", 
            "Followers", "Followers gained", "Views gained", 
            "Partnered", "Mature", "Language",
            "avg_viewers", "follower_viewer_ratio", "peak_avg_ratio",
            "follower_per_hour", "watch_per_stream", "viewers_per_follower",
            "log_Followers", "log_Average_viewers", "log_Peak_viewers", "log_Watch_time(Minutes)"
        ]
        
        # Ensure all columns
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[expected_cols]
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        prediction = max(prediction, 0)
        
        return PredictionResponse(
            watch_time_minutes=float(prediction),
            watch_time_hours=float(prediction / 60),
            model_version="GradientBoosting_v1.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"model_type": type(model).__name__, "features_count": 19}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)