from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

# Initialize app
app = FastAPI(title="Wine Quality Predictor")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],      # Allows all methods
    allow_headers=["*"],      # Allows all headers
)

# Load model on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(MODEL_PATH)

# Pydantic model for input validation
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/predict")
async def predict_quality(features: WineFeatures):
    X = [[
        features.fixed_acidity,
        features.volatile_acidity,
        features.citric_acid,
        features.residual_sugar,
        features.chlorides,
        features.free_sulfur_dioxide,
        features.total_sulfur_dioxide,
        features.density,
        features.pH,
        features.sulphates,
        features.alcohol,
    ]]
    pred = int(model.predict(X)[0])
    return {"quality": pred}

# For local testing: `uvicorn main:app --reload`
