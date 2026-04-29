# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Initialize App
app = FastAPI(title="Customer Segmentation API", version="1.0")

# Load Models
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "outputs" / "models"

try:
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    kmeans = joblib.load(MODELS_DIR / "kmeans_model.pkl")
    pca = joblib.load(MODELS_DIR / "pca_transformer.pkl")
except Exception as e:
    raise RuntimeError(f"Could not load models. Ensure Phase 5 has been run. Error: {e}")

# Persona Mapping based on our analysis
PERSONA_MAP = {
    0: "Satisfied One-Timer",
    1: "At-Risk Whale",
    2: "Detractor",
    3: "Champion"
}

# Define the expected JSON payload
class CustomerData(BaseModel):
    customer_id: str
    recency_days: int
    frequency: int
    monetary_value: float
    avg_review_score: float
    avg_installments: float

@app.get("/")
def health_check():
    return {"status": "active", "model": "K-Means RFM"}

@app.post("/predict-segment")
def predict_segment(customer: CustomerData):
    try:
        # Format input as DataFrame
        input_data = pd.DataFrame([{
            "Recency": customer.recency_days,
            "Frequency": customer.frequency,
            "Monetary": customer.monetary_value,
            "AvgReview": customer.avg_review_score,
            "AvgInstallments": customer.avg_installments
        }])
        
        # Log Transformation (Must match training pipeline!)
        for col in ["Recency", "Frequency", "Monetary"]:
            input_data[col] = np.log1p(input_data[col])
            
        # Scale Features
        scaled_data = scaler.transform(input_data)
        
        # Predict Cluster
        cluster = int(kmeans.predict(scaled_data)[0])
        
        # Get PCA Coordinates for UI Plotting
        coords = pca.transform(scaled_data)[0]
        
        return {
            "customer_id": customer.customer_id,
            "cluster_id": cluster,
            "persona": PERSONA_MAP.get(cluster, "Unknown"),
            "pca_coordinates": {"x": float(coords[0]), "y": float(coords[1])}
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))