import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Backend API for predicting fraudulent credit card transactions.",
    version="1.0"
)

# 2. Load the trained model and scaler safe from your desktop folder
try:
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading model files: {e}")
    # Fallback placeholders if loading fails initially during testing
    model = None
    scaler = None

# 3. Define the input data structure matching your dataset features
class TransactionData(BaseModel):
    # Expects an array or list of your numerical transaction features
    features: list[float]

@app.get("/")
def home():
    return {"message": "Fraud Detection API Backend is running perfectly!"}

@app.post("/predict")
def predict_fraud(data: TransactionData):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=500, 
            detail="Model or Scaler files are missing on the backend server."
        )
    
    try:
        # Convert inputs into the correct numpy format for inference
        input_features = np.array(data.features).reshape(1, -1)
        
        # Scale features using your saved scaler
        scaled_features = scaler.transform(input_features)
        
        # Run prediction (0 = Legitimate, 1 = Fraudulent)
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        
        return {
            "is_fraud": int(prediction),
            "fraud_probability": float(probability),
            "status": "Success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")