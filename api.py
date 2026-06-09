from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

#define structure of incoming data
class PredictionInput(BaseModel):
    features: list[float] #ensures items inside are numbers

@app.get("/")
def home():
    return {"message": "Welcome to the Fraud Detection API!"}

@app.post("/predict")
def predict(input_data: PredictionInput): #use the pydantic model here
    features = np.array(input_data.features).reshape(1, -1) #access list using input_data.features
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return {"prediction": int(prediction[0]),
             "result": "Fraud" if prediction[0] == 1 else "Not Fraud"}
