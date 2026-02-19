from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

# 1. Inisialisasi App
app = FastAPI(title="House Price Prediction API")

# 2. Load Model yang dihasilkan DVC
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Definisi format input data
class HouseFeatures(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int

@app.get("/")
def home():
    return {"message": "API Prediksi Harga Rumah aktif!"}

@app.post("/predict")
def predict(features: HouseFeatures):
    # Ubah input menjadi DataFrame (karena model dilatih dengan pandas)
    data = pd.DataFrame([features.dict()])
    
    # Melakukan prediksi
    prediction = model.predict(data)
    
    return {
        "prediction_price": float(prediction[0])
    }


