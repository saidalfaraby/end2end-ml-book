import pandas as pd
import json
import pickle
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train():
    # 1. Load Data
    df = pd.read_csv('data/harga_rumah.csv')
    
    # Pilih fitur sederhana untuk fokus pada engineering
    X = df[['area', 'bedrooms', 'bathrooms']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Inisialisasi MLflow Tracking
    mlflow.set_experiment("House_Price_Prediction")
    
    with mlflow.start_run():
        # 3. Training
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 4. Evaluasi
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Model Trained. R2 Score: {r2:.4f}")
        
        # 5. Log ke MLflow
        mlflow.log_param("features", ["area", "bedrooms", "bathrooms"])
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, "model")
        
        # 6. Simpan Model & Metrik untuk DVC
        os.makedirs('models', exist_ok=True)
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        metrics = {
            "mse": mse,
            "r2_score": r2
        }
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f)

if __name__ == "__main__":
    train()