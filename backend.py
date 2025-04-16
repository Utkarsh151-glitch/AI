from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and scaler
model = joblib.load("diabetes_model.pkl")  # Load trained XGBoost model
scaler = joblib.load("scaler.pkl")  # Load pre-trained scaler

# Define input schema
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    Insulin: float
    BMI: float
    Age: float

@app.post("/predict")
async def predict(data: DiabetesInput):
    # Convert input data into a structured NumPy array
    input_df = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, 
                        data.Insulin, data.BMI, data.Age]])

    # Convert to Pandas DataFrame to preserve feature names
    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "Insulin", "BMI", "Age"]
    input_df = pd.DataFrame(input_df, columns=feature_names)

    # Print raw input values for debugging
    print("Raw input values:", input_df)

    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Print scaled input values for debugging
    print("Scaled input values:", input_scaled)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Get probability of diabetes

    # Print raw model outputs for debugging
    print("Raw prediction:", prediction)
    print("Raw probability:", probability)

    # Convert NumPy types to Python native types
    prediction = int(prediction)
    probability = float(probability)

    return {
        "prediction": "likely to have diabetes" if prediction == 1 else "unlikely to have diabetes",
        "probability": round(probability, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
