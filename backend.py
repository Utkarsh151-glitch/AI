from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Allow frontend to talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for local testing)
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
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
async def predict(data: DiabetesInput):
    # Prepare input data
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, 
                            data.SkinThickness, data.Insulin, data.BMI, 
                            data.DiabetesPedigreeFunction, data.Age]])
    
    # Scale input (ensure consistency with training)
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of having diabetes

    return {
        "prediction": "likely to have diabetes" if prediction == 1 else "unlikely to have diabetes",
        "probability": round(probability, 2)
    }

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
