import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = pd.read_csv("diabetes.csv")

# Drop any completely empty rows
data.dropna(how='all', inplace=True)

# Ensure all columns are numeric (skip the first row if needed)
data = data.apply(pd.to_numeric, errors='coerce')  # Convert all values to numeric, invalid ones become NaN

# Drop any rows with missing values after conversion
data.dropna(inplace=True)


# Select relevant columns (ensure correctness of column names)
selected_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

data = data[selected_features + ["Outcome"]]

# Handle missing values by imputing median values
data.fillna(data.median(numeric_only=True), inplace=True)

# Split features and target
X = data[selected_features]
y = data["Outcome"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost model
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # Handle class imbalance
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")

# Function to predict diabetes and estimate time to diabetes onset
def predict_diabetes():
    print("Enter the following details:")
    user_input = []
    for feature in selected_features:
        value = float(input(f"{feature}: "))
        user_input.append(value)
    
    # Load model and scaler
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    
    # Scale input and predict
    user_input_scaled = scaler.transform([user_input])
    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0][1]  # Get probability of diabetes
    
    if prediction == 1:
        print("The person is likely to have diabetes.")
    else:
        print("The person is unlikely to have diabetes.")
        
        # Estimate years to diabetes onset
        age = user_input[selected_features.index("Age")]
        estimated_years = (1 - probability) * (80 - age)  # Assuming max age 80
        print(f"Based on current health data, the person may develop diabetes in approximately {estimated_years:.1f} years.")

# Uncomment the line below to enable user input mode
# predict_diabetes()
