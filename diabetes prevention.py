import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import micropip

# Load the reduced dataset
data = pd.read_csv("diabetes_extended_large.csv")

# Define features and target
features = ["Pregnancies", "Glucose", "BloodPressure", "Insulin", "BMI", "Age"]
X = data[features]
y = data["Outcome"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 5],
    'scale_pos_weight': [1, 3]  # Adjusted for class imbalance
}

grid = GridSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), 
                     param_grid, cv=3, scoring='roc_auc')
grid.fit(X_train, y_train)

# Get best model
best_model = grid.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)

print(f"Best Model Parameters: {grid.best_params_}")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"AUC-ROC Score: {auc_score:.4f}")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(best_model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
