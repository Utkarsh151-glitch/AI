# Techx
🧠 Diabetes Prediction AI Model
📋 Overview

This project uses Machine Learning to predict the probability of a person having diabetes based on key medical parameters such as glucose level, blood pressure, BMI, insulin level, and more.
The model is built using XGBoost, with data preprocessing, normalization, class balancing (SMOTE), and hyperparameter optimization to ensure accurate predictions.

🚀 Features

🔍 Predicts diabetes risk with probability score

⚙️ Uses XGBoost Classifier for high performance

📊 Automatically handles missing and zero values

🧩 Balanced dataset using SMOTE to reduce bias

🧮 Feature importance visualization

💾 Trained model and scaler are saved for real-time prediction

📂 Dataset

The model is trained on an extended dataset (diabetes_synthetic_large.csv) that includes realistic medical data with the following parameters:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Outcome (0 = Non-Diabetic, 1 = Diabetic)

🧰 Technologies Used

Python 🐍

Pandas & NumPy

Scikit-learn

XGBoost

imbalanced-learn (SMOTE)

Matplotlib

Joblib

⚙️ How It Works

Load & Clean Data: Handles missing and invalid values automatically.

Preprocess: Standardizes numerical features for stable learning.

Train Model: Uses XGBoost with hyperparameter tuning via GridSearchCV.

Balance Data: Applies SMOTE to handle class imbalance.

Evaluate: Measures accuracy, precision, recall, and F1-score.

Predict: Takes user input and predicts the probability of diabetes.

💡 Example Prediction
Pregnancies: 2
Glucose: 135
BloodPressure: 80
SkinThickness: 25
Insulin: 130
BMI: 29.4
DiabetesPedigreeFunction: 0.65
Age: 45


Output:

The person is likely to have diabetes.
Probability: 72.5%

📈 Model Performance

Accuracy: ~80–85% (depending on dataset)

Tuned and balanced for real-world reliability.

🗂️ Files Included

diabetes_prevention.py → Model training and prediction script

diabetes_extended_large.csv → Dataset file

diabetes_model.pkl → Trained XGBoost model

scaler.pkl → StandardScaler object
