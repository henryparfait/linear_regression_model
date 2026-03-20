
"""
predict.py – Diabetes Progression Prediction Script
Loads the best-trained model and makes a prediction on new patient data.
Usage: python predict.py
"""

import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature order: age, sex, bmi, bp, s2, s3, s4, s5, s6
# (s1 was dropped during feature engineering)

# Example: single patient data (raw, pre-standardization)
sample_patient = np.array([[
    0.0453,   # age
   -0.0447,   # sex
   -0.0058,   # bmi
   -0.0159,   # bp
   -0.0037,   # s2
    0.0081,   # s3
   -0.0396,   # s4
   -0.0031,   # s5
    0.0112,   # s6
]])

# Scale and predict
sample_scaled = scaler.transform(sample_patient)
prediction = model.predict(sample_scaled)[0]

print(f"Predicted diabetes progression score: {prediction:.2f}")
