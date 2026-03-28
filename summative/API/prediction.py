"""
Diabetes Progression Prediction API
FastAPI application that serves predictions using the best-trained model from Task 1.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# ── Initialize FastAPI App ───────────────────────────────────────────────────
app = FastAPI(
    title="Diabetes Progression Prediction API",
    description=(
        "Predicts diabetes disease progression one year after baseline "
        "using patient clinical metrics. Built as part of the AI in Healthcare mission."
    ),
    version="1.0.0",
)

# ── CORS Middleware Configuration ────────────────────────────────────────────
# Configured with specific origins rather than wildcard (*) for security.
# Allows the Flutter app and common development origins to access the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8080",
        "http://localhost:3000",
        "http://localhost:5000",
        "http://127.0.0.1",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
)

# ── Pydantic Input Schema with Data Types and Range Constraints ──────────────
class DiabetesInput(BaseModel):
    """
    Input schema for diabetes progression prediction.
    Each feature has an enforced data type (float) and realistic range constraints
    based on the diabetes dataset from sklearn (mean-centered and scaled values).
    """
    age: float = Field(
        ...,
        ge=-0.15, le=0.15,
        description="Age (normalized). Typical range: -0.11 to 0.11"
    )
    sex: float = Field(
        ...,
        ge=-0.07, le=0.07,
        description="Sex indicator (normalized). Values around -0.045 or 0.051"
    )
    bmi: float = Field(
        ...,
        ge=-0.10, le=0.20,
        description="Body mass index (normalized). Typical range: -0.09 to 0.17"
    )
    bp: float = Field(
        ...,
        ge=-0.15, le=0.15,
        description="Average blood pressure (normalized). Typical range: -0.11 to 0.13"
    )
    s2: float = Field(
        ...,
        ge=-0.20, le=0.25,
        description="Low-density lipoproteins (normalized). Typical range: -0.17 to 0.20"
    )
    s3: float = Field(
        ...,
        ge=-0.15, le=0.20,
        description="High-density lipoproteins (normalized). Typical range: -0.14 to 0.18"
    )
    s4: float = Field(
        ...,
        ge=-0.15, le=0.20,
        description="Total cholesterol / HDL ratio (normalized). Typical range: -0.08 to 0.15"
    )
    s5: float = Field(
        ...,
        ge=-0.20, le=0.20,
        description="Log of serum triglycerides (normalized). Typical range: -0.13 to 0.15"
    )
    s6: float = Field(
        ...,
        ge=-0.15, le=0.20,
        description="Blood sugar level (normalized). Typical range: -0.14 to 0.14"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "age": 0.0453,
                "sex": -0.0447,
                "bmi": -0.0058,
                "bp": -0.0159,
                "s2": -0.0037,
                "s3": 0.0081,
                "s4": -0.0396,
                "s5": -0.0031,
                "s6": 0.0112,
            }
        }


# ── Pydantic Output Schema ──────────────────────────────────────────────────
class PredictionOutput(BaseModel):
    prediction: float = Field(
        ..., description="Predicted diabetes disease progression score"
    )
    model_used: str = Field(
        ..., description="Name of the model used for prediction"
    )


# ── Model Loading ────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")


def load_model_and_scaler():
    """Load the saved model and scaler from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        train_and_save_model()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def train_and_save_model():
    """Train the model from scratch and save it. Used for initial setup and retraining."""
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target

    # Drop s1 (multicollinearity with s2) — same as Task 1
    df = df.drop(columns=['s1'])

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained. Test MSE: {mse:.2f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler, mse


# Load model at startup
model, scaler = load_model_and_scaler()


# ── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Diabetes Progression Prediction API",
        "mission": "Applying AI Tools in Healthcare",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/retrain": "POST - Retrain the model with fresh data",
            "/docs": "GET - Swagger UI Documentation",
        },
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: DiabetesInput):
    """
    Make a diabetes progression prediction.

    Takes 9 patient clinical features as input and returns the predicted
    disease progression score using the best-trained model (Random Forest).
    """
    global model, scaler

    try:
        # Convert input to array in the correct feature order
        input_array = np.array([[
            data.age, data.sex, data.bmi, data.bp,
            data.s2, data.s3, data.s4, data.s5, data.s6
        ]])

        # Scale the input
        input_scaled = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        return PredictionOutput(
            prediction=round(float(prediction), 2),
            model_used="Random Forest Regressor (100 trees, max_depth=5)"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/retrain")
def retrain():
    """
    Retrain the model with the latest data.

    This endpoint triggers a full model retraining pipeline:
    loads the dataset, preprocesses, trains a new Random Forest model,
    evaluates it, and saves the updated model to disk.
    """
    global model, scaler

    try:
        model, scaler, mse = train_and_save_model()
        return {
            "message": "Model retrained successfully",
            "new_test_mse": round(mse, 2),
            "model": "Random Forest Regressor (100 trees, max_depth=5)",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")
