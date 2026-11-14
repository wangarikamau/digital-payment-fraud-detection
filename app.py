
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal


# --------------------------------------------------
# 1. Load registered model from local project folder
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "registered_model"

model = mlflow.sklearn.load_model(str(MODEL_PATH))

THRESHOLD = 0.95


# --------------------------------------------------
# 2. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Digital Payment Fraud Detection API",
    description="API for predicting potentially fraudulent transactions.",
    version="1.0"
)


# --------------------------------------------------
# 3. Define transaction input validation
# --------------------------------------------------

class Transaction(BaseModel):

    type: Literal[
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]

    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)


# --------------------------------------------------
# 4. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Digital Payment Fraud Detection API is running",
        "model": "paysim-fraud-detection",
        "version": "1"
    }


# --------------------------------------------------
# 5. Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "paysim-fraud-detection",
        "model_version": "1"
    }


# --------------------------------------------------
# 6. Fraud prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_fraud(transaction: Transaction):

    input_data = pd.DataFrame([{
        "type": transaction.type,
        "amount": transaction.amount,
        "oldbalanceOrg": transaction.oldbalanceOrg,
        "oldbalanceDest": transaction.oldbalanceDest
    }])

    fraud_probability = model.predict_proba(input_data)[0][1]

    prediction = int(fraud_probability >= THRESHOLD)

    decision = (
        "Flag for review"
        if prediction == 1
        else "Not flagged by model"
    )

    return {
        "fraud_prediction": prediction,
        "risk_score": round(float(fraud_probability), 4),
        "decision": decision,
        "threshold": THRESHOLD
    }
