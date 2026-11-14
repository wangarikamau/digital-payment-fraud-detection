# Digital Payment Fraud Detection System

An end-to-end machine learning application that identifies potentially fraudulent digital payment transactions using an XGBoost classification model.

The project covers the complete machine learning deployment workflow, from model development and experiment tracking to model registration, API development, containerization, and frontend integration.

## Project Overview

Digital payment platforms process large volumes of transactions, making the identification of suspicious activity an important part of financial security.

This project uses the PaySim synthetic mobile-money transaction dataset to develop a fraud detection model and expose its predictions through a web application.

Users can submit transaction details through a Streamlit interface and receive:

- A fraud prediction.
- A risk score.
- A decision indicating whether the transaction should be flagged for review.

**Note:** This project uses synthetic data. Predictions are intended as review signals, not definitive confirmation of fraud.

## Technology Stack

  Component                 Technology                 Purpose 

Programming language        Python                  Model development and backend
Data processing             Pandas, NumPy           Data preparation
Machine learning            Scikit-learn, XGBoost   Model training and evaluation
Experiment tracking         MLflow               Tracking experiments and model registration
API                         FastAPI                Serving model predictions 
Frontend                    Streamlit              Transaction assessment interface 
API testing                 Pytest                 Automated backend testing 
Containerization            Docker                 Packaging application services 
Service orchestration       Docker Compose         Running frontend and backend together 

## System Architecture

The application follows this workflow:

1. **Data preparation:** PaySim transaction data is cleaned and prepared for model development.
2. **Model training:** Logistic Regression and XGBoost are evaluated, with XGBoost selected as the final model.
3. **Experiment tracking:** MLflow records model experiments, parameters and evaluation metrics.
4. **Model registration:** The selected model is registered in MLflow and exported for application use.
5. **Prediction API:** FastAPI loads the registered model and exposes a prediction endpoint.
6. **Frontend:** Streamlit provides an interface for submitting transactions and viewing assessment results.
7. **Containerization:** Docker packages the application, while Docker Compose runs the frontend and backend services.

### Architecture Flow

Streamlit Frontend → FastAPI Backend → Registered XGBoost Model → Prediction Result → Streamlit Frontend

## Model Development

Two classification algorithms were evaluated:

- Logistic Regression — baseline model.
- XGBoost — final selected model.

### Feature Selection

The final model uses four transaction features:

- Transaction type
- Transaction amount
- Original account balance before the transaction
- Destination account balance before the transaction

Post-transaction balance features and transaction identifiers were excluded to reduce the risk of data leakage.

### Data Splitting

The dataset was split chronologically using transaction time steps:

- Training: Steps 1–520
- Validation: Steps 521–632
- Testing: Steps 633–743

This approach evaluates the model on later transaction periods rather than randomly mixing earlier and later transactions.

### Final Model Performance

The selected XGBoost model achieved the following results on the held-out test set:

| Metric | Result |
|---|---:|
| PR-AUC (Average Precision) | 0.9278 |
| Precision | 0.9101 |
| Recall | 0.7564 |
| F1-score | 0.8262 |
| Decision threshold | 0.95 |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 88,121 | 93 |
| Actual Fraud | 303 | 941 |

### Evaluation Considerations

Fraud detection involves a highly imbalanced classification problem. Precision and recall were therefore prioritized over accuracy alone.

- Precision measures how many flagged transactions were actually fraudulent in the test dataset.
- Recall measures how many fraudulent transactions the model successfully identified.
- PR-AUC measures performance across different classification thresholds.

The decision threshold was selected using validation data before evaluating the final model on the test set.

**Limitations:** PaySim is a synthetic dataset. These results demonstrate model performance on the dataset and should not be interpreted as expected performance in a live financial institution.

## API Documentation

The backend is built with FastAPI and exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome endpoint |
| GET | `/health` | Checks API and model availability |
| POST | `/predict` | Assesses a transaction |

### Sample Prediction Request

```json
{
  "type": "TRANSFER",
  "amount": 1000,
  "oldbalanceOrg": 1000,
  "oldbalanceDest": 0
}
```

### Sample Prediction Response

```json
{
  "fraud_prediction": 1,
  "risk_score": 0.9925,
  "decision": "Flag for review",
  "threshold": 0.95
}
```

The risk score is the model's output score, not a calibrated probability.

Interactive API documentation is available at:

`http://localhost:8000/docs`

## Automated Testing

Pytest was used to test the FastAPI backend.

Seven automated tests were implemented:

1. Health endpoint returns a successful response.
2. Valid transactions receive predictions.
3. Negative transaction amounts are rejected.
4. Invalid transaction types are rejected.
5. Missing required fields are rejected.
6. Negative origin balances are rejected.
7. Negative destination balances are rejected.

**Test results: 7/7 passed.**

The tests were executed inside the Docker environment to ensure compatibility with the environment where the model successfully loads.

## Running the Application

### Prerequisites

- Python
- Docker Desktop
- Docker Compose

### Start the application

Clone the repository and navigate to the project directory.

```bash
git clone <your-repository-url>
cd fraud_detection_api
```

Build and start the application:

```bash
docker compose up --build
```

### Access the application

- Frontend: `http://localhost:8501`
- API: `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`

### Stop the application

```bash
docker compose down
```

### Run automated tests

```bash
docker compose run --rm -v "${PWD}/tests:/app/tests" -v "${PWD}/requirements-dev.txt:/app/requirements-dev.txt" api sh -c "pip install -r requirements-dev.txt && python -m pytest tests/ -v"
```