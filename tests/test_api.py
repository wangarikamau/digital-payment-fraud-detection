
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "paysim-fraud-detection"
    assert data["model_version"] == "1"


def test_valid_transaction_prediction():
    transaction = {
        "type": "TRANSFER",
        "amount": 1000,
        "oldbalanceOrg": 1000,
        "oldbalanceDest": 0
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    data = response.json()

    assert data["fraud_prediction"] in [0, 1]
    assert isinstance(data["risk_score"], (int, float))
    assert 0 <= data["risk_score"] <= 1
    assert data["decision"] in ["Flag for review", "No flag"]
    assert data["threshold"] == 0.95


def test_negative_amount_is_rejected():
    transaction = {
        "type": "TRANSFER",
        "amount": -100,
        "oldbalanceOrg": 1000,
        "oldbalanceDest": 0
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422


def test_invalid_transaction_type_is_rejected():
    transaction = {
        "type": "INVALID_TYPE",
        "amount": 1000,
        "oldbalanceOrg": 1000,
        "oldbalanceDest": 0
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422


def test_missing_field_is_rejected():
    transaction = {
        "type": "TRANSFER",
        "amount": 1000,
        "oldbalanceOrg": 1000
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422

def test_negative_origin_balance_is_rejected():
    transaction = {
        "type": "TRANSFER",
        "amount": 1000,
        "oldbalanceOrg": -500,
        "oldbalanceDest": 0
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422


def test_negative_destination_balance_is_rejected():
    transaction = {
        "type": "TRANSFER",
        "amount": 1000,
        "oldbalanceOrg": 1000,
        "oldbalanceDest": -500
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422