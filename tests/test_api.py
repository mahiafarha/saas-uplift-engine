import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    """Verify service health and model loading."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

def test_persuadable_customer_prescriptive_decision():
    """Verify that a high-uplift profile receives TARGET_WITH_OFFER."""
    payload = {
        "user_id": "usr_test_persuadable",
        "plan_tier": "Enterprise",
        "monthly_fee": 299.0,
        "tenure_days": 210,
        "logins_last_30d": 22,
        "usage_drop_ratio": 0.55,
        "support_tickets": 1
    }
    response = client.post("/predict-intervention", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommendation"] == "TARGET_WITH_OFFER"
    assert data["predicted_uplift"] > 0
    assert data["expected_roi_dollars"] > 0

def test_sleeping_dog_prescriptive_decision():
    """Verify that a customer prone to negative uplift receives DO_NOT_DISTURB."""
    payload = {
        "user_id": "usr_test_sleeping_dog",
        "plan_tier": "Enterprise",
        "monthly_fee": 299.0,
        "tenure_days": 120,
        "logins_last_30d": 8,
        "usage_drop_ratio": 0.35,
        "support_tickets": 4
    }
    response = client.post("/predict-intervention", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommendation"] == "DO_NOT_DISTURB"
    assert data["predicted_uplift"] < 0
    assert data["expected_roi_dollars"] < 0

def test_invalid_plan_tier_validation():
    """Verify that unsupported plan tiers yield HTTP 400 Bad Request."""
    payload = {
        "user_id": "usr_bad_tier",
        "plan_tier": "NonExistentTier",
        "monthly_fee": 50.0,
        "tenure_days": 100,
        "logins_last_30d": 20,
        "usage_drop_ratio": 1.0,
        "support_tickets": 1
    }
    response = client.post("/predict-intervention", json=payload)
    assert response.status_code == 400
    assert "Invalid plan tier" in response.json()["detail"]