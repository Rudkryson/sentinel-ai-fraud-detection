import pytest
from fastapi.testclient import TestClient

def test_rate_limiting(client: TestClient):
    # Test rate limiting on login endpoint
    # The limit is 5/minute
    for _ in range(5):
        response = client.post("/api/auth/login", data={"username": "rate@test.com", "password": "password"})
        assert response.status_code != 429
    
    # 6th request should be rate limited
    response = client.post("/api/auth/login", data={"username": "rate@test.com", "password": "password"})
    assert response.status_code == 429

def test_user_validation(client: TestClient):
    # Test short password
    response = client.post(
        "/api/auth/register",
        json={"email": "valid@test.com", "password": "short"}
    )
    assert response.status_code == 422
    assert "password" in response.text.lower()

def test_transaction_validation(client: TestClient):
    # Register and login to get a token
    client.post(
        "/api/auth/register",
        json={"email": "txn_test@test.com", "password": "strongpassword123"}
    )
    login_res = client.post(
        "/api/auth/login",
        data={"username": "txn_test@test.com", "password": "strongpassword123"}
    )
    token = login_res.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    valid_txn = {
        "amount": 5000.0,
        "transaction_hour": 14,
        "transaction_day": 3,
        "location": "international",
        "transaction_type": "wire_transfer",
        "transaction_freq_7d": 1,
        "avg_amount_7d": 200.0,
        "amount_deviation": 24.0,
        "is_night": 0,
        "is_weekend": 0
    }

    # Invalid amount <= 0
    invalid_txn_amount = valid_txn.copy()
    invalid_txn_amount["amount"] = -50.0
    res = client.post("/api/predict/", json=invalid_txn_amount, headers=headers)
    assert res.status_code == 422
    assert "amount" in res.text.lower()

    # Invalid hour
    invalid_txn_hour = valid_txn.copy()
    invalid_txn_hour["transaction_hour"] = 25
    res = client.post("/api/predict/", json=invalid_txn_hour, headers=headers)
    assert res.status_code == 422
    assert "transaction_hour" in res.text.lower()
