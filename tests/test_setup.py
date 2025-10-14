import pytest
from fastapi.testclient import TestClient
from app.serve import app

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Initialize test client
client = TestClient(app)

# -----------------------------
# UNIT TESTS (direct model tests)
# -----------------------------
def test_model_inference_basic():
    """Ensure model returns valid sentiment label for simple text."""
    response=client.get("/predict?text=I love this movie")
    data=response.json()

    assert response.status_code == 200
    assert data['sentiment'] in ['positive','negative']
    assert 0<=float(data["latency_ms"])<3000


# -----------------------------
# INTEGRATION TESTS (API-level)
# -----------------------------
@pytest.mark.parametrize(
    "text, expected_label",
    [
        ("I love this product","positive"),
        ("This is the worst thing ever","negative")
        ],
)

def test_api_response(text, expected_label):
    """Test that API returns correct label direction."""
    response=client.get(f"/predict?text={text}")
    data=response.json()

    assert response.status_code == 200
    assert "sentiment" in data
    
    # Ensure at least polarity consistency
    if "love" in text:
        assert data['sentiment']=='positive'
    elif "worst" in text:
        assert data['sentiment'] == 'negative'

def test_ping_endpoint():
    """Ping health endpoint"""
    response=client.get("/ping")
    assert response.status_code == 200
    assert "model_version" in response.json()

def test_switch_model_endpoint(monkeypatch):
    """Simulate model switch without breaking the app"""
    response = client.get("/switch_model?version=v1")
    assert response.status_code == 200
    assert "message" in response.json()