import pytest
from fastapi.testclient import TestClient
from app.serve import app

client = TestClient(app)

def test_ping_endpoint():
    """Health check should return 200."""
    response = client.get("/ping")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == 200


def test_model_inference_basic():
    """Ensure model returns valid sentiment label for simple text."""
    response = client.post(
        "/predict",
        json={"text": "I love this movie", "quantize": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "negative"]
    assert "latency_ms" in data
    assert "quantized" in data


@pytest.mark.parametrize(
    "text, expected_label",
    [
        ("I love this product", "positive"),
        ("This is the worst thing ever", "negative"),
    ],
)
def test_api_response(text, expected_label):
    """Test that API returns expected label direction."""
    response = client.post(
        "/predict",
        json={"text": text, "quantize": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == expected_label
