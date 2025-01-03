import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict_valid_input(client):
    # Test with valid input
    response = client.post("/predict", json={"text": "I love programming"})
    assert response.status_code == 200
    assert "label" in response.get_json()[0]  # Adjust based on model output

def test_predict_no_input(client):
    # Test with no input
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert response.get_json() == {"error": "No text provided"}
