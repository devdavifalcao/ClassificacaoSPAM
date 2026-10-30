import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.content_type.startswith("text/html")
    assert b"Classificador de SPAM" in response.data


def test_health_route(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json == {"message": "API online"}


def test_predict_missing_text(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400


def test_predict_empty_text(client):
    response = client.post("/predict", json={"text": "  "})
    assert response.status_code == 400


def test_api_full_flow(client, monkeypatch):
    def fake_spam_detector(text):
        return {"label": 0, "label_name": "ham", "probabilities": {0: 0.9, 1: 0.1}}

    monkeypatch.setattr("app.spam_detector", fake_spam_detector)
    response = client.post("/predict", json={"text": "hello friend"})
    assert response.status_code == 200
    assert response.json["label_name"] == "ham"
