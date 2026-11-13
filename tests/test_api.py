import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_pagina_inicial(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.content_type.startswith("text/html")
    assert b"Classificador de SPAM" in response.data


def test_arquivo_css_sem_alteracoes(client):
    response = client.get("/style.css")
    assert response.status_code == 200
    assert response.content_type.startswith("text/css")
    assert b":root" in response.data


def test_rota_de_saude(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json == {"message": "API online"}


def test_previsao_sem_campo_texto(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert "text" in response.json["error"]


def test_previsao_sem_json(client):
    response = client.post(
        "/predict",
        data="mensagem sem formato JSON",
        content_type="text/plain",
    )
    assert response.status_code == 400


def test_previsao_com_texto_vazio(client):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400


@pytest.mark.parametrize("valor", [None, 123, True, ["mensagem"]])
def test_previsao_com_tipo_invalido(client, valor):
    response = client.post("/predict", json={"text": valor})
    assert response.status_code == 400


def test_previsao_acima_do_limite(client):
    response = client.post("/predict", json={"text": "a" * 10001})
    assert response.status_code == 413


def test_fluxo_completo_da_api(client, monkeypatch):
    def classificador_falso(texto):
        assert texto == "mensagem legítima"
        return {
            "label": 0,
            "label_name": "ham",
            "probabilities": {0: 0.9, 1: 0.1},
        }

    monkeypatch.setattr("app.spam_detector", classificador_falso)
    response = client.post("/predict", json={"text": "  mensagem legítima  "})

    assert response.status_code == 200
    assert response.json["label"] == 0
    assert response.json["label_name"] == "ham"


def test_erro_quando_modelo_nao_existe(client, monkeypatch):
    def modelo_ausente(_texto):
        raise FileNotFoundError("modelo ausente")

    monkeypatch.setattr("app.spam_detector", modelo_ausente)
    response = client.post("/predict", json={"text": "mensagem de teste"})

    assert response.status_code == 503
    assert "modelo" in response.json["error"].lower()


def test_erro_interno_nao_expoe_detalhes(client, monkeypatch):
    def falha_interna(_texto):
        raise RuntimeError("detalhe interno que não deve aparecer")

    monkeypatch.setattr("app.spam_detector", falha_interna)
    response = client.post("/predict", json={"text": "mensagem de teste"})

    assert response.status_code == 500
    assert "detalhe interno" not in response.get_data(as_text=True)
