from flask import Flask, jsonify, render_template, request
from train import spam_detector

app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"message": "API online"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "text" not in data:
        return jsonify({"error": "Campo 'text' é obrigatório."}), 400

    message = data["text"]
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Digite uma mensagem para analisar."}), 400

    try:
        return jsonify(spam_detector(message)), 200
    except FileNotFoundError:
        return jsonify({"error": "Modelo não encontrado. Execute train.py antes de analisar mensagens."}), 503
    except Exception:
        app.logger.exception("Erro ao classificar mensagem")
        return jsonify({"error": "Não foi possível classificar a mensagem."}), 500


if __name__ == "__main__":
    app.run()
