from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)


def spam_detector(content):
    # Importa as bibliotecas de machine learning somente quando houver uma análise.
    # Assim, uma falha no carregamento do modelo não impede a página inicial de abrir.
    from train import spam_detector as detect

    return detect(content)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/style.css")
def stylesheet():
    return send_from_directory("public", "style.css")


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
        app.logger.exception("Erro ao carregar o modelo ou classificar mensagem")
        return jsonify({"error": "Não foi possível classificar a mensagem. Verifique os logs do servidor."}), 500


if __name__ == "__main__":
    app.run()
