from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

LIMITE_CARACTERES = 10000


def spam_detector(content):
    # As bibliotecas de aprendizado de máquina só são carregadas durante a análise.
    from train import spam_detector as detectar

    return detectar(content)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/style.css")
def stylesheet():
    return send_from_directory("public", "style.css")


@app.get("/health")
def health():
    return jsonify({"message": "API online"}), 200


@app.errorhandler(413)
def requisicao_muito_grande(_error):
    return jsonify({"error": "A mensagem ultrapassa o limite permitido."}), 413


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "text" not in data:
        return jsonify({"error": "Envie um JSON com o campo 'text'."}), 400

    message = data["text"]
    if not isinstance(message, str):
        return jsonify({"error": "O campo 'text' precisa ser uma mensagem de texto."}), 400

    message = message.strip()
    if not message:
        return jsonify({"error": "Digite uma mensagem para analisar."}), 400

    if len(message) > LIMITE_CARACTERES:
        return jsonify({
            "error": f"A mensagem deve ter no máximo {LIMITE_CARACTERES} caracteres."
        }), 413

    try:
        resultado = spam_detector(message)
        if not isinstance(resultado, dict):
            raise TypeError("O classificador retornou um resultado inválido.")

        rotulo = resultado.get("label")
        probabilidades = resultado.get("probabilities")
        if rotulo not in (0, 1) or not isinstance(probabilidades, dict):
            raise ValueError("O classificador retornou dados incompletos.")

        return jsonify(resultado), 200
    except FileNotFoundError:
        app.logger.exception("Arquivos do modelo não foram encontrados")
        return jsonify({
            "error": "Os arquivos do modelo não estão disponíveis no servidor."
        }), 503
    except Exception:
        app.logger.exception("Erro ao carregar o modelo ou classificar a mensagem")
        return jsonify({
            "error": "Não foi possível classificar a mensagem. Tente novamente mais tarde."
        }), 500


if __name__ == "__main__":
    app.run()
