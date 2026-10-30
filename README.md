# Classificador de SPAM

Projeto em Python que usa TF-IDF e Naive Bayes para classificar o texto de e-mails como spam ou legítimo. A aplicação conta com uma API Flask e uma página para testar mensagens.

## Tecnologias

- Python
- Flask
- Pandas
- Scikit-learn
- NLTK (Snowball Stemmer)

## Como executar

1. Clone o repositório e entre na pasta do projeto.
2. Crie um ambiente virtual:

   ```bash
   python -m venv .venv
   ```

   No Windows, ative com:

   ```bash
   .venv\Scripts\activate
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Treine o modelo:

   ```bash
   python train.py
   ```

5. Inicie a aplicação:

   ```bash
   python app.py
   ```

6. Acesse http://127.0.0.1:5000.

O treinamento gera os arquivos `spam_model.joblib` e `tfidf_vectorizer.joblib`, usados pela API.

## Rotas

- `GET /` — página de análise.
- `GET /health` — verifica se a API está online.
- `POST /predict` — recebe um JSON com o campo `text`.

Exemplo de envio:

```json
{"text": "Congratulations, you won a prize. Click here!"}
```

## Testes

Execute `pytest` na pasta do projeto.

A classificação é uma estimativa do modelo. Confira o remetente e o conteúdo antes de confiar em um e-mail.
