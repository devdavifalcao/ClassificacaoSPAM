# Classificador de SPAM

Aplicação desenvolvida em Python para identificar se uma mensagem é **spam** ou **legítima**. O projeto utiliza processamento de texto e aprendizado de máquina, com uma interface web feita em Flask para testar mensagens.

**Acesse a aplicação:** https://classificacao-de-spam.vercel.app/

## Tecnologias

- **Python** — linguagem principal
- **Flask** — aplicação web e API
- **Pandas** — leitura e tratamento dos dados
- **Scikit-learn** — TF-IDF e algoritmo Naive Bayes
- **NLTK** — normalização e redução de palavras (Snowball Stemmer)
- **Joblib** — salvamento e carregamento do modelo

## Como funciona

O texto passa por uma etapa de pré-processamento. Em seguida, o TF-IDF transforma as palavras em representações numéricas que são utilizadas pelo algoritmo **Multinomial Naive Bayes** para realizar a classificação.

O resultado indica se a mensagem foi identificada como spam ou legítima. Essa classificação é uma estimativa do modelo e não substitui a verificação do remetente e do conteúdo.

## Requisitos

- Python 3.10 ou superior
- Git
- Arquivo de dados `emails.csv` na pasta principal do projeto

A base CSV deve conter uma coluna com o texto das mensagens e outra com os rótulos. O treinamento reconhece nomes como `text`, `texto`, `message` ou `email` para o conteúdo, e `spam`, `label` ou `class` para a classificação. Os rótulos devem ser **1 para spam** e **0 para mensagem legítima**.

## Instalação e execução

1. Clone o repositório:

   ```bash
   git clone https://github.com/devdavifalcao/ClassificacaoSPAM.git
   cd ClassificacaoSPAM
   ```

2. Crie um ambiente virtual:

   ```bash
   python -m venv .venv
   ```

3. Ative o ambiente virtual.

   **Windows:**

   ```bash
   .venv\Scripts\activate
   ```

   **Linux ou macOS:**

   ```bash
   source .venv/bin/activate
   ```

4. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

5. Coloque o arquivo `emails.csv` na pasta principal e treine o modelo:

   ```bash
   python train.py
   ```

   Ao final do treinamento, serão gerados os arquivos `spam_model.joblib` e `tfidf_vectorizer.joblib`.

6. Inicie a aplicação:

   ```bash
   python app.py
   ```

7. Abra no navegador:

   [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Utilização da aplicação

Na página inicial, digite ou cole o conteúdo de uma mensagem e clique em **Analisar mensagem**. O sistema exibirá a classificação e, quando disponível, a probabilidade indicada pelo modelo.

## API

A aplicação disponibiliza as seguintes rotas:

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Exibe a página de análise |
| `GET` | `/health` | Verifica se a API está online |
| `POST` | `/predict` | Classifica o texto enviado |

Exemplo de requisição para `POST /predict`:

```json
{
  "text": "Congratulations, you won a prize. Click here!"
}
```

A requisição deve ser enviada com o cabeçalho `Content-Type: application/json`.

## Testes

Com as dependências instaladas, execute na pasta do projeto:

```bash
pytest
```

## Estrutura do projeto

```text
ClassificacaoSPAM/
├── app.py
├── train.py
├── utils.py
├── requirements.txt
├── emails.csv
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── tests/
```

## Observação

O desempenho do classificador depende da qualidade e da variedade dos dados utilizados no treinamento. Por isso, os resultados devem ser tratados como apoio à análise, e não como garantia de que uma mensagem é segura.

---

Desenvolvido por [Davi Falcão](https://github.com/devdavifalcao).
