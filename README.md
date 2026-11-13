# Classificador de SPAM

Aplicação desenvolvida em Python para analisar mensagens e estimar se são **spam** ou **legítimas**. O projeto utiliza processamento de linguagem natural, aprendizado de máquina e uma interface web feita com Flask.

**Acesse a aplicação:** https://classificacao-de-spam.vercel.app/

## Tecnologias

- **Python** — linguagem principal
- **Flask** — aplicação web e API
- **Pandas** — leitura e preparação dos dados
- **Scikit-learn** — vetorização TF-IDF, treinamento e avaliação dos classificadores
- **Joblib** — gravação e carregamento dos arquivos do modelo
- **Pytest** — testes automatizados

## Como o classificador funciona

1. O texto passa por uma normalização que preserva acentos, palavras curtas e termos importantes, inclusive negações.
2. O TF-IDF transforma palavras e sequências de caracteres em valores numéricos.
3. O treinamento compara modelos Multinomial Naive Bayes, Complement Naive Bayes e Regressão Logística.
4. A seleção do modelo utiliza validação cruzada e a métrica F1 macro.
5. A API retorna a classe prevista e as probabilidades estimadas pelo modelo.

Os rótulos utilizados são **0 para mensagem legítima** e **1 para spam**. A probabilidade é uma estimativa do classificador, não uma garantia de segurança.

## Requisitos

- Python 3.12
- Git
- Arquivo `emails.csv` na pasta principal para realizar um novo treinamento

A base CSV precisa conter uma coluna de texto e uma coluna de classificação. O treinamento reconhece nomes como `texto`, `text`, `message` ou `email` para o conteúdo, e `spam`, `label` ou `class` para a classificação.

## Instalação e execução

1. Clone o repositório:

   ```bash
   git clone https://github.com/devdavifalcao/ClassificacaoSPAM.git
   cd ClassificacaoSPAM
   ```

2. Crie e ative um ambiente virtual:

   **Windows:**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **Linux ou macOS:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Para gerar novamente os arquivos do modelo, execute:

   ```bash
   python train.py
   ```

   O treinamento gera `spam_model.joblib` e `tfidf_vectorizer.joblib` na pasta principal.

5. Inicie a aplicação:

   ```bash
   python app.py
   ```

6. Acesse http://127.0.0.1:5000.

## API

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Exibe a página de análise |
| `GET` | `/health` | Verifica se a aplicação está respondendo |
| `POST` | `/predict` | Classifica o texto enviado |

Exemplo de requisição para `POST /predict`:

```json
{
  "text": "Congratulations, you won a prize. Click here!"
}
```

Envie o cabeçalho `Content-Type: application/json`. O campo `text` deve ser uma mensagem não vazia, com até 10.000 caracteres.

## Testes automatizados

Execute os testes na pasta principal:

```bash
pytest -q
```

Os testes verificam a página inicial, a rota do CSS sem alterar seu conteúdo, a rota de saúde, os dados enviados à API, os limites de entrada e os cenários de erro.

O projeto também executa os testes automaticamente pelo GitHub Actions a cada envio de alterações para a branch principal.

## Arquivos do modelo

Os arquivos `spam_model.joblib` e `tfidf_vectorizer.joblib` precisam estar disponíveis junto à aplicação publicada. A configuração da Vercel inclui esses dois arquivos na função responsável pela API.

Para treinar novamente, utilize a base de dados original e confira a distribuição das classes antes de considerar o resultado pronto para publicação.

## Limitações

Nenhum classificador de spam garante acerto em todas as mensagens. O resultado depende da variedade, do equilíbrio e da qualidade dos exemplos usados no treinamento. Mensagens suspeitas devem ser verificadas com atenção, principalmente quando contêm links, anexos ou pedidos de dados pessoais.

---

Desenvolvido por [Davi Falcão](https://github.com/devdavifalcao).
