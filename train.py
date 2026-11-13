import os
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "emails.csv")
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")


def normalize_text(text):
    """
    Normaliza mensagens sem depender de regras exclusivas do inglês.
    Mantém acentos, palavras curtas e termos que podem indicar negação.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
        " emailtoken ",
        text,
    )
    text = re.sub(r"<[^>]+>", " htmltoken ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def etl_pipeline(df, text_col, label_col):
    df = df[[text_col, label_col]].dropna().copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df[df[label_col].isin([0, 1])].copy()
    df[label_col] = df[label_col].astype(int)
    df["text_norm"] = df[text_col].astype(str).apply(normalize_text)
    df = df[df["text_norm"].str.strip() != ""].copy()

    # Remove mensagens repetidas para evitar que cópias caiam no treino e no teste.
    conflitos = df.groupby("text_norm")[label_col].nunique()
    textos_ambiguos = conflitos[conflitos > 1].index
    if len(textos_ambiguos):
        df = df[~df["text_norm"].isin(textos_ambiguos)]

    return df.drop_duplicates(subset=["text_norm"]).reset_index(drop=True)


def preprocessing_and_train(csv_path=CSV_PATH, test_size=0.2, random_state=42):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV não encontrado em {csv_path}")

    try:
        df = pd.read_csv(
            csv_path,
            encoding="utf-8-sig",
            engine="python",
            on_bad_lines="skip",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            csv_path,
            encoding="latin-1",
            engine="python",
            on_bad_lines="skip",
        )

    if df.empty or len(df.columns) < 2:
        raise ValueError("O CSV precisa conter uma coluna de texto e uma coluna de classificação.")

    colunas = {str(coluna).strip().lower(): coluna for coluna in df.columns}
    text_col = next(
        (colunas[nome] for nome in ("texto", "text", "message", "email") if nome in colunas),
        df.columns[0],
    )
    label_col = next(
        (colunas[nome] for nome in ("spam", "label", "class") if nome in colunas),
        df.columns[-1],
    )

    if text_col == label_col:
        raise ValueError("Não foi possível identificar separadamente as colunas de texto e classificação.")

    df = etl_pipeline(df, text_col, label_col)
    distribuicao = df[label_col].value_counts().sort_index()

    if set(distribuicao.index) != {0, 1}:
        raise ValueError(
            "A base precisa conter as duas classes: legítimo (0) e spam (1). "
            f"Distribuição encontrada: {distribuicao.to_dict()}"
        )

    print(f"Total de mensagens válidas: {len(df)}")
    print(f"Distribuição das classes (0=legítimo, 1=spam): {distribuicao.to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_norm"],
        df[label_col],
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )

    menor_classe = int(y_train.value_counts().min())
    n_splits = min(5, menor_classe)
    if n_splits < 2:
        raise ValueError(
            "Há poucos exemplos de uma das classes para fazer validação cruzada. "
            "Adicione mais mensagens legítimas e spam à base."
        )

    cross_validation = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # Palavras e grupos de caracteres ajudam a reconhecer variações em idiomas diferentes.
    # Não há stemming nem remoção de stopwords em inglês.
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents=None,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=120000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=180000,
                    sublinear_tf=True,
                ),
            ),
        ],
        transformer_weights={"word": 1.0, "char": 0.8},
    )

    pipeline = Pipeline(
        [
            ("vectorizer", features),
            ("classifier", ComplementNB()),
        ]
    )

    # Todos os candidatos abaixo oferecem predict_proba, usado pelo app.py.
    parameters = [
        {
            "classifier": [MultinomialNB()],
            "classifier__alpha": [0.1, 0.5, 1.0, 2.0],
        },
        {
            "classifier": [ComplementNB()],
            "classifier__alpha": [0.1, 0.5, 1.0, 2.0],
        },
        {
            "classifier": [LogisticRegression(max_iter=2000, solver="liblinear")],
            "classifier__C": [0.5, 1.0, 2.0],
            "classifier__class_weight": [None, "balanced"],
        },
    ]

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=-1,
        pre_dispatch="2*n_jobs",
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    vectorizer = best_pipeline.named_steps["vectorizer"]
    model = best_pipeline.named_steps["classifier"]

    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["legítimo (0)", "spam (1)"],
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"Melhor modelo: {model.__class__.__name__}")
    print(f"Melhores parâmetros: {search.best_params_}")
    print(f"Melhor F1 macro na validação cruzada: {search.best_score_:.4f}")
    print(f"Acurácia no conjunto de teste: {accuracy:.4f}")
    print("Matriz de confusão (linhas=reais; colunas=previstas; ordem 0, 1):")
    print(matrix)
    print("Relatório de classificação:\n", report)

    # Mantém os mesmos nomes e formatos de arquivos esperados pelo restante do projeto.
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": matrix,
        "best_params": search.best_params_,
        "cross_validation_score": search.best_score_,
    }


def spam_detector(content):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Modelo ou vetor de palavras não encontrado. Execute train.py para treinar."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    features = vectorizer.transform([normalize_text(content)])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    return {
        "label": prediction,
        "label_name": "spam" if prediction == 1 else "ham",
        "probabilities": {
            int(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        },
    }


if __name__ == "__main__":
    preprocessing_and_train()
