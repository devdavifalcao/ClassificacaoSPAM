import os
import re

import joblib
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "emails.csv")
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")

STOPWORDS = set(ENGLISH_STOP_WORDS)
STEMMER = SnowballStemmer("english")


def normalize_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", text.lower())
    processed = [
        STEMMER.stem(word)
        for word in words
        if word.isalpha() and word not in STOPWORDS and len(word) > 2
    ]
    return " ".join(processed)


def etl_pipeline(df, text_col, label_col):
    df = df[[text_col, label_col]].dropna().copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df[df[label_col].isin([0, 1])].copy()
    df[label_col] = df[label_col].astype(int)
    df["text_norm"] = df[text_col].astype(str).apply(normalize_text)
    return df[df["text_norm"].str.strip() != ""].reset_index(drop=True)


def preprocessing_and_train(csv_path=CSV_PATH, test_size=0.2, random_state=42):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV não encontrado em {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8", engine="python", on_bad_lines="skip")
    text_col = next((c for c in df.columns if c.lower() in ("texto", "text", "message", "email")), df.columns[0])
    label_col = next((c for c in df.columns if c.lower() in ("spam", "label", "class")), df.columns[-1])

    df = etl_pipeline(df, text_col, label_col)
    if df[label_col].nunique() != 2:
        raise ValueError("A base precisa ter exemplos das duas classes: spam (1) e legítimo (0).")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_norm"], df[label_col], test_size=test_size,
        random_state=random_state, stratify=df[label_col]
    )

    # O balanceamento acontece só no treino, sem repetir exemplos no teste.
    treino = pd.DataFrame({"text": X_train, "label": y_train})
    spam = treino[treino["label"] == 1]
    legitimos = treino[treino["label"] == 0]

    if len(spam) < len(legitimos):
        spam = resample(spam, replace=True, n_samples=len(legitimos), random_state=random_state)
    elif len(legitimos) < len(spam):
        legitimos = resample(legitimos, replace=True, n_samples=len(spam), random_state=random_state)

    treino = pd.concat([spam, legitimos]).sample(frac=1, random_state=random_state)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(treino["text"])

    model = MultinomialNB()
    model.fit(X_train_tfidf, treino["label"])

    y_pred = model.predict(vectorizer.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    print(f"Acurácia no conjunto de teste: {accuracy:.4f}")
    print("Relatório de classificação:\n", report)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return {"model": model, "vectorizer": vectorizer, "accuracy": accuracy, "report": report}


def spam_detector(content):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Modelo ou vetor de palavras não encontrado. Execute train.py para treinar.")

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
