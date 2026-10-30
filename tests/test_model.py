import joblib
import pandas as pd

from train import etl_pipeline, normalize_text, preprocessing_and_train, spam_detector


def test_normalize_text_basic():
    result = normalize_text("Hello!!! This is a TEST message 123.")
    assert isinstance(result, str)
    assert "hello" in result
    assert "test" in result
    assert "this" not in result


def test_etl_pipeline_does_not_balance_before_split():
    df = pd.DataFrame({
        "text": ["spam one", "spam two", "spam three", "normal message"],
        "label": [1, 1, 1, 0],
    })
    processed = etl_pipeline(df, "text", "label")
    assert len(processed) == 4
    assert processed["label"].value_counts().to_dict() == {1: 3, 0: 1}
    assert "text_norm" in processed.columns


def test_preprocessing_and_train(tmp_path, monkeypatch):
    csv_file = tmp_path / "emails.csv"
    messages = [
        ("claim your prize now", 1), ("meeting tomorrow morning", 0),
        ("limited offer buy now", 1), ("project meeting notes", 0),
        ("winner claim free gift", 1), ("please review the report", 0),
        ("free money click here", 1), ("lunch at noon", 0),
        ("urgent prize waiting", 1), ("schedule the next meeting", 0),
        ("buy cheap products today", 1), ("thanks for your help", 0),
        ("exclusive offer today", 1), ("the document is attached", 0),
        ("click to claim reward", 1), ("see you at the office", 0),
        ("you won a free prize", 1), ("can we talk later", 0),
        ("special deal buy today", 1), ("here are the meeting notes", 0),
    ]
    pd.DataFrame(messages, columns=["text", "label"]).to_csv(csv_file, index=False)
    model_path = tmp_path / "spam_model.joblib"
    vectorizer_path = tmp_path / "vec.joblib"
    monkeypatch.setattr("train.MODEL_PATH", str(model_path))
    monkeypatch.setattr("train.VECTORIZER_PATH", str(vectorizer_path))

    result = preprocessing_and_train(csv_path=str(csv_file))
    assert model_path.exists()
    assert vectorizer_path.exists()
    assert 0 <= result["accuracy"] <= 1


def test_spam_detector_prediction(tmp_path, monkeypatch):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(["spam message", "hello friend"])
    model = MultinomialNB().fit(features, [1, 0])
    model_path = tmp_path / "model.joblib"
    vectorizer_path = tmp_path / "vec.joblib"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    monkeypatch.setattr("train.MODEL_PATH", str(model_path))
    monkeypatch.setattr("train.VECTORIZER_PATH", str(vectorizer_path))

    result = spam_detector("Hello friend")
    assert result["label"] in (0, 1)
    assert result["label_name"] in ("spam", "ham")
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6
