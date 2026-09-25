"""
train_embeddings.py  -  UPGRADE: Smarter model using sentence embeddings

Same evaluation methodology as train.py (Task 3/4), but the features come
from a pretrained sentence-transformer model instead of TF-IDF. This is a
transfer-learning approach: the embedding model already learned what
language "means" from a huge amount of text, so it captures paraphrases
and synonyms that TF-IDF can't, without needing more labeled tickets.

Pipeline:  ticket_description -> sentence embedding (384-dim) -> Logistic Regression

Also saves every training ticket's embedding + text + category, so the app
can show "most similar past tickets" as an explanation for a prediction
(nearest-neighbor explainability - the standard way to explain an
embedding-based model, since individual embedding dimensions aren't
human-readable the way TF-IDF word weights are).

Requires the embedding dependencies:
    pip install -r requirements.txt

Run:
    python src/train_embeddings.py
"""

import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from preprocessing import load_data, prepare_dataset
from embeddings_utils import embed, EMBEDDING_MODEL_NAME


def train_and_evaluate():
    df = load_data()
    df, _ = prepare_dataset(df)

    texts = df["light_clean_description"].tolist()
    y_labels = df["category"]

    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    print("Encoding all tickets with the sentence-transformer model "
          f"({EMBEDDING_MODEL_NAME})... this can take a little while the "
          "first time while the model downloads.")
    X = embed(texts)  # shape: (n_tickets, 384)

    X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
        X, y, texts, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

    print("=" * 60)
    print(" SMART MODEL (sentence embeddings) - TRAINING AND TESTING")
    print("=" * 60)
    print(f"\nTraining samples: {X_train.shape[0]}   Testing samples: {X_test.shape[0]}")
    print(f"\nAccuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print("\nClassification report:\n")
    print(report)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix - Smart Model (Sentence Embeddings)")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()
    plt.savefig("screenshots/confusion_matrix_embeddings.png", dpi=150)
    plt.close()
    print("\nSaved confusion matrix chart to screenshots/confusion_matrix_embeddings.png")

    # Save model + label encoder (kept separate from the classic model's
    # files so the two pipelines never accidentally share state)
    joblib.dump(model, "model/model_embeddings.joblib")
    joblib.dump(le, "model/label_encoder_embeddings.joblib")

    # Save only the training set's embeddings + text + category for
    # nearest-neighbor explainability at prediction time.
    joblib.dump({
        "embeddings": X_train,
        "texts": text_train,
        "categories": le.inverse_transform(y_train).tolist(),
    }, "model/embedding_index.joblib")

    metrics = {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1_score": f1,
        "train_size": X_train.shape[0], "test_size": X_test.shape[0],
        "embedding_model": EMBEDDING_MODEL_NAME,
    }
    with open("model/metrics_embeddings.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved trained smart model + nearest-neighbor index to model/")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
