"""
train.py  -  Task 3 & Task 4: Model Training and Testing

Pipeline:  ticket_description -> text cleaning -> TF-IDF -> Logistic Regression
Algorithm: Logistic Regression was chosen because it trains fast, works well
on small/medium text datasets, and gives predict_proba() confidence scores
that Task 5 needs. TF-IDF turns each cleaned description into a vector of
word-importance weights (with unigrams + bigrams) which Logistic Regression
then uses to separate the categories.

Saves:
    model/model.joblib        - trained classifier
    model/vectorizer.joblib   - fitted TF-IDF vectorizer
    model/label_encoder.joblib
    screenshots/confusion_matrix.png

Run:
    python src/train.py
"""

import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from preprocessing import load_data, prepare_dataset


def train_and_evaluate():
    df = load_data()
    df, _ = prepare_dataset(df)

    X_text = df["clean_description"]
    y_labels = df["category"]

    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=3000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

    print("=" * 60)
    print(" TASK 4: MODEL TRAINING AND TESTING")
    print("=" * 60)
    print(f"\nTraining samples: {X_train.shape[0]}   Testing samples: {X_test.shape[0]}")
    print(f"\nAccuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print("\nClassification report:\n")
    print(report)

    print(f"This means the model correctly classified approximately "
          f"{accuracy:.0%} of the test tickets.")
    if accuracy >= 0.75:
        print("The model performs satisfactorily for a first-pass "
              "classifier on a small synthetic dataset.")
    else:
        print("The model's performance is modest; this is expected given the "
              "small dataset size and could be improved with more real data.")

    # Confusion matrix chart
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()
    plt.savefig("screenshots/confusion_matrix.png", dpi=150)
    plt.close()
    print("\nSaved confusion matrix chart to screenshots/confusion_matrix.png")

    # Persist model artifacts
    joblib.dump(model, "model/model.joblib")
    joblib.dump(vectorizer, "model/vectorizer.joblib")
    joblib.dump(le, "model/label_encoder.joblib")

    metrics = {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1_score": f1,
        "train_size": X_train.shape[0], "test_size": X_test.shape[0],
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved trained model + vectorizer + label encoder to model/")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
