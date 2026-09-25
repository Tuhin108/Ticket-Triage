"""
predict.py  -  Task 5: New Ticket Prediction
              + Optional Bonus: Generative auto-response

Loads the trained model/vectorizer and predicts the category (with a
confidence score) for a new ticket description. Also returns a rule-based
suggested response for the customer, based on the predicted category.

Usage:
    python src/predict.py                      # interactive prompt
    python src/predict.py --test               # run 5 built-in test cases
    python src/predict.py "some ticket text"    # predict a single ticket
"""

import sys
import joblib

from preprocessing import clean_text

MODEL_PATH = "model/model.joblib"
VECTORIZER_PATH = "model/vectorizer.joblib"
ENCODER_PATH = "model/label_encoder.joblib"

# Bonus Task: rule-based auto-response generator, keyed by predicted category
SUGGESTED_RESPONSES = {
    "Login Issue": (
        "Please use the 'Forgot Password' option on the login page to reset "
        "your password. If the problem continues, please contact the support team."
    ),
    "Application Error": (
        "We're sorry for the inconvenience. Please try refreshing the app or "
        "logging out and back in. If the error persists, our engineering team "
        "will investigate further."
    ),
    "Report": (
        "Thank you for reporting this. Please try regenerating the report; if "
        "it still fails, share the exact filters used so we can reproduce the issue."
    ),
    "Account Update": (
        "Your account update request has been received. Changes are usually "
        "reflected within 24 hours. Contact us if you don't see them by then."
    ),
    "Performance": (
        "We're aware some users are experiencing slowness and are working to "
        "improve response times. Try clearing your cache in the meantime."
    ),
}


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    le = joblib.load(ENCODER_PATH)
    return model, vectorizer, le


def predict_ticket(description, model, vectorizer, le):
    cleaned = clean_text(description)
    X = vectorizer.transform([cleaned])
    pred_idx = model.predict(X)[0]
    category = le.inverse_transform([pred_idx])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(proba[pred_idx]) * 100

    response = SUGGESTED_RESPONSES.get(category, "Thank you, our team will look into this shortly.")
    return category, confidence, response


def predict_with_breakdown(description, model, vectorizer, le):
    """Like predict_ticket, but also returns the probability for every
    category (sorted highest first) so a UI can show why the model chose
    what it chose, not just the single winning label."""
    cleaned = clean_text(description)
    X = vectorizer.transform([cleaned])
    pred_idx = model.predict(X)[0]
    category = le.inverse_transform([pred_idx])[0]

    breakdown = []
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(proba[pred_idx]) * 100
        breakdown = sorted(
            [{"category": cat, "percent": float(p) * 100}
             for cat, p in zip(le.classes_, proba)],
            key=lambda row: row["percent"],
            reverse=True,
        )

    response = SUGGESTED_RESPONSES.get(category, "Thank you, our team will look into this shortly.")
    return {
        "category": category,
        "confidence": confidence,
        "breakdown": breakdown,
        "response": response,
    }


EMBEDDING_MODEL_PATH = "model/model_embeddings.joblib"
EMBEDDING_ENCODER_PATH = "model/label_encoder_embeddings.joblib"
EMBEDDING_INDEX_PATH = "model/embedding_index.joblib"

_embedding_artifacts = None  # lazy cache, so classic-only usage never touches this


def load_embedding_artifacts():
    """Loads the smart (sentence-embeddings) model's files. Raises a clear
    error if train_embeddings.py hasn't been run yet."""
    global _embedding_artifacts
    if _embedding_artifacts is None:
        import os
        if not os.path.exists(EMBEDDING_MODEL_PATH):
            raise FileNotFoundError(
                "The smart model hasn't been trained yet. Run:\n"
                "  pip install -r requirements.txt\n"
                "  python src/train_embeddings.py"
            )
        smart_model = joblib.load(EMBEDDING_MODEL_PATH)
        smart_le = joblib.load(EMBEDDING_ENCODER_PATH)
        index = joblib.load(EMBEDDING_INDEX_PATH)
        _embedding_artifacts = (smart_model, smart_le, index)
    return _embedding_artifacts


def classify(description, model_choice="classic", classic_artifacts=None):
    """Unified entry point used by the web app: runs either the classic
    (TF-IDF) or smart (sentence-embeddings) model on one ticket description,
    and returns category + confidence breakdown + a suggested response +
    a model-appropriate explanation."""
    from explain import explain_tfidf, explain_embeddings
    from preprocessing import clean_text, light_clean

    if model_choice == "smart":
        smart_model, smart_le, index = load_embedding_artifacts()
        from embeddings_utils import embed
        cleaned = light_clean(description)
        vec = embed(cleaned)
        pred_idx = smart_model.predict([vec])[0]
        category = smart_le.inverse_transform([pred_idx])[0]
        proba = smart_model.predict_proba([vec])[0]
        confidence = float(proba[pred_idx]) * 100
        breakdown = sorted(
            [{"category": cat, "percent": float(p) * 100}
             for cat, p in zip(smart_le.classes_, proba)],
            key=lambda row: row["percent"], reverse=True,
        )
        similar = explain_embeddings(vec, index, top_k=3)
        explanation = {"type": "similar_tickets", "items": similar}
    else:
        model, vectorizer, le = classic_artifacts or load_artifacts()
        cleaned = clean_text(description)
        X = vectorizer.transform([cleaned])
        pred_idx = model.predict(X)[0]
        category = le.inverse_transform([pred_idx])[0]
        proba = model.predict_proba(X)[0]
        confidence = float(proba[pred_idx]) * 100
        breakdown = sorted(
            [{"category": cat, "percent": float(p) * 100}
             for cat, p in zip(le.classes_, proba)],
            key=lambda row: row["percent"], reverse=True,
        )
        top_terms, highlighted_html = explain_tfidf(
            description, cleaned, model, vectorizer, pred_idx, top_n=6
        )
        explanation = {
            "type": "keywords",
            "top_terms": top_terms,
            "highlighted_html": highlighted_html,
        }

    response = SUGGESTED_RESPONSES.get(category, "Thank you, our team will look into this shortly.")
    return {
        "category": category,
        "confidence": confidence,
        "breakdown": breakdown,
        "response": response,
        "explanation": explanation,
    }


def classify_batch(descriptions, model_choice="classic"):
    """Vectorized version of classify() for many tickets at once (bulk CSV
    upload). Returns a list of {category, confidence} dicts, same order as
    the input. No per-ticket explanation (kept fast for large batches)."""
    from preprocessing import clean_text, light_clean

    if model_choice == "smart":
        smart_model, smart_le, _ = load_embedding_artifacts()
        from embeddings_utils import embed
        cleaned = [light_clean(d) for d in descriptions]
        vectors = embed(cleaned)
        preds = smart_model.predict(vectors)
        probas = smart_model.predict_proba(vectors)
        categories = smart_le.inverse_transform(preds)
    else:
        model, vectorizer, le = load_artifacts()
        cleaned = [clean_text(d) for d in descriptions]
        X = vectorizer.transform(cleaned)
        preds = model.predict(X)
        probas = model.predict_proba(X)
        categories = le.inverse_transform(preds)

    results = []
    for i, category in enumerate(categories):
        confidence = float(probas[i][preds[i]]) * 100
        results.append({"category": category, "confidence": confidence})
    return results


# 5 new ticket descriptions NOT used during training, for Task 5's required test
TEST_TICKETS = [
    "I can't sign into my account, it keeps saying wrong password even though I reset it.",
    "The app crashed twice today while I was uploading a document.",
    "My invoice report has the wrong total, can you check it?",
    "I would like to change the email address associated with my account.",
    "Everything on the website loads so slowly today, is there an outage?",
]


def run_test_cases():
    model, vectorizer, le = load_artifacts()
    print("=" * 60)
    print(" TASK 5: NEW TICKET PREDICTION - TEST CASES")
    print("=" * 60)
    for i, ticket in enumerate(TEST_TICKETS, 1):
        category, confidence, response = predict_ticket(ticket, model, vectorizer, le)
        print(f"\nTest Case {i}")
        print(f"Ticket Description: {ticket}")
        print(f"Predicted Category: {category}")
        if confidence is not None:
            print(f"Confidence: {confidence:.1f}%")
        print(f"Suggested Response: {response}")


def run_interactive():
    model, vectorizer, le = load_artifacts()
    print("====================================")
    print(" Customer Support Classifier")
    print("====================================")
    description = input("\nEnter Ticket Description:\n\n> ")
    category, confidence, response = predict_ticket(description, model, vectorizer, le)
    print(f"\nPredicted Category:\n{category}")
    if confidence is not None:
        print(f"Confidence: {confidence:.1f}%")
    print(f"\nSuggested Response:\n{response}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test_cases()
    elif len(sys.argv) > 1:
        model, vectorizer, le = load_artifacts()
        text = " ".join(sys.argv[1:])
        category, confidence, response = predict_ticket(text, model, vectorizer, le)
        print(f"Predicted Category: {category}")
        if confidence is not None:
            print(f"Confidence: {confidence:.1f}%")
        print(f"Suggested Response: {response}")
    else:
        run_interactive()
