"""
explain.py  -  UPGRADE: Explainability

Two different explanation techniques, because the two models work
completely differently under the hood:

- Classic model (TF-IDF + Logistic Regression): each feature IS a word or
  short phrase, and each category has a learned weight for it, so we can
  directly show which words in this ticket pushed the prediction toward
  its category. This is the standard way to explain a linear text model.

- Smart model (sentence embeddings + Logistic Regression): the 384 numbers
  in an embedding don't correspond to words, so word-level highlighting
  isn't meaningful here. Instead we show the training tickets whose
  embeddings are most similar (nearest neighbors) to this one - "the model
  predicted this because it looks like these past tickets."
"""

import html
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def explain_tfidf(raw_text, cleaned_text, model, vectorizer, predicted_idx, top_n=6):
    """Returns (top_terms, highlighted_html) for the classic model.
    top_terms: list of {"term": str, "weight": float} sorted by contribution.
    highlighted_html: raw_text with the top contributing terms wrapped in
    <mark> tags (case-insensitive match against the original text)."""
    X = vectorizer.transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[predicted_idx]

    nonzero_idx = X.nonzero()[1]
    contributions = [
        (feature_names[i], float(X[0, i] * coefs[i]))
        for i in nonzero_idx
    ]
    contributions.sort(key=lambda pair: pair[1], reverse=True)
    top_terms = [
        {"term": term, "weight": weight}
        for term, weight in contributions[:top_n] if weight > 0
    ]

    # Escape first so any stray <, >, & in the user's ticket text can't be
    # interpreted as HTML once we insert real <mark> tags around it.
    safe_text = html.escape(raw_text)
    highlighted_html = _highlight_spans(safe_text, [t["term"] for t in top_terms])

    return top_terms, highlighted_html


def _highlight_spans(raw_text, terms):
    """Wrap each term found in raw_text with <mark> tags, without letting a
    shorter term's match overlap (and re-wrap) a longer term's match -
    e.g. "keeps" and "keeps failing" both matching would otherwise nest."""
    spans = []
    for term in sorted(terms, key=len, reverse=True):
        for m in re.finditer(re.escape(term), raw_text, re.IGNORECASE):
            start, end = m.span()
            if any(start < e and s < end for s, e in spans):
                continue  # overlaps an already-accepted (longer) match
            spans.append((start, end))

    spans.sort()
    pieces = []
    cursor = 0
    for start, end in spans:
        pieces.append(raw_text[cursor:start])
        pieces.append(f"<mark>{raw_text[start:end]}</mark>")
        cursor = end
    pieces.append(raw_text[cursor:])
    return "".join(pieces)


def explain_embeddings(query_embedding, index, top_k=3):
    """index is the dict saved by train_embeddings.py
    ({"embeddings", "texts", "categories"}). Returns a list of
    {"text", "category", "similarity"} for the most similar past tickets."""
    sims = cosine_similarity([query_embedding], index["embeddings"])[0]
    order = np.argsort(sims)[::-1][:top_k]
    return [
        {
            "text": index["texts"][i],
            "category": index["categories"][i],
            "similarity": float(sims[i]) * 100,
        }
        for i in order
    ]
