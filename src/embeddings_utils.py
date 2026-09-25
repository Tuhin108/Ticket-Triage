"""
embeddings_utils.py  -  UPGRADE: shared helper for the "Smart" (semantic
embeddings) model.

Uses a small pretrained sentence-transformer model to turn ticket text into
a dense vector that captures meaning, not just word overlap. Unlike TF-IDF,
this understands that "I can't sign in" and "unable to log into my account"
are about the same thing, even though they share almost no words.

The model is downloaded once (about 90 MB) from Hugging Face the first time
this runs, then cached locally (~/.cache/torch/sentence_transformers) for
every run after that - no internet needed once it's cached, and no API key
ever required.

This import is intentionally kept out of preprocessing.py/predict.py's
top-level imports so the required assignment tasks (1-6) never need
sentence-transformers or torch installed - only this upgrade does.
"""

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None  # lazy singleton, loaded on first use


def get_embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed(texts):
    """texts: a string or a list of strings. Returns a numpy array of shape
    (n_texts, 384) for all-MiniLM-L6-v2."""
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    model = get_embedder()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors[0] if single else vectors
