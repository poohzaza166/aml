"""
Task 1 — Evaluate the chosen model on the NLTK `movie_reviews` corpus
(Pang & Lee 2004 polarity dataset v2).

This corpus is OUR-OF-DISTRIBUTION relative to the training data:
  - 2000 full-length movie reviews (≈600–700 tokens each), not the
    one-sentence Pang-Lee 2005 snippets we trained on.
  - Older (pre-2004) so the lexicon shifts (different actors, less
    internet slang).
  - No spam.

So this is a *domain-shift* test of:
  (a) whether the spam stage spuriously flags any of these,
  (b) whether the sentiment stage generalises from short snippets to
      long, multi-paragraph reviews.

We compare two of our trained models:
  - D (word+char TF-IDF + LR) — our best on validation
  - E1 (mean GloVe-100 + LR) — the dense-embedding baseline
                                 (hypothesis: should close some of the
                                 gap on longer text via averaging).
"""


import json
import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import movie_reviews
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "outputs"

SPAM_RE = re.compile(r"^\s*subject\s*:", flags=re.IGNORECASE)


def is_spam(t: str) -> bool:
    return isinstance(t, str) and bool(SPAM_RE.search(t))


def load_nltk_corpus():
    """Returns (texts, labels) where labels are 0=neg, 1=pos."""
    try:
        movie_reviews.fileids()
    except LookupError:
        nltk.download("movie_reviews", quiet=True)

    texts, labels = [], []
    for cat, label in [("neg", 0), ("pos", 1)]:
        for fid in movie_reviews.fileids(cat):
            raw = movie_reviews.raw(fid).replace("\n", " ")
            texts.append(raw)
            labels.append(label)
    return texts, np.array(labels, dtype=int)


def main() -> None:
    texts, y = load_nltk_corpus()
    print(f"NLTK movie_reviews: {len(texts)} docs   "
          f"mean tokens = {np.mean([len(t.split()) for t in texts]):.0f}")

    # (a) Stage-1 sanity: confirm no spam flags
    spam = joblib.load(OUT / "spam_model.joblib")
    spam_pred = spam.predict(texts)
    n_spam = int(spam_pred.sum())
    n_rule = int(sum(is_spam(t) for t in texts))
    print(f"stage-1 flagged as spam: {n_spam}/{len(texts)}  "
          f"(rule: {n_rule}/{len(texts)})")

    # (b) Stage-2: compare model D vs E1
    print("\n=== Model D (word+char TF-IDF + LR) ===")
    D = joblib.load(OUT / "sent_D_LR_word_char.joblib")
    yD = D.predict(texts)
    print(f"acc = {accuracy_score(y, yD):.4f}  macro F1 = {f1_score(y, yD, average='macro'):.4f}")
    print(confusion_matrix(y, yD, labels=[0, 1]))
    print(classification_report(y, yD, target_names=["neg", "pos"], digits=4))

    print("\n=== Model E1 (mean GloVe + LR) ===")
    import gensim.downloader as api
    kv = api.load("glove-wiki-gigaword-100")
    # use module 04's mean_embed
    import importlib.util, importlib.machinery
    spec = importlib.util.spec_from_file_location(
        "m04", Path(__file__).resolve().parent / "04_glove_sentiment.py"
    )
    m04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m04)
    E1 = joblib.load(OUT / "sent_E1_mean_glove.joblib")
    X_emb = m04.mean_embed(texts, kv, idf=None)
    yE = E1.predict(X_emb)
    print(f"acc = {accuracy_score(y, yE):.4f}  macro F1 = {f1_score(y, yE, average='macro'):.4f}")
    print(confusion_matrix(y, yE, labels=[0, 1]))
    print(classification_report(y, yE, target_names=["neg", "pos"], digits=4))

    summary = {
        "n_docs": len(texts),
        "stage1_spam_flags": n_spam,
        "D_word_char_LR": {
            "acc": float(accuracy_score(y, yD)),
            "macro_f1": float(f1_score(y, yD, average="macro")),
            "cm": confusion_matrix(y, yD, labels=[0, 1]).tolist(),
        },
        "E1_mean_glove_LR": {
            "acc": float(accuracy_score(y, yE)),
            "macro_f1": float(f1_score(y, yE, average="macro")),
            "cm": confusion_matrix(y, yE, labels=[0, 1]).tolist(),
        },
    }
    (OUT / "nltk_external_metrics.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved:", OUT / "nltk_external_metrics.json")


if __name__ == "__main__":
    main()
