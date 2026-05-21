"""
Task 1 — Stage-2 with pre-trained word-embedding representations.

Two embedding-based representations are compared:
  E1. Mean GloVe-100 (Pennington et al. 2014). Each review = mean of in-vocab
      token vectors. Captures distributional semantics but discards word
      order and treats all tokens equally.
  E2. TF-IDF weighted mean GloVe-100. Same as E1 but each token's vector
      is weighted by its IDF (lower weight to common words like "the",
      "and"). This is the SIF-lite recipe of Arora et al. (2017).

Both are L2-normalised and fed to a logistic regression. The point is
to contrast a *bag-of-words* sparse representation with a *dense
distributional* representation under the same downstream classifier.

References:
  - Pennington, Socher & Manning (2014), "GloVe: Global Vectors for Word
    Representation", EMNLP.
  - Arora, Liang & Ma (2017), "A Simple but Tough-to-Beat Baseline for
    Sentence Embeddings", ICLR.
"""


import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "outputs"
OUT.mkdir(exist_ok=True)

SPAM_RE = re.compile(r"^\s*subject\s*:", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")


def is_spam(text: str) -> bool:
    return isinstance(text, str) and bool(SPAM_RE.search(text))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_data():
    tr = pd.read_csv(ROOT / "sentiment_analysis_training_data.csv")
    va = pd.read_csv(ROOT / "sentiment_analysis_validation_data.csv")
    tr_clean = tr[~tr["text"].apply(is_spam)].reset_index(drop=True)
    va_clean = va[~va["text"].apply(is_spam)].reset_index(drop=True)
    return tr_clean, va_clean


def mean_embed(texts, kv, idf=None) -> np.ndarray:
    dim = kv.vector_size
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = tokenize(t)
        vecs = []
        weights = []
        for tok in toks:
            if tok in kv.key_to_index:
                vecs.append(kv[tok])
                if idf is not None:
                    weights.append(idf.get(tok, 1.0))
                else:
                    weights.append(1.0)
        if vecs:
            V = np.stack(vecs)
            w = np.asarray(weights, dtype=np.float32)
            X[i] = (V * w[:, None]).sum(0) / max(w.sum(), 1e-8)
    X = normalize(X, norm="l2", axis=1, copy=False)
    return X


def build_idf(texts) -> dict[str, float]:
    """IDF for embedding weighting; vocab restricted to tokenized form."""
    vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, min_df=2)
    vec.fit(texts)
    idf = dict(zip(vec.get_feature_names_out(), vec.idf_))
    return idf


def bootstrap_acc_ci(y_true, y_pred, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    n_s = len(y_true)
    accs = [accuracy_score(y_true[rng.integers(0, n_s, n_s)],
                            y_pred[rng.integers(0, n_s, n_s)]) for _ in range(n)]
    # Note: paired bootstrap on accuracy of the *same* predictions
    accs = []
    for _ in range(n):
        idx = rng.integers(0, n_s, n_s)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
    accs = np.asarray(accs)
    return float(np.quantile(accs, 0.025)), float(np.quantile(accs, 0.975))


def main() -> None:
    print("loading GloVe-100...")
    import gensim.downloader as api
    kv = api.load("glove-wiki-gigaword-100")
    print(f"GloVe vocab: {len(kv.key_to_index)}  dim: {kv.vector_size}")

    tr, va = load_data()
    print(f"train: {len(tr)}  val: {len(va)}")

    # Coverage diagnostic
    sample_toks = [tokenize(t) for t in tr["text"].iloc[:500]]
    flat = [tok for toks in sample_toks for tok in toks]
    in_vocab = sum(1 for t in flat if t in kv.key_to_index)
    print(f"GloVe token coverage on first 500 reviews: {in_vocab}/{len(flat)}  "
          f"({100*in_vocab/max(len(flat),1):.1f}%)")

    idf = build_idf(tr["text"].tolist())
    print(f"IDF vocab size: {len(idf)}")

    results = {}
    for name, idf_use in [("E1_mean_glove", None), ("E2_tfidf_glove", idf)]:
        Xtr = mean_embed(tr["text"].tolist(), kv, idf_use)
        Xva = mean_embed(va["text"].tolist(), kv, idf_use)
        ytr = tr["label"].astype(int).values
        yva = va["label"].astype(int).values

        clf = LogisticRegression(C=4.0, max_iter=2000)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        cv_scores = cross_val_score(clf, Xtr, ytr, cv=cv, scoring="accuracy", n_jobs=-1)
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xva)
        val_acc = accuracy_score(yva, y_pred)
        val_f1 = f1_score(yva, y_pred, average="macro")
        cm = confusion_matrix(yva, y_pred, labels=[0, 1])
        ci_lo, ci_hi = bootstrap_acc_ci(yva, y_pred)
        rep = classification_report(yva, y_pred, target_names=["neg", "pos"], digits=4)
        print(
            f"\n[{name}] cv-acc={cv_scores.mean():.4f}±{cv_scores.std():.4f}  "
            f"val-acc={val_acc:.4f}  val-F1={val_f1:.4f}  "
            f"95%CI=({ci_lo:.4f},{ci_hi:.4f})"
        )
        print(rep)

        results[name] = dict(
            cv_acc_mean=float(cv_scores.mean()),
            cv_acc_std=float(cv_scores.std()),
            val_acc=float(val_acc),
            val_acc_ci=[ci_lo, ci_hi],
            val_macro_f1=float(val_f1),
            val_cm=cm.tolist(),
        )
        joblib.dump(clf, OUT / f"sent_{name}.joblib")

    (OUT / "sentiment_glove_metrics.json").write_text(json.dumps(results, indent=2))
    print("Saved:", OUT / "sentiment_glove_metrics.json")


if __name__ == "__main__":
    main()
