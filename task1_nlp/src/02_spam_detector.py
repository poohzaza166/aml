"""
Task 1 — Stage-1 classifier: spam vs movie-review.

Design:
  - Weak label: a sample is spam iff its text starts with /^\s*subject\s*:/i
    (this surface marker is present in 100% of suspected spam and 0% of
    reviews — verified in 01_explore).
  - Train logistic regression on char n-gram TF-IDF features so the model
    learns the *style* (CRLF, digit clusters, business-email vocabulary)
    rather than just the literal prefix word.
  - Validate by 5-fold CV on training, then test on validation set
    (which also has weak labels of the same form).
  - The output predict_proba is later used as the spam mask for the
    sentiment pipeline; the report contrasts this learned spam mask
    against the pure rule for robustness analysis.

References:
  - Sebastiani (2002), "Machine Learning in Automated Text Categorization"
  - Manning, Raghavan & Schütze (2008), "Introduction to Information
    Retrieval", ch. 13 (text classification, n-grams, Naive Bayes).
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
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parents[1]
OUT_DIR = TASK / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SPAM_RE = re.compile(r"^\s*subject\s*:", flags=re.IGNORECASE)


def weak_spam_label(text: str) -> int:
    return 1 if isinstance(text, str) and SPAM_RE.search(text) else 0


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, tag: str) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n[{tag}] confusion matrix [rows=true, cols=pred]:")
    print("           pred 0     pred 1")
    print(f"true 0   {cm[0,0]:>8d}   {cm[0,1]:>8d}")
    print(f"true 1   {cm[1,0]:>8d}   {cm[1,1]:>8d}")
    print(f"[{tag}] precision={p:.4f}  recall={r:.4f}  f1={f1:.4f}")
    return {
        "tag": tag,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "cm": cm.tolist(),
    }


def main() -> None:
    tr = pd.read_csv(ROOT / "sentiment_analysis_training_data.csv")
    va = pd.read_csv(ROOT / "sentiment_analysis_validation_data.csv")

    tr["spam"] = tr["text"].apply(weak_spam_label)
    va["spam"] = va["text"].apply(weak_spam_label)
    print(f"weak spam — train: {tr['spam'].mean():.3f}  val: {va['spam'].mean():.3f}")

    pipe = build_pipeline()

    # 5-fold cross-validated predictions on training (out-of-fold)
    cv_pred = cross_val_predict(
        pipe, tr["text"].fillna(""), tr["spam"], cv=5, n_jobs=-1
    )
    cv_metrics = evaluate(tr["spam"].values, cv_pred, "train CV (5-fold)")

    # Fit on full training, evaluate on validation
    pipe.fit(tr["text"].fillna(""), tr["spam"])
    va_pred = pipe.predict(va["text"].fillna(""))
    va_metrics = evaluate(va["spam"].values, va_pred, "validation")

    # Inspect any disagreements between rule and learned model on validation
    rule = va["spam"].values
    learned = va_pred
    disagree = np.where(rule != learned)[0]
    print(f"\n[validation] rule vs learned disagreements: {len(disagree)}")
    for idx in disagree[:10]:
        print(
            f"  idx={idx} rule={rule[idx]} learned={learned[idx]}  "
            f"text={va['text'].iloc[idx][:120]!r}"
        )

    joblib.dump(pipe, OUT_DIR / "spam_model.joblib")
    metrics = {"train_cv": cv_metrics, "validation": va_metrics}
    (OUT_DIR / "spam_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved: {OUT_DIR/'spam_model.joblib'}")
    print(f"Saved: {OUT_DIR/'spam_metrics.json'}")


if __name__ == "__main__":
    main()
