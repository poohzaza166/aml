"""
Task 1 — Stage-2 sentiment classifier (positive vs negative reviews).

Pipeline (overall, two stages):
    text  ─► [stage-1 spam detector]  ─┬─ spam   ─► label = 2 (dummy)
                                       └─ review ─► [stage-2 sentiment] ─► label ∈ {0, 1}

This file deals with stage 2. Spam is excluded from training because the
assignment specifies that spam labels are randomly assigned to {0, 1}, so
keeping them would inject label noise of ~50% on roughly a quarter of the
training data, biasing the decision boundary toward chance.

Models compared (all linear, controlled for representation only):
  A. word TF-IDF (1–2 gram)  +  Logistic Regression
  B. word TF-IDF (1–2 gram)  +  Multinomial Naive Bayes  [Pang & Lee 2002]
  C. word TF-IDF (1–2 gram)  +  Linear SVM (hinge loss)
  D. word + char TF-IDF (union) + Logistic Regression  [richer representation]

For each: 5-fold CV on (non-spam) training, then held-out validation.

Evaluation:
  - accuracy, macro-F1, confusion matrix on validation (non-spam subset).
  - per-class precision/recall on validation.
  - 95% bootstrap CI on accuracy (1000 resamples).

References:
  - Pang, Lee & Vaithyanathan (2002), "Thumbs up? Sentiment Classification
    using Machine Learning Techniques", EMNLP.
  - Joachims (1998), "Text Categorization with Support Vector Machines",
    ECML.
  - Wang & Manning (2012), "Baselines and Bigrams: Simple, Good Sentiment
    and Topic Classification", ACL.
"""


import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "outputs"
OUT.mkdir(exist_ok=True)

SPAM_RE = re.compile(r"^\s*subject\s*:", flags=re.IGNORECASE)


def is_spam(text: str) -> bool:
    return isinstance(text, str) and bool(SPAM_RE.search(text))


def clean_review(text: str) -> str:
    """Light pre-processing for movie reviews.

    The corpus is already tokenised (lowercase, separated punctuation),
    so we only normalise whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- pipelines ------------------------------------------------------


def pipe_lr_word() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                C=4.0, max_iter=2000, solver="liblinear",
            )),
        ]
    )


def pipe_nb_word() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=False,  # NB prefers raw counts/TF
                strip_accents="unicode",
            )),
            ("clf", MultinomialNB(alpha=0.3)),
        ]
    )


def pipe_svm_word() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("clf", LinearSVC(C=1.0, max_iter=3000)),
        ]
    )


def pipe_lr_word_char() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=3,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
        ]
    )
    return Pipeline([("feats", features), ("clf", LogisticRegression(
        C=4.0, max_iter=2000, solver="liblinear"
    ))])


MODELS: dict[str, Callable[[], Pipeline]] = {
    "A_LR_word": pipe_lr_word,
    "B_NB_word": pipe_nb_word,
    "C_SVM_word": pipe_svm_word,
    "D_LR_word_char": pipe_lr_word_char,
}


# ---------- evaluation -----------------------------------------------------


def bootstrap_acc_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    accs = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
    accs = np.asarray(accs)
    return float(np.quantile(accs, 0.025)), float(np.quantile(accs, 0.975))


@dataclass
class Result:
    name: str
    cv_acc_mean: float
    cv_acc_std: float
    val_acc: float
    val_acc_ci: tuple[float, float]
    val_macro_f1: float
    val_cm: list[list[int]]
    val_report: str


def evaluate_model(
    name: str,
    model: Pipeline,
    Xtr: list[str], ytr: np.ndarray,
    Xva: list[str], yva: np.ndarray,
) -> Result:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(model, Xtr, ytr, cv=cv, scoring="accuracy", n_jobs=-1)
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xva)
    val_acc = accuracy_score(yva, y_pred)
    val_f1 = f1_score(yva, y_pred, average="macro")
    cm = confusion_matrix(yva, y_pred, labels=[0, 1])
    ci_lo, ci_hi = bootstrap_acc_ci(yva, y_pred)
    rep = classification_report(yva, y_pred, target_names=["neg", "pos"], digits=4)
    print(
        f"\n[{name}] cv-acc={cv_scores.mean():.4f}±{cv_scores.std():.4f}  "
        f"val-acc={val_acc:.4f}  val-F1={val_f1:.4f}  "
        f"95%CI=({ci_lo:.4f}, {ci_hi:.4f})"
    )
    print(rep)
    return Result(
        name=name,
        cv_acc_mean=float(cv_scores.mean()),
        cv_acc_std=float(cv_scores.std()),
        val_acc=float(val_acc),
        val_acc_ci=(ci_lo, ci_hi),
        val_macro_f1=float(val_f1),
        val_cm=cm.tolist(),
        val_report=rep,
    )


# ---------- main -----------------------------------------------------------


def main() -> None:
    tr = pd.read_csv(ROOT / "sentiment_analysis_training_data.csv")
    va = pd.read_csv(ROOT / "sentiment_analysis_validation_data.csv")

    tr["is_spam"] = tr["text"].apply(is_spam)
    va["is_spam"] = va["text"].apply(is_spam)

    tr_clean = tr[~tr["is_spam"]].copy()
    va_clean = va[~va["is_spam"]].copy()
    tr_clean["text"] = tr_clean["text"].apply(clean_review)
    va_clean["text"] = va_clean["text"].apply(clean_review)

    print(f"non-spam train: {len(tr_clean)}  non-spam val: {len(va_clean)}")
    print(f"train balance: {tr_clean['label'].value_counts().to_dict()}")
    print(f"val balance:   {va_clean['label'].value_counts().to_dict()}")

    Xtr = tr_clean["text"].tolist()
    ytr = tr_clean["label"].astype(int).values
    Xva = va_clean["text"].tolist()
    yva = va_clean["label"].astype(int).values

    all_results: list[Result] = []
    for name, ctor in MODELS.items():
        model = ctor()
        res = evaluate_model(name, model, Xtr, ytr, Xva, yva)
        all_results.append(res)
        joblib.dump(model, OUT / f"sent_{name}.joblib")

    # Persist a compact summary
    summary = {
        r.name: {
            "cv_acc_mean": r.cv_acc_mean,
            "cv_acc_std": r.cv_acc_std,
            "val_acc": r.val_acc,
            "val_acc_ci": list(r.val_acc_ci),
            "val_macro_f1": r.val_macro_f1,
            "val_cm": r.val_cm,
        }
        for r in all_results
    }
    (OUT / "sentiment_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {OUT/'sentiment_metrics.json'}")


if __name__ == "__main__":
    main()
