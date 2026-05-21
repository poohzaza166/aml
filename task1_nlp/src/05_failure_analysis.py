"""
Task 1 — Qualitative inspection of the best sentiment model (D: word+char
TF-IDF + LR).

What this produces:
  - figures/fig_confusion_*.png : confusion matrices for each model on
    the de-spammed validation set.
  - outputs/failures.md         : a curated set of misclassifications
    with the per-example margin (decision-function value), grouped by
    failure mode (negation, sarcasm/irony, mixed sentiment, very short).
  - outputs/top_features.md     : the highest-weight n-grams per class
    from model A (word-only LR), useful for the report's discussion of
    learned lexicon.
"""


import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "outputs"
FIG = TASK / "figures"
FIG.mkdir(exist_ok=True)

SPAM_RE = re.compile(r"^\s*subject\s*:", flags=re.IGNORECASE)


def is_spam(t):
    return isinstance(t, str) and bool(SPAM_RE.search(t))


def main() -> None:
    va = pd.read_csv(ROOT / "sentiment_analysis_validation_data.csv")
    va_clean = va[~va["text"].apply(is_spam)].reset_index(drop=True)
    yva = va_clean["label"].astype(int).values

    # Confusion matrices for all models
    model_files = sorted(OUT.glob("sent_*.joblib"))
    print("models:", [m.name for m in model_files])

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    for ax, mf in zip(axes, model_files):
        m = joblib.load(mf)
        # E1/E2 use embedding features so we can't just call .predict on text;
        # skip those (they're handled separately).
        try:
            y_pred = m.predict(va_clean["text"].tolist())
        except Exception:
            ax.set_visible(False)
            continue
        cm = confusion_matrix(yva, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["neg", "pos"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(mf.stem.replace("sent_", ""))
    for ax in axes[len(model_files):]:
        ax.set_visible(False)
    fig.suptitle("Confusion matrices on de-spammed validation (n=1066)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_confusion_all.png", dpi=130)
    plt.close(fig)
    print("Saved", FIG / "fig_confusion_all.png")

    # ---- Best model: D ----
    best = joblib.load(OUT / "sent_D_LR_word_char.joblib")
    # Decision function ~ unbounded log-odds for LR; we use predict_proba.
    proba = best.predict_proba(va_clean["text"].tolist())
    y_pred = best.predict(va_clean["text"].tolist())
    margin = proba[:, 1] - proba[:, 0]  # +ve => pred "pos"

    df = va_clean.copy()
    df["pred"] = y_pred
    df["margin_pos"] = margin
    df["abs_margin"] = np.abs(margin)
    df["wrong"] = (df["pred"] != df["label"]).astype(int)

    wrong = df[df["wrong"] == 1].sort_values("abs_margin", ascending=False)
    print(f"errors: {len(wrong)}/{len(df)}")

    # Tag candidate failure modes by regex on the text
    def tag(t: str) -> str:
        tl = t.lower()
        tags = []
        if re.search(r"\b(not|no|never|n't|nor)\b", tl):
            tags.append("negation")
        if re.search(r"\bbut\b|however|although|despite|even so", tl):
            tags.append("contrast")
        if len(t) < 60:
            tags.append("very-short")
        if re.search(r"\b(too|so|very) (bad|good)\b", tl):
            tags.append("intensifier")
        return ",".join(tags) if tags else "other"

    wrong["failmode"] = wrong["text"].apply(tag)
    counts = wrong["failmode"].str.get_dummies(sep=",").sum().sort_values(ascending=False)
    print("\nFailure mode tag counts (multi-tag):")
    print(counts.head(10))

    # Pick a few of each mode for the report
    sections = []
    sections.append(f"# Task 1 — Curated failures of best model (D: word+char TF-IDF + LR)\n")
    sections.append(f"\nTotal errors: **{len(wrong)} / {len(df)}** "
                    f"({100*len(wrong)/len(df):.2f}%)\n")
    sections.append("\nFailure-mode tag counts (a sample may have multiple tags):\n\n")
    sections.append(counts.head(10).to_markdown())
    sections.append("\n\n## Most confident errors (top 12)\n")
    for _, r in wrong.head(12).iterrows():
        sections.append(
            f"- pred=**{r['pred']}** true=**{r['label']}** "
            f"margin_pos={r['margin_pos']:+.2f}  tags=`{r['failmode']}`\n"
            f"  > {r['text'][:240]}\n"
        )
    (OUT / "failures.md").write_text("\n".join(sections))
    print("Saved", OUT / "failures.md")

    # ---- Top features from model A (word-only) for interpretability ----
    A = joblib.load(OUT / "sent_A_LR_word.joblib")
    vec = A.named_steps["tfidf"]
    clf = A.named_steps["clf"]
    feats = np.asarray(vec.get_feature_names_out())
    coef = clf.coef_[0]
    top_pos = feats[np.argsort(coef)[-25:][::-1]]
    top_neg = feats[np.argsort(coef)[:25]]
    md = [
        "# Top n-gram features (model A: word-only LR)\n",
        "## Top 25 toward **positive** (label=1)\n",
        ", ".join(f"`{f}`" for f in top_pos),
        "\n\n## Top 25 toward **negative** (label=0)\n",
        ", ".join(f"`{f}`" for f in top_neg),
    ]
    (OUT / "top_features.md").write_text("\n".join(md))
    print("Saved", OUT / "top_features.md")


if __name__ == "__main__":
    main()
