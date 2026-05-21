"""
Task 1 — Data exploration.

Goals:
  - Quantify class balance.
  - Characterise the contaminating spam (length, structure markers).
  - Surface a candidate spam signature ("Subject:" prefix, \r\n, digit density).
  - Estimate the spam contamination rate in training / validation.
"""


import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(DATA_DIR / "sentiment_analysis_training_data.csv")
    va = pd.read_csv(DATA_DIR / "sentiment_analysis_validation_data.csv")
    te = pd.read_csv(DATA_DIR / "sentiment_analysis_test_data.csv")
    return tr, va, te


SPAM_RE = re.compile(r"^\s*subject\s*:", flags=re.IGNORECASE)


def looks_like_spam(text: str) -> bool:
    """A deliberately conservative heuristic used only for exploration."""
    if not isinstance(text, str):
        return False
    if SPAM_RE.search(text):
        return True
    # The reviews corpus is single-sentence; spam carries CRLF line breaks.
    if "\r\n" in text:
        return True
    return False


def summarise(name: str, df: pd.DataFrame) -> None:
    n = len(df)
    has_label = "label" in df.columns
    print(f"\n=== {name} ({n} rows) ===")
    if has_label:
        print("label counts:")
        print(df["label"].value_counts())
    lens = df["text"].str.len()
    print(f"len: mean={lens.mean():.0f}  median={lens.median():.0f}  "
          f"p95={lens.quantile(0.95):.0f}  max={lens.max()}")
    spam_mask = df["text"].apply(looks_like_spam)
    print(f"heuristic spam: {spam_mask.sum()}/{n}  ({100*spam_mask.mean():.1f}%)")
    if has_label:
        print("heuristic spam per label:")
        print(df.assign(spam=spam_mask).groupby("label")["spam"].agg(["sum", "mean"]))


def plot_length_hist(tr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    spam_mask = tr["text"].apply(looks_like_spam)
    ax[0].hist(np.log10(tr.loc[~spam_mask, "text"].str.len().clip(lower=1)),
               bins=40, alpha=0.7, label="review-like")
    ax[0].hist(np.log10(tr.loc[spam_mask, "text"].str.len().clip(lower=1)),
               bins=40, alpha=0.7, label="spam-like (heuristic)")
    ax[0].set_xlabel("log10(text length in chars)")
    ax[0].set_ylabel("count")
    ax[0].set_title("Length distribution (training)")
    ax[0].legend()

    ax[1].bar(["spam-like", "review-like"],
              [spam_mask.sum(), (~spam_mask).sum()])
    ax[1].set_title("Heuristic spam vs review (training)")
    ax[1].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(OUT / "fig_explore_lengths.png", dpi=120)
    plt.close(fig)


def main() -> None:
    tr, va, te = load()
    summarise("training", tr)
    summarise("validation", va)
    summarise("test", te)
    plot_length_hist(tr)
    print(f"\nFigure saved: {OUT/'fig_explore_lengths.png'}")


if __name__ == "__main__":
    main()
