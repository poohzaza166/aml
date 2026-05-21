"""
Task 1 — Produce the final test-set predictions CSV.

Output: shape (n_test, 1) — one column of integer labels in {0, 1, 2}:
  0 = negative review
  1 = positive review
  2 = spam (dummy label, per the assignment)

Pipeline at test time:
  1. spam_model (char-n-gram TF-IDF + LR) predicts whether the item is
     spam. Items with spam==1 receive label 2.
  2. Remaining items are passed to sent_D (word+char TF-IDF + LR) and
     receive label 0 or 1.

The output file is written using the "save as csv" recipe from the
provided Colab worksheet: a single-column dataframe.
"""


from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    test = pd.read_csv(ROOT / "sentiment_analysis_test_data.csv")
    print(f"test items: {len(test)}")

    spam = joblib.load(OUT / "spam_model.joblib")
    sent = joblib.load(OUT / "sent_D_LR_word_char.joblib")

    is_spam = spam.predict(test["text"].fillna(""))
    print(f"  flagged as spam: {int(is_spam.sum())}")

    # Initialise all to spam dummy label, overwrite non-spam with sentiment
    labels = np.full(len(test), 2, dtype=int)
    review_mask = is_spam == 0
    if review_mask.any():
        sent_pred = sent.predict(test.loc[review_mask, "text"].fillna("").tolist())
        labels[np.asarray(review_mask)] = sent_pred.astype(int)

    # Mirror the Colab "save as csv" function: single column, no header
    # (using the standard sklearn pipeline convention).
    out_df = pd.DataFrame({"label": labels})
    out_path = OUT / "sentiment_test_predictions.csv"
    out_df.to_csv(out_path, index=False)

    print(f"label distribution: "
          f"0={int((labels==0).sum())} "
          f"1={int((labels==1).sum())} "
          f"2={int((labels==2).sum())}")
    print(f"Saved: {out_path}")

    # Quick sanity-check report on first 5 rows
    print("\nFirst 5 predictions:")
    for i, (txt, lab) in enumerate(zip(test["text"].head(5), labels[:5])):
        snip = txt[:90].replace("\n", " ").replace("\r", " ")
        print(f"  [{i}] label={lab}  {snip!r}")


if __name__ == "__main__":
    main()
