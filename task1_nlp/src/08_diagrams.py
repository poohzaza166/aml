"""
Task 1 — Diagrams for the report.

Generates:
  - figures/fig_pipeline.png : block-diagram of the two-stage system.
  - figures/fig_model_comparison.png : grouped bars of val accuracy with
    95% CIs for all six models (A–E2).
"""


import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "outputs"
FIG = TASK / "figures"


# ---------- pipeline diagram ----------------------------------------------


def pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.set_axis_off()

    def box(x, y, w, h, text, fc):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.4, edgecolor="black", facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=11, wrap=True)

    def arrow(x0, y0, x1, y1, text=None):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black"),
        )
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, text,
                    ha="center", va="bottom", fontsize=10, style="italic")

    # Input
    box(0.2, 2.0, 1.6, 1.0, "raw\nreview text", "#f1f5f9")

    # Stage 1
    box(2.4, 2.0, 2.4, 1.0,
        "Stage 1\nchar 3–5 TF-IDF\n+ LR  (spam vs review)",
        "#dbeafe")

    # Split
    arrow(1.8, 2.5, 2.4, 2.5)

    # Dummy label branch
    arrow(4.8, 2.7, 6.4, 4.0, text="spam")
    box(6.4, 3.5, 2.0, 1.0, "label = 2\n(dummy)", "#fee2e2")

    # Stage 2 branch
    arrow(4.8, 2.3, 6.4, 1.5, text="review")
    box(6.4, 1.0, 2.4, 1.0,
        "Stage 2\nword 1–2 + char 3–5\nTF-IDF  +  LR",
        "#dcfce7")

    # Output
    box(9.0, 1.0, 1.6, 1.0, "label\n∈ {0, 1}", "#fef3c7")
    arrow(8.8, 1.5, 9.0, 1.5)
    box(9.0, 3.5, 1.6, 1.0, "(test only)", "#fef3c7")
    arrow(8.4, 4.0, 9.0, 4.0)

    ax.text(5.5, 0.2,
            "Spam decoupled because its labels are random in {0,1}; "
            "keeping it would inject 50% label-noise on ≈26% of training data.",
            ha="center", fontsize=9, style="italic", color="#475569")

    fig.tight_layout()
    fig.savefig(FIG / "fig_pipeline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Saved", FIG / "fig_pipeline.png")


# ---------- model comparison ----------------------------------------------


def model_comparison() -> None:
    sparse = json.loads((OUT / "sentiment_metrics.json").read_text())
    dense = json.loads((OUT / "sentiment_glove_metrics.json").read_text())

    rows = []
    for key, label in [
        ("A_LR_word",        "A: word TF-IDF + LR"),
        ("B_NB_word",        "B: word TF-IDF + NB"),
        ("C_SVM_word",       "C: word TF-IDF + SVM"),
        ("D_LR_word_char",   "D: word+char TF-IDF + LR"),
        ("E1_mean_glove",    "E1: mean GloVe + LR"),
        ("E2_tfidf_glove",   "E2: tfidf-GloVe + LR"),
    ]:
        m = sparse.get(key) or dense.get(key)
        rows.append((label, m["val_acc"], m["val_acc_ci"]
                    [0], m["val_acc_ci"][1]))

    labels, accs, lo, hi = zip(*rows)
    yerr = np.array([[a - l for a, l in zip(accs, lo)],
                     [h - a for a, h in zip(accs, hi)]])

    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = ["#3b82f6"] * 4 + ["#10b981"] * 2
    bars = ax.bar(labels, accs, yerr=yerr, capsize=4,
                  color=colors, edgecolor="black")
    ax.axhline(0.5, color="gray", linestyle="--", label="random")
    ax.set_ylabel("validation accuracy (de-spammed)")
    ax.set_ylim(0.5, 0.85)
    ax.set_title(
        "Sentiment classifier comparison (n=1066 val, ±95% bootstrap CI)")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.005,
                f"{a:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_model_comparison.png", dpi=140)
    plt.close(fig)
    print("Saved", FIG / "fig_model_comparison.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    pipeline_diagram()
    model_comparison()
