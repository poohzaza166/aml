"""
Task 2 — Comparison / summary charts.

Produces:
  - figures/fig_model_comparison.png : NME + px-error bars across all
                                       three models, with annotations.
  - figures/fig_cascade_progress.png : NME at each cascade stage.

Run after 02, 03, 04 and 05 have completed.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import FIG, OUT


def model_comparison():
    M0 = json.loads((OUT / "metrics_M0.json").read_text())
    M1 = json.loads((OUT / "metrics_M1.json").read_text())
    M2 = json.loads((OUT / "metrics_M2.json").read_text())

    names = ["M0\nmean shape", "M1\nHOG + Ridge", "M2\ncascaded HOG"]
    nme_vals = [
        M0["val_NME_mean"],
        M1["best_val_NME"],
        M2["final_val_NME_mean"],
    ]
    px_vals = [
        M0["val_mean_euclid_px_mean"],
        M1["sweep"][[s["alpha"] for s in M1["sweep"]].index(M1["best_alpha"])]["val_mean_px_err"],
        M2["final_val_px_err_mean"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#fca5a5", "#93c5fd", "#86efac"]

    ax = axes[0]
    bars = ax.bar(names, nme_vals, color=colors, edgecolor="black")
    ax.set_ylabel("validation NME (lower is better)")
    ax.set_title("NME by model (n=211)")
    for b, v in zip(bars, nme_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003,
                f"{v:.4f}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(nme_vals) * 1.18)

    ax = axes[1]
    bars = ax.bar(names, px_vals, color=colors, edgecolor="black")
    ax.set_ylabel("mean Euclidean error (pixels, 256-canvas)")
    ax.set_title("Mean per-face pixel error (n=211)")
    for b, v in zip(bars, px_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3,
                f"{v:.2f}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(px_vals) * 1.18)

    fig.tight_layout()
    fig.savefig(FIG / "fig_model_comparison.png", dpi=140)
    plt.close(fig)
    print("saved", FIG / "fig_model_comparison.png")


def cascade_progress():
    M2 = json.loads((OUT / "metrics_M2.json").read_text())
    stages = M2["stages"]
    x = [s["stage"] for s in stages]
    nme = [s["val_NME_mean"] for s in stages]
    px = [s["val_px_err_mean"] for s in stages]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(x, nme, "o-", lw=2, color="#0ea5e9")
    for xi, yi in zip(x, nme):
        ax.text(xi, yi + 0.001, f"{yi:.4f}", ha="center", fontsize=9)
    ax.set_xlabel("cascade stage")
    ax.set_ylabel("validation NME")
    ax.set_title("NME by cascade stage")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)

    ax = axes[1]
    ax.plot(x, px, "o-", lw=2, color="#16a34a")
    for xi, yi in zip(x, px):
        ax.text(xi, yi + 0.05, f"{yi:.2f}", ha="center", fontsize=9)
    ax.set_xlabel("cascade stage")
    ax.set_ylabel("mean px error (256-canvas)")
    ax.set_title("Pixel error by cascade stage")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)

    fig.tight_layout()
    fig.savefig(FIG / "fig_cascade_progress.png", dpi=140)
    plt.close(fig)
    print("saved", FIG / "fig_cascade_progress.png")


if __name__ == "__main__":
    model_comparison()
    cascade_progress()
