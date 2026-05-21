"""
Task 2 — Evaluation figures.

Outputs:
  - figures/fig_ced.png             : cumulative error distribution curves
  - figures/fig_box_per_method.png  : boxplot of NME per method
  - figures/fig_box_per_landmark.png: boxplot of per-landmark Euclidean
                                       error for the best model
  - figures/fig_qualitative.png     : example predictions vs ground truth
"""


import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import FIG, OUT, load_splits, nme, per_face_errors


METHODS = [
    ("M0_mean_shape", "pred_val_M0.npy", "M0: mean shape"),
    ("M1_HOG_ridge", "pred_val_M1.npy", "M1: HOG + Ridge"),
    ("M2_cascaded", "pred_val_M2.npy", "M2: Cascaded HOG"),
]


def main() -> None:
    _, _, Xva, pva, _ = load_splits()

    per_face_all = {}
    nme_all = {}
    for key, fname, label in METHODS:
        pred = np.load(OUT / fname)
        per_face_all[label] = per_face_errors(pred, pva)
        nme_all[label] = nme(pred, pva)
        print(f"{label}:  NME mean = {nme_all[label].mean():.4f}  "
              f"median = {np.median(nme_all[label]):.4f}  "
              f"px mean = {per_face_all[label].mean():.2f}")

    # ---- CED curve ----
    fig, ax = plt.subplots(figsize=(7, 5))
    thresholds = np.linspace(0, 0.2, 200)
    for label, nme_v in nme_all.items():
        ax.plot(thresholds,
                [np.mean(nme_v <= t) for t in thresholds],
                lw=2, label=label)
    ax.set_xlabel("NME (normalised by inter-ocular distance)")
    ax.set_ylabel("fraction of validation images with NME ≤ x")
    ax.set_title("Cumulative Error Distribution (n=211 val images)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ced.png", dpi=140)
    plt.close(fig)
    print("saved", FIG / "fig_ced.png")

    # ---- Boxplot of NME per method ----
    fig, ax = plt.subplots(figsize=(7, 4.6))
    labels = list(nme_all.keys())
    data = [nme_all[k] for k in labels]
    bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True)
    colors = ["#fca5a5", "#93c5fd", "#86efac"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_ylabel("NME per face")
    ax.set_title("Validation NME distribution by method")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig_box_per_method.png", dpi=140)
    plt.close(fig)
    print("saved", FIG / "fig_box_per_method.png")

    # ---- Per-landmark boxplot for best model ----
    pred_best = np.load(OUT / "pred_val_M2.npy")
    per_lm = np.sqrt(np.sum((pred_best - pva) ** 2, axis=-1))  # (n, 5)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    lm_names = ["L eye (0)", "R eye (1)", "Nose (2)", "L mouth (3)", "R mouth (4)"]
    bp = ax.boxplot([per_lm[:, k] for k in range(5)], labels=lm_names,
                    showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#bfdbfe")
    ax.set_ylabel("Euclidean error (pixels, 256-canvas)")
    ax.set_title("Per-landmark error for M2 (cascaded)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig_box_per_landmark.png", dpi=140)
    plt.close(fig)
    print("saved", FIG / "fig_box_per_landmark.png")

    # ---- Qualitative: best, median, worst predictions ----
    pf_best = per_face_errors(pred_best, pva)
    order = np.argsort(pf_best)
    idx_good = order[:6]
    idx_bad = order[-6:]
    idx_mid = order[len(order)//2 - 3:len(order)//2 + 3]
    panel = list(idx_good) + list(idx_mid) + list(idx_bad)
    rows, cols = 3, 6
    titles = ["best 6", "median 6", "worst 6"]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    for r in range(rows):
        for c in range(cols):
            i = panel[r * cols + c]
            ax = axes[r, c]
            ax.imshow(Xva[i])
            ax.plot(pva[i, :, 0], pva[i, :, 1], "+g", markersize=12, mew=2, label="GT")
            ax.plot(pred_best[i, :, 0], pred_best[i, :, 1], "xr",
                    markersize=10, mew=2, label="pred")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"#{i} px={pf_best[i]:.1f}", fontsize=9)
        axes[r, 0].set_ylabel(titles[r])
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle("M2 cascaded — qualitative examples (green=GT, red=pred)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_qualitative.png", dpi=130)
    plt.close(fig)
    print("saved", FIG / "fig_qualitative.png")

    # summary numbers for the report
    summary = {}
    for label, v in nme_all.items():
        summary[label] = {
            "NME_mean": float(v.mean()),
            "NME_median": float(np.median(v)),
            "NME_p90": float(np.percentile(v, 90)),
            "px_mean": float(per_face_all[label].mean()),
            "px_median": float(np.median(per_face_all[label])),
        }
    (OUT / "comparison_table.json").write_text(json.dumps(summary, indent=2))
    print("saved comparison table")


if __name__ == "__main__":
    main()
