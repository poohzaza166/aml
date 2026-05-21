"""Task 2 — pipeline diagram for the report."""


from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from common import FIG


def main():
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
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
                fontsize=10)

    def arrow(x0, y0, x1, y1, text=None, dy=0.18):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black"))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text,
                    ha="center", va="bottom", fontsize=9, style="italic")

    # Row 1: pipeline
    box(0.1, 2.2, 1.9, 1.0, "256×256\nRGB face", "#f1f5f9")
    box(2.3, 2.2, 2.2, 1.0, "preprocess\ngrey + hist-eq\nresize 128", "#e0f2fe")
    box(4.7, 2.2, 2.4, 1.0, "Stage 0\nglobal HOG → Ridge\n(8100-d)", "#dbeafe")
    box(7.3, 2.2, 2.7, 1.0,
        "Stage k ∈ {1,2,3}\nshape-indexed local HOG\nridge → shape Δ", "#dcfce7")
    box(10.2, 2.2, 2.6, 1.0, "5 landmarks\n(x, y) each", "#fef3c7")

    arrow(2.0, 2.7, 2.3, 2.7)
    arrow(4.5, 2.7, 4.7, 2.7)
    arrow(7.1, 2.7, 7.3, 2.7, text="Ŝ₀")
    arrow(10.0, 2.7, 10.2, 2.7, text="Ŝ₃")

    # Loop back arrow for cascade
    ax.annotate("",
                xy=(7.3, 2.0), xytext=(10.0, 2.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#6b7280",
                                connectionstyle="arc3,rad=0.25"))
    ax.text(8.65, 1.45, "Ŝₖ → Ŝₖ + Δₖ  (×3)", ha="center", fontsize=9,
            color="#6b7280", style="italic")

    ax.text(6.5, 0.3,
            "Loss at each Ridge stage: ‖Sₖ − Sₖ₋₁ − Δₖ‖² + α‖W‖². "
            "Initial estimate is the global-HOG regression; "
            "subsequent stages refine by features extracted around the "
            "current landmark estimates.",
            ha="center", fontsize=9, style="italic", color="#475569", wrap=True)

    fig.tight_layout()
    fig.savefig(FIG / "fig_pipeline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / "fig_pipeline.png")


if __name__ == "__main__":
    main()
