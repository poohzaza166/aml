"""
Task 2 — Data exploration for face alignment.

Goals:
  - Visualise random faces with their 5 landmarks overlaid.
  - Look for the "variability" the brief warns about (rotation, scale,
    occlusion, lighting).
  - Compute mean shape (used both as a sanity baseline AND as starting
    point for cascaded shape regression).
  - Diagnose inter-ocular distance distribution (used for NME).
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parents[1]
FIG = TASK / "figures"
FIG.mkdir(exist_ok=True)


def load():
    tr = np.load(TASK / "face_alignment_training_data.npz", allow_pickle=True)
    va = np.load(TASK / "face_alignment_validation_data.npz", allow_pickle=True)
    te = np.load(TASK / "face_alignment_test_data.npz", allow_pickle=True)
    return (
        tr["images"], tr["points"],
        va["images"], va["points"],
        te["images"],
    )


def plot_grid(images, points, title, save_path, n=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=n, replace=False)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(images[i])
        if points is not None:
            ax.plot(points[i, :, 0], points[i, :, 1], "+r", markersize=10, mew=2)
            for k, (x, y) in enumerate(points[i]):
                ax.text(x + 3, y - 3, str(k), color="yellow", fontsize=8)
        ax.set_title(f"#{i}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print("saved", save_path)


def main():
    Xtr, ptr, Xva, pva, Xte = load()
    print("train:", Xtr.shape, ptr.shape)
    print("val:  ", Xva.shape, pva.shape)
    print("test: ", Xte.shape)

    plot_grid(Xtr, ptr, "Training samples", FIG / "fig_train_samples.png", n=12)
    plot_grid(Xva, pva, "Validation samples", FIG / "fig_val_samples.png", n=12)
    plot_grid(Xte, None, "Test samples (no points)", FIG / "fig_test_samples.png", n=12)

    # Mean shape — average landmark positions across training
    mean_shape = ptr.mean(axis=0)  # (5, 2)
    print("mean shape (5x2):")
    print(mean_shape)

    # Inter-ocular distance per face (commonly used to normalise error)
    # We need to identify which two indices are eyes — usually 0 and 1.
    # Inspect the mean shape: typically eyes are above nose (smaller y).
    # x increases left-to-right, y increases top-to-bottom.
    print("\nLandmark Y values in mean shape:", mean_shape[:, 1])
    # The two with smallest y are presumably eyes.

    # Spread of landmark positions across the dataset:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.tab10.colors
    for k in range(ptr.shape[1]):
        ax.scatter(ptr[:, k, 0], ptr[:, k, 1], s=2, alpha=0.3,
                   color=colors[k], label=f"pt {k}")
    ax.scatter(mean_shape[:, 0], mean_shape[:, 1], s=200, marker="X",
               edgecolor="black", facecolor="white", zorder=5,
               label="mean")
    ax.invert_yaxis()
    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)
    ax.set_aspect("equal")
    ax.set_title("Landmark position distribution across training")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_landmark_scatter.png", dpi=120)
    plt.close(fig)
    print("saved", FIG / "fig_landmark_scatter.png")

    # Save mean shape for later use
    np.save(TASK / "outputs" / "mean_shape.npy", mean_shape)

    # Image statistics — any nan / weird?
    print("\nTraining image stats:")
    print(f"  mean: {Xtr.mean():.2f}  std: {Xtr.std():.2f}")
    print(f"  range: [{Xtr.min()}, {Xtr.max()}]")


if __name__ == "__main__":
    (TASK / "outputs").mkdir(exist_ok=True)
    main()
