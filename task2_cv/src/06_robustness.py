"""
Task 2 — Robustness analysis (Section 10-mark requirement).

We take the best model (M2 cascaded HOG-ridge), retrain it once on the
clean training set, then evaluate it on perturbed *validation* sets to
see how the prediction quality degrades.

Perturbations tested:
  (a) Gaussian noise of increasing std (sigma in pixel intensity units
      on uint8 images).
  (b) In-plane rotation around the image centre (the ground-truth
      points are rotated identically, so the *aligned* prediction is
      what we score).
  (c) Multiplicative contrast/brightness change.

Why these three?
  - Gaussian noise: the canonical sensor-noise model; tells us how
    sensitive HOG (which is built on local gradients) is to small
    perturbations of intensity.
  - In-plane rotation: HOG is *not* rotation-invariant — the histogram
    of oriented gradients literally encodes orientations. So we expect
    fast degradation. This is exactly the "transformations that
    distort the images" scenario from the brief.
  - Contrast change: histogram-equalisation in our pre-processing
    should largely absorb this, so we expect *little* degradation —
    this is a positive robustness test.
"""


import json
from pathlib import Path
from time import time

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

from common import (
    FIG, OUT,
    load_splits, preprocess_stack, scale_points,
    per_face_errors, nme, to_grey_eq,
)

# re-import params from the cascaded module to stay consistent
import importlib.util
spec = importlib.util.spec_from_file_location(
    "casc", Path(__file__).resolve().parent / "04_cascaded.py"
)
casc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(casc)

IMG_SIZE = casc.IMG_SIZE
GLOBAL_HOG = casc.GLOBAL_HOG
shape_indexed_features = casc.shape_indexed_features
global_hog = casc.global_hog


def fit_cascade(Gtr, ptr_128):
    """Retrain a 3-stage cascade and return (base, stages).
    Used so we can deterministically re-apply it to perturbed images."""
    n_train = Gtr.shape[0]
    Fg = global_hog(Gtr)
    base = Ridge(alpha=100.0).fit(Fg, ptr_128.reshape(n_train, -1))
    Ŝ = base.predict(Fg).reshape(n_train, 5, 2)
    stages = []
    for _ in range(3):
        Fl = shape_indexed_features(Gtr, Ŝ)
        reg = Ridge(alpha=50.0).fit(Fl, (ptr_128 - Ŝ).reshape(n_train, -1))
        Ŝ = Ŝ + reg.predict(Fl).reshape(n_train, 5, 2)
        stages.append(reg)
    return base, stages


def predict_cascade(images_g, base, stages):
    n = images_g.shape[0]
    Ŝ = base.predict(global_hog(images_g)).reshape(n, 5, 2)
    for reg in stages:
        Ŝ = Ŝ + reg.predict(shape_indexed_features(images_g, Ŝ)).reshape(n, 5, 2)
    return Ŝ


# ---------- perturbations on 256-canvas RGB uint8 ----------------------


def add_gaussian_noise(images: np.ndarray, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    noise = rng.normal(0, sigma, images.shape)
    out = images.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def rotate_set(images: np.ndarray, points: np.ndarray, angle_deg: float):
    h, w = images.shape[1], images.shape[2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rot_imgs = np.empty_like(images)
    for i in range(len(images)):
        rot_imgs[i] = cv2.warpAffine(images[i], M, (w, h),
                                     borderMode=cv2.BORDER_REPLICATE)
    # Apply same affine to landmarks
    pts = points.reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.concatenate([pts, ones], axis=1)
    pts_rot = (M @ pts_h.T).T  # (n*5, 2)
    return rot_imgs, pts_rot.reshape(points.shape)


def scale_contrast(images: np.ndarray, gamma: float) -> np.ndarray:
    """Apply intensity gamma correction. gamma=1 is identity; gamma<1
    brightens, gamma>1 darkens."""
    lut = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(images, lut)


def main():
    print("loading + training one cascade on clean data...")
    Xtr, ptr, Xva, pva, _ = load_splits()
    Gtr = preprocess_stack(Xtr, IMG_SIZE)
    ptr_128 = scale_points(ptr, 256, IMG_SIZE)
    t0 = time()
    base, stages = fit_cascade(Gtr, ptr_128)
    print(f"  trained in {time()-t0:.1f}s")

    # baseline val NME
    Gva = preprocess_stack(Xva, IMG_SIZE)
    Ŝva = predict_cascade(Gva, base, stages)
    base_nme = nme(Ŝva * 2, pva).mean()
    print(f"clean val NME = {base_nme:.4f}")

    sweeps = {}

    # ---- (a) Gaussian noise ----
    sigmas = [0, 5, 10, 15, 20, 30, 40, 60]
    sweeps["gaussian_noise_sigma"] = {"x": sigmas, "nme": []}
    for s in sigmas:
        if s == 0:
            G = Gva
        else:
            Xn = add_gaussian_noise(Xva, s)
            G = preprocess_stack(Xn, IMG_SIZE)
        Ŝ = predict_cascade(G, base, stages)
        v = nme(Ŝ * 2, pva).mean()
        sweeps["gaussian_noise_sigma"]["nme"].append(float(v))
        print(f"  sigma={s:>3}  NME={v:.4f}")

    # ---- (b) Rotation ----
    angles = [-30, -20, -10, -5, 0, 5, 10, 20, 30]
    sweeps["rotation_deg"] = {"x": angles, "nme": []}
    for ang in angles:
        if ang == 0:
            Xr, pvr = Xva, pva
        else:
            Xr, pvr = rotate_set(Xva, pva, ang)
        G = preprocess_stack(Xr, IMG_SIZE)
        Ŝ = predict_cascade(G, base, stages)
        v = nme(Ŝ * 2, pvr).mean()
        sweeps["rotation_deg"]["nme"].append(float(v))
        print(f"  angle={ang:+}  NME={v:.4f}")

    # ---- (c) Gamma contrast ----
    gammas = [0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5]
    sweeps["gamma"] = {"x": gammas, "nme": []}
    for g in gammas:
        if g == 1.0:
            G = Gva
        else:
            Xg = scale_contrast(Xva, g)
            G = preprocess_stack(Xg, IMG_SIZE)
        Ŝ = predict_cascade(G, base, stages)
        v = nme(Ŝ * 2, pva).mean()
        sweeps["gamma"]["nme"].append(float(v))
        print(f"  gamma={g}  NME={v:.4f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    a = axes[0]
    a.plot(sweeps["gaussian_noise_sigma"]["x"],
           sweeps["gaussian_noise_sigma"]["nme"],
           "o-", lw=2, color="#dc2626")
    a.axhline(base_nme, ls="--", color="gray", label=f"clean NME = {base_nme:.3f}")
    a.set_xlabel("Gaussian noise σ (intensity 0–255)")
    a.set_ylabel("validation NME")
    a.set_title("Robustness: Gaussian noise")
    a.grid(True, alpha=0.3); a.legend()

    a = axes[1]
    a.plot(sweeps["rotation_deg"]["x"], sweeps["rotation_deg"]["nme"],
           "o-", lw=2, color="#2563eb")
    a.axhline(base_nme, ls="--", color="gray")
    a.set_xlabel("rotation angle (deg)")
    a.set_ylabel("validation NME")
    a.set_title("Robustness: in-plane rotation")
    a.grid(True, alpha=0.3)

    a = axes[2]
    a.plot(sweeps["gamma"]["x"], sweeps["gamma"]["nme"],
           "o-", lw=2, color="#16a34a")
    a.axhline(base_nme, ls="--", color="gray")
    a.set_xlabel("gamma (intensity ^gamma)")
    a.set_ylabel("validation NME")
    a.set_title("Robustness: gamma contrast")
    a.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "fig_robustness.png", dpi=140)
    plt.close(fig)
    print("saved", FIG / "fig_robustness.png")

    (OUT / "robustness.json").write_text(json.dumps(sweeps, indent=2))


if __name__ == "__main__":
    main()
