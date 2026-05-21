"""
Task 2 — Model M2: cascaded shape regression with shape-indexed HOG.

Why this approach?
  - Kazemi & Sullivan (2014) and Cao et al. (2014, "Face Alignment by
    Explicit Shape Regression") observed that regressing the full
    shape from global image features is hard because the same pixel
    means different things depending on where the face actually is.
  - Cascaded shape regression solves this by *iterating*: predict an
    initial shape from a global descriptor, then extract features
    relative to that estimate, then predict a residual update, etc.
  - Local patches around each predicted landmark carry much more
    specific information than the whole image.

Architecture:
  Stage 0: initialise to the global HOG + ridge prediction (M1).
  Stage 1: extract HOG patches (32×32) around each landmark estimate,
           concatenate, predict shape update with ridge.
  Stage 2: same again on the updated estimate.

Loss at each stage: residual sum of squares + alpha * ||W||² (ridge).
Image features per stage: 5 landmarks × HOG(32×32) ≈ 5 × ~500 = 2500.
"""


import json
from pathlib import Path
from time import time

import numpy as np
from skimage.feature import hog
from sklearn.linear_model import Ridge

from common import (
    FIG, OUT,
    load_splits, preprocess_stack, scale_points,
    per_face_errors, nme,
)

# Stage 0 uses the same global HOG as M1
IMG_SIZE = 128

GLOBAL_HOG = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True,
)

# Local HOG around each landmark
PATCH_SIZE = 32  # in pixels of the 128-canvas
LOCAL_HOG = dict(
    orientations=8,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True,
)


def global_hog(images_g):
    return np.stack([hog(im, **GLOBAL_HOG).astype(np.float32) for im in images_g])


def crop_patch(img: np.ndarray, cx: float, cy: float, half: int) -> np.ndarray:
    """Return a (2*half, 2*half) patch around (cx, cy) with zero pad
    when outside the image."""
    h, w = img.shape
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x1 - w), max(0, y1 - h)
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    patch = np.zeros((2 * half, 2 * half), dtype=img.dtype)
    patch[py0:2 * half - py1, px0:2 * half - px1] = img[y0c:y1c, x0c:x1c]
    return patch


def shape_indexed_features(images_g, shapes_128):
    """For each image, extract HOG around each landmark prediction.

    images_g: (n, 128, 128) uint8
    shapes_128: (n, 5, 2) in 128-canvas units
    returns: (n, d)
    """
    half = PATCH_SIZE // 2
    out = []
    for i in range(images_g.shape[0]):
        feats = []
        for k in range(shapes_128.shape[1]):
            cx, cy = shapes_128[i, k]
            patch = crop_patch(images_g[i], cx, cy, half)
            feats.append(hog(patch, **LOCAL_HOG))
        out.append(np.concatenate(feats).astype(np.float32))
    return np.stack(out, axis=0)


def main() -> None:
    print("loading data...")
    Xtr, ptr, Xva, pva, Xte = load_splits()
    Gtr = preprocess_stack(Xtr, IMG_SIZE)
    Gva = preprocess_stack(Xva, IMG_SIZE)
    Gte = preprocess_stack(Xte, IMG_SIZE)

    ptr_128 = scale_points(ptr, 256, IMG_SIZE)  # (n, 5, 2)
    pva_128 = scale_points(pva, 256, IMG_SIZE)
    n_train, n_val, n_test = len(Xtr), len(Xva), len(Xte)

    # ---- Stage 0 : global HOG + ridge → initial shape ----
    print("\nStage 0 (global HOG + ridge)...")
    t0 = time()
    Fg_tr = global_hog(Gtr)
    Fg_va = global_hog(Gva)
    Fg_te = global_hog(Gte)
    print(f"  global HOG done in {time()-t0:.1f}s shape={Fg_tr.shape}")
    base = Ridge(alpha=100.0)
    base.fit(Fg_tr, ptr_128.reshape(n_train, -1))
    Ŝtr = base.predict(Fg_tr).reshape(n_train, 5, 2)
    Ŝva = base.predict(Fg_va).reshape(n_val, 5, 2)
    Ŝte = base.predict(Fg_te).reshape(n_test, 5, 2)

    nme_s0 = nme(Ŝva * 2, pva)  # convert 128→256
    print(f"  stage-0 val NME = {nme_s0.mean():.4f}")

    stage_results = [{"stage": 0, "val_NME_mean": float(nme_s0.mean()),
                       "val_px_err_mean": float(per_face_errors(Ŝva * 2, pva).mean())}]

    # ---- Cascaded stages: shape-indexed local HOG + ridge ----
    n_stages = 3
    alpha_stage = 50.0
    for s in range(1, n_stages + 1):
        print(f"\nStage {s} (shape-indexed local HOG)...")
        t0 = time()
        Fl_tr = shape_indexed_features(Gtr, Ŝtr)
        Fl_va = shape_indexed_features(Gva, Ŝva)
        Fl_te = shape_indexed_features(Gte, Ŝte)
        print(f"  local HOG done in {time()-t0:.1f}s  shape={Fl_tr.shape}")
        delta_target = (ptr_128 - Ŝtr).reshape(n_train, -1)
        reg = Ridge(alpha=alpha_stage)
        reg.fit(Fl_tr, delta_target)
        Ŝtr = Ŝtr + reg.predict(Fl_tr).reshape(n_train, 5, 2)
        Ŝva = Ŝva + reg.predict(Fl_va).reshape(n_val, 5, 2)
        Ŝte = Ŝte + reg.predict(Fl_te).reshape(n_test, 5, 2)
        nme_s = nme(Ŝva * 2, pva)
        px_s = per_face_errors(Ŝva * 2, pva)
        print(f"  stage-{s} val NME = {nme_s.mean():.4f}  "
              f"px = {px_s.mean():.2f}")
        stage_results.append({"stage": s, "val_NME_mean": float(nme_s.mean()),
                              "val_px_err_mean": float(px_s.mean())})

    pred_va_256 = Ŝva * 2
    pred_te_256 = Ŝte * 2
    np.save(OUT / "pred_val_M2.npy", pred_va_256)
    np.save(OUT / "pred_test_M2.npy", pred_te_256)

    summary = {
        "model": "M2_cascaded_HOG_ridge",
        "img_size": IMG_SIZE,
        "patch_size": PATCH_SIZE,
        "n_stages": n_stages,
        "alpha_stage": alpha_stage,
        "stages": stage_results,
        "final_val_NME_mean": stage_results[-1]["val_NME_mean"],
        "final_val_px_err_mean": stage_results[-1]["val_px_err_mean"],
    }
    (OUT / "metrics_M2.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT / 'metrics_M2.json'}")


if __name__ == "__main__":
    main()
