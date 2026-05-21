"""
Task 2 — Model M1: HOG features + Ridge regression.

Design rationale:
  - HOG (Dalal & Triggs 2005) captures local edge-orientation
    histograms. For aligned face crops, the gradient pattern around
    eyes/nose/mouth is highly structured, so HOG is a strong
    hand-crafted descriptor.
  - Greyscale + histogram-equalisation pre-processing removes most
    illumination variance (Pizer et al. 1987).
  - We resize to 128×128 to keep feature dimensionality manageable
    (~8k features) while keeping enough resolution to separate
    eyes/mouth corners (which are ~10–20 px apart in the original
    256×256 image, so ~5–10 px after the resize — still resolvable).
  - Ridge regression (closed-form L2-regularised least squares) is the
    correct linear regressor when the design matrix has more columns
    than samples (n=2600 < p=8100). Alpha is tuned on validation.
  - Multi-output: 10 outputs simultaneously, one regressor per
    coordinate (sklearn's Ridge handles multi-output natively).

Loss: residual sum of squares + alpha * ||W||².
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
    per_face_errors, nme, inter_ocular_distance,
)

IMG_SIZE = 128

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True,
)


def hog_batch(images_g: np.ndarray) -> np.ndarray:
    """images_g: (n, h, w) uint8 -> (n, d) float32."""
    feats = []
    for i in range(images_g.shape[0]):
        f = hog(images_g[i], **HOG_PARAMS)
        feats.append(f.astype(np.float32))
    return np.stack(feats, axis=0)


def main() -> None:
    print("loading data...")
    Xtr, ptr, Xva, pva, Xte = load_splits()

    print("pre-processing (grey + equalise + resize to 128)...")
    t0 = time()
    Gtr = preprocess_stack(Xtr, IMG_SIZE)
    Gva = preprocess_stack(Xva, IMG_SIZE)
    Gte = preprocess_stack(Xte, IMG_SIZE)
    print(f"  done in {time()-t0:.1f}s")

    # scale points from 256 to 128 px so we regress in the same frame
    ptr_s = scale_points(ptr, 256, IMG_SIZE).reshape(len(ptr), -1)
    pva_s = scale_points(pva, 256, IMG_SIZE).reshape(len(pva), -1)

    print("computing HOG features...")
    t0 = time()
    Ftr = hog_batch(Gtr)
    Fva = hog_batch(Gva)
    Fte = hog_batch(Gte)
    print(f"  done in {time()-t0:.1f}s  shape={Ftr.shape}")

    # Hyperparameter sweep: pick alpha by val NME
    results = []
    best = (None, np.inf, None)
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha)
        t0 = time()
        model.fit(Ftr, ptr_s)
        pred = model.predict(Fva)
        dt = time() - t0
        pred_orig = pred.reshape(-1, 5, 2) * (256 / IMG_SIZE)
        nme_v = nme(pred_orig, pva)
        pe = per_face_errors(pred_orig, pva)
        results.append({
            "alpha": alpha,
            "val_NME_mean": float(nme_v.mean()),
            "val_NME_median": float(np.median(nme_v)),
            "val_mean_px_err": float(pe.mean()),
            "val_median_px_err": float(np.median(pe)),
            "fit_s": dt,
        })
        print(f"  alpha={alpha:>7g}  NME={nme_v.mean():.4f}  "
              f"px={pe.mean():.2f}  ({dt:.1f}s)")
        if nme_v.mean() < best[1]:
            best = (model, nme_v.mean(), alpha)

    print(f"\nbest alpha: {best[2]}  val NME: {best[1]:.4f}")
    final = best[0]
    pred_va = final.predict(Fva).reshape(-1, 5, 2) * (256 / IMG_SIZE)
    np.save(OUT / "pred_val_M1.npy", pred_va)
    np.save(OUT / "feat_test_M1.npy", Fte)  # cache for later

    # Predict test for submission
    pred_te = final.predict(Fte).reshape(-1, 5, 2) * (256 / IMG_SIZE)
    np.save(OUT / "pred_test_M1.npy", pred_te)

    summary = {
        "model": "M1_HOG_ridge",
        "img_size": IMG_SIZE,
        "hog_params": {k: v for k, v in HOG_PARAMS.items() if k != "feature_vector"},
        "hog_dim": int(Ftr.shape[1]),
        "best_alpha": best[2],
        "best_val_NME": float(best[1]),
        "sweep": results,
    }
    (OUT / "metrics_M1.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {OUT / 'metrics_M1.json'}")


if __name__ == "__main__":
    main()
