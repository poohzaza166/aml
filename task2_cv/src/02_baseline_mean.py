"""
Task 2 — Baseline M0: predict the training mean shape for every image.

Why include this baseline?
  - It's the *trivial* floor of any face-alignment system: a model
    that ignores the image entirely. Any non-trivial model has to beat
    it.
  - Reports two metrics: raw mean Euclidean error in pixels and NME
    normalised by inter-ocular distance (the standard 300W metric).
"""


import json

import numpy as np

from common import (
    OUT,
    inter_ocular_distance,
    load_splits,
    nme,
    per_face_errors,
)


def main() -> None:
    Xtr, ptr, Xva, pva, _ = load_splits()
    mean_shape = ptr.mean(axis=0)  # (5, 2)
    np.save(OUT / "mean_shape.npy", mean_shape)

    pred_val = np.broadcast_to(mean_shape, pva.shape).copy()
    pf = per_face_errors(pred_val, pva)
    iod = inter_ocular_distance(pva)
    nme_v = nme(pred_val, pva)

    metrics = {
        "model": "M0_mean_shape",
        "val_mean_euclid_px_mean": float(pf.mean()),
        "val_mean_euclid_px_median": float(np.median(pf)),
        "val_NME_mean": float(nme_v.mean()),
        "val_NME_median": float(np.median(nme_v)),
        "val_IOD_mean_px": float(iod.mean()),
    }
    print(json.dumps(metrics, indent=2))

    np.save(OUT / "pred_val_M0.npy", pred_val)
    (OUT / "metrics_M0.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
