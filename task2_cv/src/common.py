"""Shared utilities for Task 2 face alignment.

Why each design choice here:
  * Greyscale + histogram equalisation → standardises lighting so that
    later feature descriptors are not dominated by absolute brightness.
  * Resize to 128×128 for HOG → keeps HOG dimensionality tractable
    (~7k features at 8-px cells) while losing little useful detail at
    256×256.
  * Landmark error metric: per-landmark Euclidean (provided
    `euclid_dist`) and per-face mean Euclidean. We also report
    "Normalised Mean Error" (NME) normalised by the inter-ocular
    distance from the ground-truth shape — the standard 300W metric.
"""


from pathlib import Path

import cv2
import numpy as np

TASK = Path(__file__).resolve().parents[1]
DATA = TASK
OUT = TASK / "outputs"
FIG = TASK / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


# ---------- data --------------------------------------------------------


def load_splits():
    tr = np.load(DATA / "face_alignment_training_data.npz", allow_pickle=True)
    va = np.load(DATA / "face_alignment_validation_data.npz", allow_pickle=True)
    te = np.load(DATA / "face_alignment_test_data.npz", allow_pickle=True)
    return tr["images"], tr["points"], va["images"], va["points"], te["images"]


# ---------- pre-processing ----------------------------------------------


def to_grey_eq(img: np.ndarray) -> np.ndarray:
    """RGB uint8 → equalised greyscale uint8."""
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        g = img
    return cv2.equalizeHist(g)


def preprocess_stack(images: np.ndarray, out_size: int = 128) -> np.ndarray:
    """Apply greyscale + equalise + resize to a batch of images."""
    n = images.shape[0]
    out = np.empty((n, out_size, out_size), dtype=np.uint8)
    for i, im in enumerate(images):
        g = to_grey_eq(im)
        if g.shape[0] != out_size:
            g = cv2.resize(g, (out_size, out_size), interpolation=cv2.INTER_AREA)
        out[i] = g
    return out


def scale_points(points: np.ndarray, src_size: int, dst_size: int) -> np.ndarray:
    return points * (dst_size / src_size)


# ---------- error metrics ----------------------------------------------


def euclid_dist(pred_pts: np.ndarray, gt_pts: np.ndarray) -> np.ndarray:
    """Provided in the worksheet — Euclidean distance per landmark.
    Inputs are (n, p, 2) or (p, 2)."""
    pred_pts = np.reshape(pred_pts, (-1, 2))
    gt_pts = np.reshape(gt_pts, (-1, 2))
    return np.sqrt(np.sum(np.square(pred_pts - gt_pts), axis=-1))


def per_face_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Mean Euclidean distance per face. Inputs (n, p, 2). Returns (n,)."""
    return np.mean(np.sqrt(np.sum((pred - gt) ** 2, axis=-1)), axis=-1)


def inter_ocular_distance(points: np.ndarray) -> np.ndarray:
    """|p0 - p1| per face — landmarks 0 (left eye) and 1 (right eye)."""
    return np.sqrt(np.sum((points[:, 0] - points[:, 1]) ** 2, axis=-1))


def nme(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Normalised Mean Error per face (300W convention).

    NME_i = mean_k |pred_ik - gt_ik| / d_eye_i
    """
    iod = inter_ocular_distance(gt)
    pf = per_face_errors(pred, gt)
    return pf / np.clip(iod, 1e-6, None)
