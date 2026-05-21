"""
Task 2 — Produce the test-set predictions CSV.

The chosen model is M2 (cascaded HOG-ridge, 3 stages). Its test
predictions have already been written to outputs/pred_test_M2.npy by
04_cascaded.py. Here we just save them in the format required by the
worksheet's save_as_csv helper.

Format: comma-separated, shape (554, 10) — each row is
x0,y0,x1,y1,x2,y2,x3,y3,x4,y4.
"""


from pathlib import Path

import numpy as np

from common import OUT


def save_as_csv(points: np.ndarray, location: str = "."):
    """Mirror of the worksheet's helper, with the assertions."""
    location = Path(location)
    assert points.shape[0] == 554, "wrong number of image points, should be 554 test images"
    assert np.prod(points.shape[1:]) == 5 * 2, "wrong number of points provided. There should be 5 points with 2 values (x,y) per point"
    np.savetxt(location / "results_task2.csv",
               np.reshape(points, (points.shape[0], -1)),
               delimiter=",")


def main():
    pred = np.load(OUT / "pred_test_M2.npy")
    print("pred_test_M2 shape:", pred.shape)
    save_as_csv(pred, location=str(OUT))
    out = OUT / "results_task2.csv"
    print(f"Wrote {out}")
    # quick sanity print
    arr = np.loadtxt(out, delimiter=",")
    print("loaded back shape:", arr.shape)
    print("first row:", arr[0])


if __name__ == "__main__":
    main()
