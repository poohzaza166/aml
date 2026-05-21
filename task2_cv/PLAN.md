# Task 2 — Face alignment: planned approach

**Status:** blocked on missing data files
(`face_alignment_{training,validation,test}_data.npz`). The plan below
is what we'll run the moment they're in the project folder.

## What the task is

Predict the (x, y) image-pixel coordinates of K facial landmarks from
an image of a face. This is a **multi-output regression** problem (the
outputs are continuous coordinates), not classification. Loss is
typically L1 or L2 on coordinates (after appropriate normalisation).

There's an explicit robustness component: the data contains
"transformations that distort the images". We should build in invariance
to nuisance variability (rotation, scale, illumination, possibly
masking) by either pre-processing (image normalisation, face detection
& cropping) or data augmentation at training time, or both.

## Planned approaches (comparing two families, as the brief asks)

### Approach A: Classical — hand-crafted features + linear regression

A direct lecture-week-8/9 implementation:

1. **Pre-processing**: convert to greyscale, resize to a fixed canvas
   (e.g. 128×128), histogram equalise to standardise contrast.
2. **Feature representation**: HOG descriptor of the whole image
   (Dalal & Triggs 2005). Tunable: cell size, block size, orientations.
   Optional: a second branch with SIFT-on-patches.
3. **Regression**: ridge regression (closed-form) or linear regression
   with L2 regularisation, one head per landmark × {x,y}.
4. **Augmentation at training time**: random small rotations, scales,
   horizontal flips with correct landmark re-indexing, brightness jitter.

Expected ballpark: this is the classical baseline; sensitive to head
pose & occlusion. Reference: Cootes, Edwards & Taylor (2001) for the
Active Shape Model lineage; HOG-LR is a simpler cousin.

### Approach B: Cascaded shape regression (Kazemi & Sullivan 2014)

The dlib-style approach: start from the *mean shape* and iteratively
update it using regression on shape-indexed pixel features. Two
options:

- **B1** Implement a 2-stage cascade ourselves on top of A (small
  rotation/scale invariance baked in).
- **B2** Use dlib's `shape_predictor` (which is trained externally on
  iBUG-300W); evaluate as a *strong baseline* the way we'd compare a
  pre-trained ResNet to a from-scratch CNN.

### Approach C (stretch): tiny CNN

If time allows, a small CNN (4–6 conv layers, output 2K floats) trained
with L1 loss and the same augmentations. PyTorch on CPU is tight but
should work for 128×128 input. This addresses the "use modern methods"
sub-current.

## Evaluation plan

- Primary metric: **mean per-landmark Euclidean error**, normalised
  by inter-ocular distance (the standard "NME" — Normalised Mean
  Error — for the 300W competition).
- Secondary: **cumulative error distribution (CED)** curve and a
  boxplot per method.
- Robustness study (10 marks): re-evaluate the chosen model under
  increasing Gaussian noise, random rotation, contrast scaling. Plot
  NME vs perturbation magnitude.

## Compute notes

CPU only is fine for HOG + ridge. CNN on CPU is OK for tiny networks
trained for a few epochs.

## References

- Cootes, Edwards & Taylor (2001) *Active Appearance Models*, IEEE TPAMI.
- Dalal & Triggs (2005) *Histograms of Oriented Gradients for Human
  Detection*, CVPR.
- Kazemi & Sullivan (2014) *One Millisecond Face Alignment with an
  Ensemble of Regression Trees*, CVPR.
- Sagonas et al. (2013) *300 Faces in-the-Wild Challenge*, ICCV WS.
- Wu et al. (2018) *Look at Boundary: A Boundary-Aware Face Alignment
  Algorithm*, CVPR. (Useful for the robustness discussion.)
