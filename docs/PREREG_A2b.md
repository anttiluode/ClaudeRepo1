# Gate A2b preregistration — a different readout, tested on photographs not used so far

Written **after** Gate A2 failed (see README) and after the post-hoc diagnosis in
`diagnose_a.py`, **before** the readout below has been computed on any image.

## Why

A2 used a typicality readout: how close an image's co-activations are to the *average* learned ones.
The post-hoc diagnosis showed the information is in the rectified relations (a supervised probe tells
real from band-matched controls at 0.97 vs 0.87 for per-band shape), but the typicality readout
rewards controls for being average: each real image is atypical in its own way.

A Hebbian unit does not compute a distance to its weights; it computes its **drive**, the overlap of
its input with its weights. So the readout here is

    drive(x) = < R(x), W > / ||W||

over the rectified relations, leaving out the same-band zero-offset entries (identically 1 after
standardisation). The comparison readout is the same overlap for the per-band shape features
(drive of the marginal vector onto its learned mean). W is the Gate A layer, unchanged
(trained on the same 15 photographs).

## Fresh test data

50 crops each from `cat` and `cell` (scikit-image, never used in Gate A), seed 22, same crop rules.
Controls as in Gate A. The original Gate A test set is reported too, labelled as reused.

## Gate

| gate | pass if (fresh photographs) |
|---|---|
| **A2b** | rectified drive AUC on `marginal` controls >= 0.70 **and** >= per-band-shape drive AUC + 0.10, **and** rectified drive AUC on `phase` >= 0.80 |

`shuffle` reported only (seams). Gate B is built only if A2b passes; otherwise Gate A is closed as
"the information is there, an unsupervised Hebbian readout did not use it".
