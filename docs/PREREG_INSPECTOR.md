# Preregistration: the relation inspector on MVTec LOCO AD

Written 22 October 2026, before any run on MVTec LOCO AD. Code: `inspector/` at the commit that adds
this file. Nothing below may be tuned on the LOCO test split. If a setting changes after the first real
run, the change and its reason are recorded here and the run is labelled post hoc.

## Question

Does adding learned, self-tested relations between part types to a unary patch memory (PatchCore) catch
logical anomalies that the unary memory misses, and do its explanations point at the right place?

## Fixed settings

* Images resized to 256×256. Backbone: ImageNet WideResNet-50-2 (torchvision), layer2+layer3, PatchCore
  aggregation, 1024 dims, 32×32 patches. PatchCore coreset 10%, nearest-neighbour patch score, image score = max.
* Relations: K = 12 part types (k-means on good training patches); variables = area of each type in each
  of 3×3 regions plus number of pieces of each type (blobs ≥ 3 patches); 5-nearest-neighbour prediction;
  candidates searched by leave-one-out R² on (up to 200) training images: per variable the best one- or
  two-variable predictor, plus 3000 random triples kept only if R² beats the best single predictor by
  ≥ 0.2; a candidate is kept only if its R² is also ≥ 0.5 on held-out images. Validation/good is split in
  two: even images are that held-out self-test, odd images set error scales and the score normalisation. Relation score = largest branch or variable surprise.
* Combined score = PatchCore score + relation score, each standardised on validation/good.
* Seed 0. All five categories.

## Gates (image AUROC, mean over the five categories, measured in one run)

| gate | claim | pass if |
|---|---|---|
| **L1** | relations see logical anomalies | relations alone, logical AUROC ≥ PatchCore's logical AUROC + 5 points |
| **L2** | adding them helps without breaking the rest | combined logical AUROC ≥ PatchCore + 8 points, and combined structural AUROC ≥ PatchCore structural − 2 points |
| **E1** | the explanation points at the problem | on logical anomalies with masks, the flagged variable's location overlaps the ground truth more often than a random variable's location does, by ≥ 15 points |

If L1 fails, the relation term is not doing its job on real data and the line stops here.

## Reference numbers (published, not gates)

Logical-anomaly image AUROC, mean of five categories, from the LogicAD paper's table (Jin et al. 2025):
PatchCore 74.0, GCAD 86.0, ComAD 87.7, LogicAD 86.0 (one-shot, VLM-based). PUAD (2024) reports 92.0
for logical anomalies with EfficientAD plus a global Mahalanobis term. Passing the gates above would make
the inspector worth developing; it would not by itself make it competitive with these methods, and the
write-up must say where it stands against them.

## Changes made before the first LOCO run (from the synthetic smoke test, 22 October)

Recorded here because they were made after this file was first pushed (c7c8c86); all were made on
`inspector/make_synthetic.py` data only, before any LOCO image was seen.

1. **No arithmetic relations.** Within one region the K type areas sum to 1, so an area may not be
   predicted from other areas in its own region. The first smoke run's top 'relations' were all of this kind.
2. **Surprise = distance to the closest allowed value**, not to the neighbours' mean. A branch now asks
   whether any of the 10 good images most similar in its predictors has a target value near the observed
   one. Where good images allow two values, their average is neither, and scoring against it made good
   images look anomalous.
3. **Conservative error scale.** Each branch's scale is the larger of its held-out (calibration-half)
   value and its leave-one-out value over all training images, floor 0.25.
4. **Combination stays the sum** of the two standardised scores (max was also tried on the toy and kept
   structure worse: 0.91 against 0.94).

Smoke result (synthetic tray, colour-statistic features via `--pixels`, not evidence about LOCO; receipt
in `results/inspector_smoke/tray_pixels.json`): logical AUROC PatchCore 0.48, relations 0.85, sum 0.84;
structural PatchCore 0.995, sum 0.94 (that would fail L2 on structure); explanations overlap the truth
0.60 vs 0.30 for a random variable. The gates above are unchanged.

## What the inspector does not claim

Part types are k-means clusters, not named objects; a person reads them from example patches. Branch
search is greedy plus random, not exhaustive. The structural detector is plain PatchCore.
