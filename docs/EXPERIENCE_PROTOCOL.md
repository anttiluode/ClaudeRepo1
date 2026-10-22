# Bounded experience gate — Astra, 22 October 2026

Written before running the new experiment. Scope: audit the existing controls and
one CPU-only synthetic association test; no persistence network or image model.

## Question
Does changing training experience change the prediction from the SAME partial
image in the appropriate direction? This is a mechanism check, not object learning.

Use two 16x16 diagonal-line primitives in known slots of a 64x64 image. Environment
A pairs equal orientations; B pairs opposite orientations. Both environments have
the same balanced primitive frequencies and nuisance distribution. Test images
show only the left slot; two fixed, hand-designed right-slot hypotheses are supplied
to every method. The missing target never enters feature extraction. The scaffold
supplies slots, candidate primitives and their separation, not the learned pairing.

Reuse rectified Sihti relation features, adding a displacement of 24 pixels to span
the known slots. A contrastive covariance rule averages observed pair relations
minus relations with the right pieces recombined independently. The null uses the
average of both possible right orientations with the same nuisance parameters.
There is no backpropagation, temporal persistence or biological claim. Compare:
original mean Hebb drive, all-ones weights, shuffled learned weights, learning from
randomly reassigned partners, and ordinary pixel ridge regression.

Fixed budget: five seeds 101..105, 64 training pairs per environment, 32 test cues
per seed, new nuisance draws. No tuning after test. Report per-environment accuracy,
paired success (both environments correct for a cue), flip rate, and each seed.
Gate: contrastive learner >=90% accuracy in EACH environment and >=80% paired
success, and >=20 percentage-point paired advantage over all-ones, shuffled and
randomly reassigned-partner controls. Report failures without adjusting the gate.
Ridge is a reference, not a required loser. Passing establishes learned association
in this constructed task, not natural-image generalisation, learned features,
novelty, or the right to build Gate B automatically.

## Audit
Reject source-identity overlap including cat/chelsea. Keep original result files
unchanged and mark A2b's freshness claim withdrawn. Replace the fresh default with
cell and eagle; guard it against training-source overlap. Do not rerun the entire
photograph benchmark in this budget. Audit final control images AFTER reanalysis:
relative Fourier-amplitude error and per-band sorted-coefficient error. These are
measurements, not a claim that alternating projections achieves exact matching.
Use deterministic synthetic images for a small offline receipt; the audit CLI may
also take a folder. No claims about photograph mismatch magnitudes from that receipt.
