# Gate A preregistration — does a Hebbian relation layer on rectified Sihti bands carry anything?

Committed before `gate_a.py` is run on any test image. There are no tuned parameters: sieve depths
(0,1,2,4,8,16,32), diffusion step 0.2, offsets ({0} and 4 directions x {2, 6} px) and crop settings are
fixed in the code as written.

## Question

From the Oct 22 Astra/Sol/Antti discussion: *which configurations across the bands tend to occur
together, and what do they support?* Freeman et al. (2013) found V2, unlike V1, responds more to
textures with natural higher-order structure than to spectrally matched noise. On a torus the Sihti
sieve is a convolution, so every **linear** relation between bands (energy, signed correlation at any
offset) is fixed by the power spectrum. A **rectified** relation (co-activation of |r_k| and |r_k'| at
offset delta) is fourth-order and can see more.

## Setup

* Training: 40 random 64x64 crops (scales 0.35/0.5/0.75, std >= 0.05) from each of 15 scikit-image
  photographs = 600 crops. Test: 50 crops from each of `camera`, `coffee`, `clock` (never in training).
* Three Hebbian layers, each trained only on natural crops, never on a control:
  `rectified` (relations of |r|), `linear` (relations of r), `marginal` (per-band kurtosis, mean |z|,
  mean |z|^3 — no relations at all). Update W <- W + (R - W)/n. Score of an image = minus the mean
  squared z-deviation of its features from W.
* Controls for every test crop: `phase` (random phases, identical spectrum), `marginal`
  (Heeger–Bergen-style: band histograms and spectrum alternately re-imposed, 8 iterations),
  `shuffle` (16x16 blocks permuted).
* Metric: AUC of score, real vs control.

## Gates

| gate | claim | pass if |
|---|---|---|
| **A1** | rectified relations see what the spectrum cannot (Freeman's test) | `rectified` AUC on `phase` >= 0.80 **and** `linear` AUC on `phase` in [0.40, 0.60] |
| **A2** | relations, not just per-band shape | on `marginal`: `rectified` AUC >= 0.70 **and** >= `marginal`-features AUC + 0.10 |
| A3 | arrangement (report only) | AUCs on `shuffle` reported; block seams add new edges, so a high AUC may be seam detection, not configuration |

## Said in advance

A1 linear-control is a mathematical certainty up to floating point (tested in `tests/`); it is in the
gate as an instrument check. Phase scrambling also Gaussianises each band, so per-band shape alone will
separate `phase` — that is why A2 exists. The band-histogram matching does **not** converge exactly
with this sieve (bands are not orthogonal projections); development on one camera crop left band
kurtosis 10–30% below the real image's. A2's relative criterion is there for that reason.

Gate B (fast/slow persistence) is built only if A1 and A2 pass.
