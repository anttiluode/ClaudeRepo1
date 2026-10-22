# ClaudeRepo1 — which configurations across the Sihti bands occur together?

> **New page:** [`world.html`](world.html) — *The world that learns what it cannot see*, Sol's proposed demo.
> Two learners live in two worlds built from the same nine pieces. Every piece and every pair is equally
> frequent in both; only a three-way relation differs (world A: matching eyes come with a smile; world B: with a
> frown). Nobody tells them which pieces belong together. Then both are shown the same face with the mouth covered.
> See [The world that learns](#the-world-that-learns-worldhtml) below.

> **Live toy:** [`index.html`](index.html) — *Living Paper*, the Model 5 idea in the browser: a webcam
> view with an occluder, filled from a motion-carried memory that is checked against the present before
> it is believed. Works with GitHub Pages (Settings → Pages → main, root). Simplified: smoothness prior
> instead of Model 5's patch mixture, global translation only. Core logic tested with
> `node tests/test_index_core.mjs` (memory fill 34× lower error than the plain fill on a synthetic pan;
> foreign memory rejected, trust 0.00).

> **Astra branch audit (22 October):** A2b's claim of two fresh source photographs
> is withdrawn: `cat` aliases the training image `chelsea`. Original receipts below
> are preserved, not rerun. The histogram/spectrum controls are approximate, not
> exact; their remaining marginal differences are predictive. See
> [the bounded audit and experience result](docs/ASTRA_RESULT.md).
> The new synthetic experience gate passes, **with ordinary ridge tying it**.
> No persistence network was built.


Gate A of the plan from the Oct 22 2026 Antti / Astra / Sol / Claude discussion: before building a
fast/slow persistence machine (Gate B), check whether a Hebbian relation layer over the Sihti bands
carries anything a frequency ledger cannot see. Everything was preregistered and pushed before the
test runs ([docs/PREREG_A.md](docs/PREREG_A.md), then [docs/PREREG_A2b.md](docs/PREREG_A2b.md)).

**Short version:** the structure is there and it is exactly where Freeman et al. said it would be, in
how *rectified* responses co-occur across scale and space. The Hebbian learning, on this test, adds
nothing: an untrained all-ones weight vector does as well.

## The setup

* **Sieve** (`sieve.py`): linear diffusion on a torus, bands r_k = x_{d_k} − x_{d_{k+1}} at depths
  0,1,2,4,8,16,32; lossless. Here, spatially averaged signed band correlations on a torus are fixed by the power spectrum.
* **Relation layer** (`relations.py`): per image, standardised activities of each band, either
  rectified |r_k| or signed r_k; the relation tensor R[k,k',δ] = mean_p a_k(p)·a_k'(p+δ) over all band
  pairs and 9 offsets (309 relations). Hebbian learning W ← W + (R − W)/n over natural training crops only.
* **Controls**, made from each test crop: `phase` (random phases, identical spectrum — Freeman's
  control), `marginal` (Heeger–Bergen: band histograms and spectrum re-imposed alternately),
  `shuffle` (16×16 blocks permuted; seams add new edges, so report-only).
* Training: 560 crops from 15 scikit-image photographs. Test: 150 crops from `camera`, `coffee`,
  `clock`; the original A2b gate on 100 crops from `cat`, `cell` (source overlap: `cat` is training `chelsea`).

![learned links](results/natural/learned_links.png)

The learned co-activations are readable. The same place lights up across neighbouring scales (0.80–0.87
between adjacent mid bands, falling to 0.13 between finest and coarsest), and laterally the coarser
bands reach further.

## Results

| gate | claim | result | |
|---|---|---|---|
| **A1** | rectified relations see what the spectrum cannot | typicality AUC real vs phase-scrambled **0.87**; linear relations **0.50** exactly (they are identical by construction) | **PASS** |
| **A2** | relations, beyond per-band shape (band-histogram-matched controls) | typicality AUC **0.40** (below chance) vs per-band shape 0.42 | **FAIL** |
| **A2b** | same, with the Hebbian unit's *drive* ⟨R, W⟩ as the readout, fresh photographs | **0.78** vs per-band shape 0.56; phase **0.998** | **PASS** |
| *post-hoc* | does the learning matter? | drive with W = all ones (no experience): **0.79**; learned W with entries shuffled: **0.79**; learned W after only 2 crops: **0.78** | **no** |

![links broken by each control](results/natural/broken_links.png)

## What happened, honestly

1. **A1 is an analogous computational statistic, not a replication of V2 physiology.** Phase scrambling leaves every band's energy and every
   signed cross-band correlation identical (a test checks this to 10⁻⁹), and the linear relation
   layer scores exactly chance. The rectified layer sees the difference.
2. **A2 failed because of the readout, not the features.** The preregistered score was typicality:
   distance from the average learned co-activation. Each real photograph is atypical in its own way,
   while scrambles regress toward the average, so controls scored as *more* natural. A post-hoc
   supervised probe (`diagnose_a.py`) showed the information was there: 0.97 for rectified relations
   vs 0.87 for per-band shape on the matched controls.
3. **A2b changed the readout and met its numerical threshold, but its fresh-source claim failed the later audit.** A Hebbian unit does
   not measure distance to its weights; it responds with its drive, the overlap of input and weights.
4. **Then the deflation.** Every learned co-activation is positive (0.009 to 0.87), so the controls show little benefit from the arrangement of W on this test (positivity alone would not establish that). The drive is essentially *how much rectified activity coincides across scale
   and nearby space*, and that is what tells a photograph from its scrambles: edges line up across
   scales in real images, and in scrambles they do not. That statistic is known (the cross-scale
   magnitude dependencies of Simoncelli and colleagues; the Portilla–Simoncelli texture statistics
   Freeman used). The rectification is what matters. The learning is not, on this test.

![learning curve](results/natural/learning_curve_drive.png)

## What this means for Gate B

The preregistered condition for building Gate B (A2b passes) is met, but building it now would test
the wrong thing. A slow recurrent state supported by this W would be supported by "any coincident
rectified activity". That is a coherence amplifier, and it needs no experience. Learning only
becomes necessary when the task needs **specific** relations: telling a bird from a rearrangement of
bird pieces (Rust & DiCarlo's selectivity), or one learned category from another. Gate A's shuffle
control was the nearest thing to that, and untrained weights did as well there too (0.71 vs 0.68).

So the next honest step is a selectivity gate before Gate B: relations learned from one kind of
image must tell that kind from another *natural* kind that has the same "edges align across
scales" property, and must beat an untrained W at doing so.

## Claude's review of the Astra branch (merged into main)

* Verified: `np.array_equal(skimage.data.cat(), skimage.data.chelsea())` is `True`. A2b's "fresh" set was
  half a training photograph; that claim is withdrawn, as Astra wrote. My error.
* Reproduced: all 9 tests pass, and `python experience_gate.py` regenerates `results/astra/experience.json`
  byte-for-byte (gate passes; contrastive rule 100% paired, mean-Hebb and all-ones 0%, ridge 100%).
* Partial corrected-source receipt: `eagle` cannot be downloaded in my sandbox, so I ran `cell` alone
  (100 crops; not the full preregistered cell+eagle set). Rectified drive vs band-matched controls
  **0.99**, per-band shape **0.67**; no-learning control: learned 0.993, all ones 0.976, shuffled 0.980
  (`results/natural/*cell_only_partial.json`). Same picture as before: the rectified statistic carries
  it, learning adds almost nothing. Run `pip install pooch` then `python gate_a2b.py` on a machine with
  internet for the full corrected run.
* What the experience gate adds: mean-Hebb and all-ones get world A (orientations agree) right and
  world B (orientations oppose) wrong, every time. Generic coherence *is* a prior: "aligned edges go
  together". In the tested associative implementations, subtracting the independently recombined
  baseline lets the rule learn against that prior; this is not necessary for all learners (ridge ties it).
  That is the computational
  shape of Li & DiCarlo's swap experiment: altered experience overriding the default association.
  Ridge ties it, so this shows learned association is *possible* in these features, not that it is better.

## Three-piece follow-up (Astra)

The [preregistered three-piece gate](docs/THREE_PIECE_PROTOCOL.md) passed:
pairwise Hebb and linear ridge scored 50%; a supplied three-way feature with
learned weights scored 100% in both opposite worlds. Ridge given a product
feature also scored 100%. The previous whole-image Sihti descriptor scored
49.375%. **The useful conjunction was supplied, not discovered.**
See [the result and limitations](docs/THREE_PIECE_RESULT.md) and
[the full receipt](results/astra/three_piece.json). Gate B remains unbuilt.

## The world that learns (`world.html`)

Sol's proposed demo, after the three-piece result: show the same incomplete input producing different answers
after different experience, show the internal relationships forming, then erase them and show the ability go.
Astra's three-piece gate said the conjunction had to be *supplied*; this page asks whether a learner can find
it itself, from pixels.

**The worlds.** A 3×3 grid of 12×12-pixel pieces; each piece shows one of two glyphs, with jitter and pixel
noise. Two eyes, a mouth, six clutter pieces. In both worlds every piece and every pair of pieces is independent
and 50/50 (a test checks all 36 pairs); only the product eye × eye × mouth differs, +1 in A and −1 in B.

**The learner.** 24 branches, each wired at random to 2 or 3 pieces, each with up to 6 codewords of the joint
look of its pieces (competitive learning: only the closest codeword moves; a new codeword only for an experience
that is new in at least one piece). Before learning from an experience, each branch hides one of its own pieces,
predicts it from the others, and checks. Branches that stay at chance are rewired; branches that predict reliably
are kept. Completion: every kept branch touching a covered piece weighs its codewords against the visible pieces
and any peeked pixels; its say is reduced when its best explanation fits badly or a peek contradicts it.

**Receipt** (`node world_receipt.mjs` → [results/world/receipt.json](results/world/receipt.json); 10 seeds per
world, 3,000 experiences per learner, 400 fresh test scenes each, a grey answer scored as a coin flip):

| | covered mouth right |
|---|---|
| learner, own world (20 learners) | **100%** (every learner) |
| same experience, delta-rule linear map (pairs only) | 49.9% |
| learned links cut (codewords kept, re-paired) | 46% mean (range 9–100%) |
| erased | 50% |
| mouth never in its experience | 50% |

* All 20 learners found the eye–eye–mouth wiring by themselves, after 62 to 1,121 experiences (median 250).
  No learner kept any other wiring. The kept branch's read-out rule is exactly +1.00 (A) or −1.00 (B) for the
  three-way product, and 0.00 for every pair inside it.
* `node tests/test_world_core.mjs` (about 40 s) also checks: the same face gets a smile from A and a frown from
  B; a peek that contradicts B drops its certainty below 0.3 while the same peek keeps A above 0.9; with an
  eye and the mouth both covered, certainty starts low and rises with a mouth peek, and the hidden eye comes out
  round for A and flat for B; a scene with a noise eye gets no support; and the linear control does learn a
  world where the mouth simply copies one eye (99%), so its 50% here is a real failure, not a broken control.

**What this does not show.** It is a toy built so that only the three-way relation carries the answer: nine
pieces in fixed places, two glyphs each, no translation, a hand-set pixel noise level (measured on the world:
same glyph ≤ 0.031, different glyph ≥ 0.052 per-pixel squared distance). Finding the wiring is a random search
over 84 triples (plus pairs), judged by self-prediction; with real images the number of possible groupings
explodes, and that search cost is the open problem. An early version with whole-scene competing units (no
self-test) found the conjunction only when initialisation was lucky (K=4: 65–100% across seeds), which is why
the scarcity result from the scratch check was not used here. The parts are known: competitive learning
(Rumelhart & Zipser 1985), Barlow's suspicious coincidences, masked self-prediction, synaptic turnover,
mixture-model completion.

## The relation inspector (`inspector/`) — leaving the toy world

The same mechanism as `world.html`, moved onto real images and a public benchmark. Industrial inspection's
strongest simple method, PatchCore, is a memory of good patches: in energy terms, unary terms only. It
catches scratches and dents but is weak on MVTec LOCO AD's *logical* anomalies, where every part looks
normal and the combination is wrong (published mean logical AUROC 74.0, against 86–92 for methods built
for them). The inspector adds learned relations between part types on top of PatchCore:

* **part types**: k-means on the backbone's patch features from good images;
* **readable variables**: how much of each type lies in each 3×3 region, how many pieces of it there are;
* **branches**: one variable predicted from one or two others, searched by leave-one-out on good training
  images and kept only if they also predict held-out good images;
* **explanations**: the branch that is most surprised names the problem, e.g. *given part type 3 in the
  left = 0.21: part type 7 in the right should be about 0.18, it is 0.02*.

Gates L1, L2 and E1 are preregistered in [docs/PREREG_INSPECTOR.md](docs/PREREG_INSPECTOR.md), written
before any LOCO run. To run on a GPU machine with the dataset downloaded from MVTec:

```bash
pip install torch torchvision scipy scikit-learn pillow
python -m inspector.run --root /path/to/mvtec_loco            # about 10-20 min on an RTX 3060
python -m inspector.make_synthetic synth && python -m inspector.run --root synth --categories tray   # smoke test
```

Results land in `results/inspector/` (per-category JSON with AUROCs, kept relations, and explanations of
the most anomalous test images, plus `summary.json`).

## Run it

```bash
pip install -r requirements.txt pytest
python -m pytest -q          # sieve, controls, source isolation, experience and parity checks
node tests/test_world_core.mjs   # world.html's learner and controls (about 40 s)
node world_receipt.mjs           # the 10-seed receipt in results/world/receipt.json (about 4 min)
python three_piece_gate.py   # bounded synthetic three-way relation test; no downloads
python gate_a.py             # Gate A (about a minute)
python diagnose_a.py         # post-hoc probe
python gate_a2b.py           # A2b; gate_a2b.curve_figure() and no_learning_control() for the rest
python gate_a.py --folders ~/THINGS/unseen --name folders   # test crops from your own photos
```

## Prior art

Freeman, Ziemba, Heeger, Simoncelli & Movshon 2013 (V2 and naturalistic texture statistics);
Portilla & Simoncelli 2000; Heeger & Bergen 1995; Simoncelli 1997 / Buccigrossi & Simoncelli 1999
(cross-scale magnitude dependencies); Rust & DiCarlo 2010 (selectivity and tolerance);
Freeman & Simoncelli 2011 (metamers of the ventral stream).
