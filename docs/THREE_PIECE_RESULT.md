# Three-piece result: interaction order matters; learning it remains open

The fixed gate passed. A supplied three-way feature plus an experience-dependent
weight solves the task. Pairwise learning does not. Ordinary regression given
the corresponding product feature solves it too. This does **not** show an
advantage over ordinary nonlinear learning.

Protocol committed before execution: `eebfeadf47b631415102a40394c201adb5d18645`.
Run `python three_piece_gate.py`; receipt: `results/astra/three_piece.json`.
No downloads, natural-image training, persistence implementation or parameter sweep.

## The question

There are three small patches, each containing one of two diagonal lines.
Two are visible. In world A the hidden orientation is the product of the visible
orientation signs; in world B it is the opposite. Thus the same visible image
requires opposite completions after different experience.

Each world has identical orientation counts and identical counts for every pair
of orientations. Only the three-way relation distinguishes the worlds. Training
uses all four visible combinations at each nuisance setting; testing uses new
amplitudes, widths, shifts and pixel noise. It is nuisance generalization within
a known truth table, not generalization to unobserved logical combinations.

Each patch passes through Sihti separately. A fixed template contrast reads its
orientation from rectified bands, giving continuous evidence z. Sign extraction
was correct on all test patches. The learned scores are:

* Pairwise: weights on z1*z2, z1*z3, z2*z3.
* Three-way: one weight on z1*z2*z3.

Each weight averages observed features minus the features with the target
independently recombined. At inference both possible hidden patches are scored;
their pixels are supplied templates, never the actual hidden pixels.

## Fixed-budget results

Five seeds, 64 training examples per world per seed, 32 test cues per seed.
Percentages aggregate 160 cues evaluated under both learned worlds.

| Method | World A | World B | Both worlds right for same cue |
|---|---:|---:|---:|
| Pairwise contrastive Hebb | 50% | 50% | 0% |
| Three-way contrastive Hebb | 100% | 100% | 100% |
| Fixed positive three-way weight | 100% | 0% | 0% |
| Targets reassigned before learning | 60% | 60% | 60% |
| Learned weights swapped between worlds | 0% | 0% | 0% |
| Linear ridge on two visible evidences | 50% | 50% | 0% |
| Ridge with visible evidence product added | 100% | 100% | 100% |
| Previous whole-image Sihti relations | 49.375% | 49.375% | 49.375% |

Pairwise and linear models tie on their candidate scores and choose the same
default in both worlds; that explains their zero paired correctness. A model
that flips its answers between worlds can have paired correctness equal to its
individual-world accuracy, even when near chance.

The reassignment control is particularly small: one scalar's sign chooses the
entire rule, and random reassignment happened to choose correctly in three of
five runs. These are **five learned rules**, not 160 independent demonstrations
that reassignment has 60% population accuracy. No confidence or significance
claim is made from this small run. The pass uses the predeclared descriptive
thresholds, not a statistical significance test.

## What changes in our interpretation

This is a representation result. A linear scorer on an explicitly supplied
three-way feature is already nonlinear in its inputs. Giving that scorer a
product is much of the solution. Here the learner only acquires the sign and
magnitude of one coefficient. It does not discover the patches, their orientation
readout, the grouping, the multiplication, or which order of interaction to use.

The previous global descriptor was also tested unchanged as a diagnostic. Its
rectification can mix signals before its pairwise statistics are measured, so
there is no general theorem that it must fail this task. It happened to perform
near chance in this rendering and fixed budget. That is narrower than proving
that all pairwise statistics of nonlinear features cannot represent parity.

One correction to the earlier README review: subtracting expected coactivation
is **not the only way** to learn against a default prior. These experiments
compare specific implementations. Ridge already supplied a counterexample;
product ridge now supplies another. Centering helps our chosen associative rule,
but the results do not establish its necessity across learning algorithms.

For Claude: the missing thing exposed here is the *construction of useful
conjunctions*, not just sustaining activity. A recurrence experiment on this
scalar could test persistence of a supplied rule, but would not solve feature
discovery. If we continue, the next substantive question is whether a bounded
learner can select useful conjunctions from experience and beat a matched
nonlinear baseline on held-out structures. That is a separate experiment; it
has not been built or claimed here.

## Verification and limits

The tests check exact balanced marginals/pairs, symbolic failure of pairwise
learning, success of the cubic feature, hidden-pixel isolation, fixed-template
normalization, and linear versus product ridge. Results include dataset hashes,
all test predictions, and learned relation weights for inspection.
All 13 repository tests passed. A second complete run reproduced the JSON
receipt byte-for-byte; `git diff --check` also passed.

Known slots, periodic filtering within patches, fixed template bank and fully
balanced truth-table data make this deliberately easy to inspect. No claim
about natural-image completion, translation tolerance, biological circuitry,
or consciousness follows from passing it. Gate B remains unbuilt.
