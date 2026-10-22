# Three-piece gate: protocol fixed before execution

This is one bounded extension of the experience gate, not Gate B.
Base: main 1e41ba9; branch astra/experience-gate.

## Question and representation

Two visible orientation signs a,b determine a hidden sign c.
World A: c=a*b. World B: c=-a*b.
Every group contains all four (a,b) combinations with the same nuisance draws.
Thus all sign marginals and all sign pairs match across worlds. The triple differs.

Render three separate 16x16 line patches using experience_gate.primitive.
Measure each patch with a fixed orientation-template contrast over rectified
Sihti bands. This supplied readout and the supplied slots are not learned.
Use soft scalar evidence z1,z2,z3, not the true signs, for relation learning.

Compare pairwise features (z1*z2,z1*z3,z2*z3) with an explicitly supplied cubic
feature z1*z2*z3. Learn mean observed minus mean independently recombined target
(the exact average over both candidate target patches). The cubic feature is a
new representation; success cannot establish that pairwise Hebb discovers it.

## Fixed budget and controls

Five seeds 201..205. Per seed 64 train examples (16 balanced groups), 32 test
examples (8 new groups). Same visible patches for both worlds; hidden data never
enter prediction. Amplitude .25..45, width .8..1.3, vertical shift -1..1,
pixel noise sigma .01; independent nuisance per slot, shared within each group.
No downloads, tuning, recurrence, or subsequent gate in this branch.

Controls: pairwise contrastive rule, fixed positive cubic weight, independently
reassigned targets, weights swapped between worlds, linear ridge on the two
visible soft evidences, ridge on those evidences plus their product.
Both ridge models predict the target sign with regularization .1 and intercept.
The product ridge is the ordinary nonlinear reference and may tie the cubic rule.
All ties choose -1. Reassignment preserves target counts.

Also measure the previous global Sihti relation_vector (including offset 24)
on full three-patch images as a diagnostic. It is nonlinear in pixels and may
already mix three slots: no theorem says it must fail parity.

## Gate and reporting

Report each world's accuracy, paired correctness (both worlds correct for the
same input), and prediction flips, per seed and pooled.
Pass: cubic accuracy >= .9 in each world, paired >= .8, and paired margin >= .2
over pairwise, fixed positive weight, reassignment, and linear ridge.
Product ridge and the previous global descriptor are references, not required losers.
Report failures without changing thresholds. Report the extracted sign accuracy,
learned weights, predictions, and dataset hashes for reproducibility.
A pass demonstrates learning a supplied higher-order relation, not feature
discovery, object recognition, translation tolerance, superiority to nonlinear
regression, cortical circuitry, or conscious access.
