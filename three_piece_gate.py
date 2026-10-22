"""Balanced parity with supplied slots/features; docs/THREE_PIECE_PROTOCOL.md.

No true hidden sign or hidden patch is an input to candidate scoring.
The three-way feature is engineered, not discovered by Hebbian learning.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from experience_gate import primitive, feature, metrics
from sieve import sieve

SIGNS = np.array([-1, 1])
SLOTS = [(slice(24, 40), slice(x, x+16)) for x in (4, 24, 44)]


def dataset(groups, seed):
    rng = np.random.default_rng(seed)
    q = np.tile([[-1, -1], [-1, 1], [1, -1], [1, 1]], (groups, 1))
    patches = []
    for _ in range(groups):
        slots = []
        for _ in range(3):
            amp, width, dy = rng.uniform(.25, .45), rng.uniform(.8, 1.3), rng.uniform(-1, 1)
            noise = rng.normal(0, .01, (16, 16))
            slots.append([.5+noise-amp*primitive(s, width, dy) for s in SIGNS])
        patches.extend([slots]*4)
    return q, np.asarray(patches)


def rectified(patch):
    return np.abs(np.asarray(sieve(patch)[0])).ravel()


TEMPLATES = np.array([.5-.35*primitive(s) for s in SIGNS])
TEMPLATE_BANDS = np.array([rectified(p) for p in TEMPLATES])
CONTRAST = TEMPLATE_BANDS[1]-TEMPLATE_BANDS[0]
MIDPOINT = TEMPLATE_BANDS.mean(0)


def evidence(patch):
    """Fixed local orientation readout, normalized to +/-1 on templates."""
    return 2*(rectified(patch)-MIDPOINT)@CONTRAST/(CONTRAST@CONTRAST)


def candidate_evidence(visible):
    """Only two visible patches accepted; fixed alternatives supply slot three."""
    z = np.array([[evidence(p) for p in pair] for pair in visible])
    return np.stack([np.column_stack([z, np.full(len(z), evidence(p))])
                     for p in TEMPLATES], axis=1)


def relation_features(z, degree):
    if degree == 2:
        return np.stack([z[..., 0]*z[..., 1], z[..., 0]*z[..., 2],
                         z[..., 1]*z[..., 2]], axis=-1)
    if degree == 3:
        return np.prod(z, axis=-1)[..., None]
    raise ValueError('degree must be 2 or 3')


def fit_relation(features, targets):
    selected = features[np.arange(len(targets)), (targets+1)//2]
    return np.mean(selected-features.mean(axis=1), axis=0)


def predict_relations(features, weights):
    delta = (features[:, 1]-features[:, 0])@weights
    return np.where(delta > 1e-12, 1, -1)


def full_image(patches):
    x = np.full((64, 64), .5)
    for slot, patch in zip(SLOTS, patches):
        x[slot] = patch
    return x


def visible_patches(q, patches):
    return np.array([[p[0, (a+1)//2], p[1, (b+1)//2]]
                     for (a, b), p in zip(q, patches)])


def training_candidates(q, patches):
    visible = visible_patches(q, patches)
    return np.array([[np.concatenate([v, p[2, j][None]]) for j in (0, 1)]
                     for v, p in zip(visible, patches)])


def ridge_predictions(train_z, test_z, target, product=False):
    X, Xt = train_z[:, 0, :2], test_z[:, 0, :2]
    if product:
        X = np.column_stack([X, np.prod(X, axis=1)])
        Xt = np.column_stack([Xt, np.prod(Xt, axis=1)])
    mu, ym = X.mean(0), target.mean()
    xc = X-mu
    w = np.linalg.solve(xc.T@xc+.1*np.eye(X.shape[1]), xc.T@(target-ym))
    return np.where((Xt-mu)@w+ym > 1e-12, 1, -1)


def one_seed(seed):
    q, patches = dataset(16, seed)
    qt, pt = dataset(8, seed+1000)
    train_images = training_candidates(q, patches)
    # Construct test alternatives from visible patches alone, never pt[:, 2].
    visible = visible_patches(qt, pt)
    test_images = np.array([[np.concatenate([v, p[None]]) for p in TEMPLATES]
                            for v in visible])
    train_z = np.array([[[evidence(p) for p in candidate] for candidate in pair]
                        for pair in train_images])
    test_z = candidate_evidence(visible)
    truth, test_truth = np.prod(q, axis=1), np.prod(qt, axis=1)
    train_f = {d: relation_features(train_z, d) for d in (2, 3)}
    test_f = {d: relation_features(test_z, d) for d in (2, 3)}
    weights, predictions = {}, {}
    for d, name in ((2, 'pairwise_hebb'), (3, 'cubic_hebb')):
        ww = [fit_relation(train_f[d], e*truth) for e in (1, -1)]
        weights[name] = [w.tolist() for w in ww]
        predictions[name] = [predict_relations(test_f[d], w) for w in ww]
    predictions['fixed_positive'] = [predict_relations(test_f[3], np.ones(1))]*2
    reassigned = np.random.default_rng(seed+10000).permutation(truth)
    predictions['reassigned_targets'] = [predict_relations(test_f[3],
        fit_relation(train_f[3], e*reassigned)) for e in (1, -1)]
    predictions['swapped_weights'] = predictions['cubic_hebb'][::-1]
    for product, name in ((False, 'linear_ridge'), (True, 'product_ridge')):
        predictions[name] = [ridge_predictions(train_z, test_z, e*truth, product)
                             for e in (1, -1)]
    # Reuse old whole-image nonlinear descriptor without changing its features.
    old_train = np.array([[feature(full_image(c)) for c in pair] for pair in train_images])
    old_test = np.array([[feature(full_image(c)) for c in pair] for pair in test_images])
    predictions['previous_global_relations'] = [predict_relations(old_test,
        fit_relation(old_train, e*truth)) for e in (1, -1)]
    scores = {name: metrics(*pred, test_truth) for name, pred in predictions.items()}
    return dict(seed=seed, scores=scores, weights=weights,
                visible_sign_accuracy=float(np.mean(np.where(test_z[:, 0, :2] > 0, 1, -1) == qt)),
                dataset_sha256=hashlib.sha256(patches.tobytes()+pt.tobytes()).hexdigest(),
                test_cues=qt.tolist(), predictions={k: np.asarray(v).tolist() for k, v in predictions.items()})


def main():
    runs = [one_seed(s) for s in range(201, 206)]
    means = {name: {key: float(np.mean([r['scores'][name][key] for r in runs]))
                    for key in runs[0]['scores'][name]} for name in runs[0]['scores']}
    learned = means['cubic_hebb']
    controls = ('pairwise_hebb', 'fixed_positive', 'reassigned_targets', 'linear_ridge')
    passed = (min(learned['accuracy_A'], learned['accuracy_B']) >= .9
              and learned['paired_correct'] >= .8
              and all(learned['paired_correct']-means[c]['paired_correct'] >= .2 for c in controls))
    result = dict(protocol='docs/THREE_PIECE_PROTOCOL.md', gate_pass=bool(passed),
                  means=means, runs=runs,
                  limits='Supplied slots, template readout and cubic feature. Not learned feature discovery.')
    path = Path('results/astra/three_piece.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(gate_pass=bool(passed), means=means), indent=2))


if __name__ == '__main__':
    main()
