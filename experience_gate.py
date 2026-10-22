"""Small preregistered experience-reversal test; see docs/EXPERIENCE_PROTOCOL.md.

Hand-designed slots and two candidate primitives. Learns their association, not
features or object categories. Reuses Sihti statistics, with a known slot offset.
"""
import json
from pathlib import Path
import numpy as np
from relations import relation_vector, OFFSETS

LEFT = np.s_[24:40, 12:28]
RIGHT = np.s_[24:40, 36:52]
OFFSETS_TASK = OFFSETS + [(0, 24)]


def primitive(sign, width=1., dy=0.):
    y, x = np.mgrid[-7.5:8:1, -7.5:8:1]
    return np.exp(-.5*((y-dy-sign*x)/width)**2) * np.exp(-(x*x+(y-dy)**2)/70.)


def draw_examples(n, seed):
    if n % 2:
        raise ValueError('balanced sample count must be even')
    rng = np.random.default_rng(seed)
    cues = rng.permutation(np.tile([-1, 1], n//2))
    nuisance = [(rng.uniform(.25, .45), rng.uniform(.8, 1.3), rng.uniform(-1, 1),
                 int(rng.integers(2**31))) for _ in range(n)]
    return cues, nuisance


def scene(cue, target, nuisance):
    amplitude, width, dy, seed = nuisance
    x = .5 + np.random.default_rng(seed).normal(0, .01, (64, 64))
    x[LEFT] -= amplitude * primitive(cue, width, dy)
    x[RIGHT] -= amplitude * primitive(target, width, dy)
    return x


def visible_mask():
    mask = np.ones((64, 64), bool)
    mask[RIGHT] = False
    return mask


def candidates(observation):
    """Fixed hypothesis bank, identical for every learner; hidden data overwritten."""
    out = []
    for sign in (-1, 1):
        x = observation.copy()
        x[RIGHT] = .5 - .35*primitive(sign)
        out.append(x)
    return np.array(out)


def feature(x):
    return relation_vector(x, offsets=OFFSETS_TASK)


def metrics(pred_a, pred_b, cues):
    a, b = np.asarray(pred_a), np.asarray(pred_b)
    return dict(accuracy_A=float(np.mean(a == cues)),
                accuracy_B=float(np.mean(b == -cues)),
                paired_correct=float(np.mean((a == cues) & (b == -cues))),
                prediction_flip=float(np.mean(a != b)))


def choose(features, weight):
    # Near-ties use the same candidate for both environments, never the truth.
    delta = (features[:, 1] - features[:, 0]) @ weight
    return np.where(delta > 1e-12, 1, -1)


def one_seed(seed):
    cues, nuisances = draw_examples(64, seed)
    images = np.array([[scene(q, z, n) for z in (-1, 1)] for q, n in zip(cues, nuisances)])
    feats = np.array([[feature(x) for x in pair] for pair in images])
    null = feats.mean(axis=1)
    row = np.arange(len(cues))
    observed = [feats[row, (e*cues+1)//2] for e in (1, -1)]
    weights = [np.mean(f-null, axis=0) for f in observed]
    raw = [f.mean(axis=0) for f in observed]
    rng = np.random.default_rng(seed+10000)
    permutation = rng.permutation(feats.shape[-1])
    # Independent reassignment preserves the exact target primitive counts.
    reassigned = rng.permutation(cues)
    unpaired = [np.mean(feats[row, (e*reassigned+1)//2]-null, axis=0) for e in (1, -1)]
    test_q, test_n = draw_examples(32, seed+1000)
    # Hidden orientation here is a dummy; candidates() overwrites it completely.
    obs = np.array([scene(q, 1, n) for q, n in zip(test_q, test_n)])
    candidate_images = np.array([candidates(x) for x in obs])
    test_f = np.array([[feature(x) for x in pair] for pair in candidate_images])
    methods = {'contrastive_hebb': weights, 'mean_hebb': raw,
               'all_ones': [np.ones_like(weights[0])]*2,
               'shuffled_weights': [w[permutation] for w in weights],
               'reassigned_partners': unpaired}
    result, predictions = {}, {}
    for name, ww in methods.items():
        pa, pb = [choose(test_f, w) for w in ww]
        result[name] = metrics(pa, pb, test_q)
        predictions[name] = [pa.tolist(), pb.tolist()]
    # Ordinary ridge: visible left-slot pixels -> hidden right-slot pixels.
    # Dual solve, fixed regularisation, no test-target fitting.
    X = images[:, 0, 24:40, 12:28].reshape(64, -1)
    Xt = obs[:, 24:40, 12:28].reshape(32, -1)
    mu = X.mean(0)
    X, Xt = X-mu, Xt-mu
    kernel = np.linalg.solve(X@X.T + .1*np.eye(len(X)), X)
    ridge_predictions = []
    for e in (1, -1):
        Y = images[row, (e*cues+1)//2, 24:40, 36:52].reshape(64, -1)
        ym = Y.mean(0)
        pred = Xt @ kernel.T @ (Y-ym) + ym
        choices = candidate_images[:, :, 24:40, 36:52].reshape(32, 2, -1)
        cost = np.mean((choices-pred[:, None])**2, axis=2)
        ridge_predictions.append(np.where(cost[:, 1] < cost[:, 0], 1, -1))
    result['pixel_ridge'] = metrics(*ridge_predictions, test_q)
    predictions['pixel_ridge'] = [p.tolist() for p in ridge_predictions]
    # Swap weights at inference: direct causal intervention on learned association.
    result['swapped_experience_weights'] = metrics(choose(test_f, weights[1]),
                                                  choose(test_f, weights[0]), test_q)
    return {'seed': seed, 'scores': result, 'test_cues': test_q.tolist(),
            'predictions': predictions, 'weight_norm': [float(np.linalg.norm(w)) for w in weights]}


def main():
    runs = [one_seed(seed) for seed in range(101, 106)]
    means = {name: {key: float(np.mean([r['scores'][name][key] for r in runs]))
                    for key in runs[0]['scores'][name]} for name in runs[0]['scores']}
    learned = means['contrastive_hebb']
    controls = ('all_ones', 'shuffled_weights', 'reassigned_partners')
    passed = (min(learned['accuracy_A'], learned['accuracy_B']) >= .9
              and learned['paired_correct'] >= .8
              and all(learned['paired_correct']-means[c]['paired_correct'] >= .2 for c in controls))
    result = {'protocol': 'docs/EXPERIENCE_PROTOCOL.md', 'gate_pass': bool(passed),
              'means': means, 'runs': runs,
              'limits': 'Synthetic known slots and candidate bank; no natural-image, recurrence or biological claim.'}
    path = Path('results/astra/experience.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'gate_pass': bool(passed), 'means': means}, indent=2))


if __name__ == '__main__':
    main()
