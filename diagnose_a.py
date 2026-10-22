"""POST-HOC diagnosis of Gate A (written after A2 failed; not preregistered).

1. paired: for each test crop, does the real crop score higher than ITS OWN control?
   (removes image-to-image differences in typicality, which the unpaired AUC mixes in)
2. probe: a supervised logistic readout trained on TRAINING crops and their controls,
   tested on test crops - no longer Hebbian-only; asks whether the features carry the
   information at all when a readout is allowed to learn which direction matters.
"""
import json
import numpy as np
import sieve as S
from relations import HebbLayer, FEATURES


def logistic(X, y, Xt, l2=1.0, iters=400, lr=.5):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X, Xt = (X - mu) / sd, (Xt - mu) / sd
    w, b = np.zeros(X.shape[1]), 0.
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w / len(y))
        b -= lr * float(np.mean(p - y))
    return 1 / (1 + np.exp(-(Xt @ w + b)))


def main():
    train = S.crops([S.photo(p) for p in S.TRAIN_PHOTOS], 40, seed=1)
    test = S.crops([S.photo(p) for p in S.TEST_PHOTOS], 50, seed=2)
    rng = np.random.default_rng(3)
    ctrl_test = {c: np.array([f(x, rng) for x in test]) for c, f in S.CONTROLS.items()}
    rng2 = np.random.default_rng(13)
    ctrl_train = {c: np.array([f(x, rng2) for x in train]) for c, f in S.CONTROLS.items()}
    out = {"paired": {}, "probe": {}}
    feats = {}
    for k in ("rectified", "linear", "marginal"):
        L = HebbLayer(k).fit(train)
        out["paired"][k] = {c: float(np.mean([L.score(x) > L.score(y) for x, y in zip(test, ctrl_test[c])])) for c in ctrl_test}
        F = FEATURES[k]
        feats[k] = (np.array([F(x) for x in train]), {c: np.array([F(x) for x in v]) for c, v in ctrl_train.items()},
                    np.array([F(x) for x in test]), {c: np.array([F(x) for x in v]) for c, v in ctrl_test.items()})
    feats["rectified+marginal"] = tuple(
        (np.hstack([feats["rectified"][i], feats["marginal"][i]]) if i in (0, 2) else
         {c: np.hstack([feats["rectified"][i][c], feats["marginal"][i][c]]) for c in feats["rectified"][i]})
        for i in range(4))
    for k, (tr, trc, te, tec) in feats.items():
        out["probe"][k] = {}
        for c in ctrl_test:
            X = np.vstack([tr, trc[c]]); y = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
            p = logistic(X, y, np.vstack([te, tec[c]]))
            out["probe"][k][c] = float(np.mean((p > .5) == np.r_[np.ones(len(te)), np.zeros(len(te))]))
    print(json.dumps(out, indent=2))
    with open("results/natural/diagnosis_posthoc.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
