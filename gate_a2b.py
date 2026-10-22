"""Gate A2b (docs/PREREG_A2b.md): the Hebbian unit's DRIVE as readout, on fresh photographs."""
import json
from pathlib import Path
import numpy as np
import sieve as S
from relations import HebbLayer, FEATURES, pair_index, auc

FRESH = ["cat", "cell"]


def drive_fn(layer, kind):
    W = layer.W.copy()
    if kind == "rectified":
        keep = np.array([not (k == j and oi == 0) for k, j, oi in pair_index(6)])
    else:
        keep = np.ones_like(W, bool)
    w = W[keep] / np.linalg.norm(W[keep])
    return lambda x: float(FEATURES[kind](x)[keep] @ w)


def evaluate(test, layers, seed):
    rng = np.random.default_rng(seed)
    controls = {c: [f(x, rng) for x in test] for c, f in S.CONTROLS.items()}
    out = {}
    for kind, L in layers.items():
        d = drive_fn(L, kind)
        real = [d(x) for x in test]
        out[kind] = {c: auc(real, [d(y) for y in imgs]) for c, imgs in controls.items()}
    return out


def main():
    train = S.crops([S.photo(p) for p in S.TRAIN_PHOTOS], 40, seed=1)
    layers = {k: HebbLayer(k).fit(train) for k in ("rectified", "marginal")}
    fresh = S.crops([S.photo(p) for p in FRESH], 50, seed=22)
    reused = S.crops([S.photo(p) for p in S.TEST_PHOTOS], 50, seed=2)
    res = {"fresh": evaluate(fresh, layers, 23), "reused_gateA_test": evaluate(reused, layers, 3),
           "n_fresh": len(fresh)}
    f = res["fresh"]
    res["A2b_PASS"] = bool(f["rectified"]["marginal"] >= .70 and
                           f["rectified"]["marginal"] >= f["marginal"]["marginal"] + .10 and
                           f["rectified"]["phase"] >= .80)
    print(json.dumps(res, indent=2))
    Path("results/natural/gate_a2b.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()


def curve_figure(path="results/natural/learning_curve_drive.png"):
    """Descriptive (after A2b): drive-readout AUC on the fresh photographs vs experience."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    train = S.crops([S.photo(p) for p in S.TRAIN_PHOTOS], 40, seed=1)
    fresh = S.crops([S.photo(p) for p in FRESH], 50, seed=22)
    rng = np.random.default_rng(23)
    controls = {c: [f(x, rng) for x in fresh] for c, f in S.CONTROLS.items()}
    ns, curves = (2, 5, 10, 20, 50, 100, 200, len(train)), {c: [] for c in controls}
    for n in ns:
        idx = np.random.default_rng(n).permutation(len(train))[:n]
        d = drive_fn(HebbLayer("rectified").fit(train[idx]), "rectified")
        real = [d(x) for x in fresh]
        for c, imgs in controls.items():
            curves[c].append(auc(real, [d(y) for y in imgs]))
    fig, ax = plt.subplots(figsize=(5, 3))
    for (c, v), col in zip(curves.items(), ("#e8743b", "#c0392b", "#9aa5b1")):
        ax.plot(ns, v, marker="o", color=col, label=c)
    ax.set_xscale("log"); ax.axhline(.5, color="k", lw=.6, ls=":"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("natural training crops seen", fontsize=8); ax.set_ylabel("AUC, Hebbian drive (fresh photos)", fontsize=8)
    ax.legend(fontsize=7); ax.set_title("how much experience the Hebbian unit needs", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return dict(zip(map(str, ns), zip(*curves.values())))


def no_learning_control():
    """Post-hoc (after A2b): does the LEARNING matter? Same drive readout with W replaced by
    all ones (no experience) or by the learned W with its entries shuffled."""
    fresh = S.crops([S.photo(p) for p in FRESH], 50, seed=22)
    rng = np.random.default_rng(23)
    controls = {c: [f(x, rng) for x in fresh] for c, f in S.CONTROLS.items()}
    keep = np.array([not (k == j and oi == 0) for k, j, oi in pair_index(6)])
    F = lambda x: FEATURES["rectified"](x)[keep]
    real = [F(x) for x in fresh]
    ctrl = {c: [F(y) for y in v] for c, v in controls.items()}
    train = S.crops([S.photo(p) for p in S.TRAIN_PHOTOS], 40, seed=1)
    W = HebbLayer("rectified").fit(train).W[keep]
    out = {}
    for name, w in (("learned", W), ("all_ones_no_learning", np.ones(keep.sum())),
                    ("learned_shuffled", np.random.default_rng(0).permutation(W))):
        out[name] = {c: auc([r @ w for r in real], [y @ w for y in v]) for c, v in ctrl.items()}
    out["W_min_max"] = [float(W.min()), float(W.max())]
    Path("results/natural/no_learning_control_posthoc.json").write_text(json.dumps(out, indent=2))
    return out
