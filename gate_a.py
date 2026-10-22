"""Gate A: does a Hebbian relation layer on rectified Sihti bands carry anything?

    python gate_a.py                         # the preregistered run (docs/PREREG_A.md)
    python gate_a.py --folders ~/THINGS/unseen --name folders   # test crops from your own photos

The layers learn ONLY from natural training crops; they never see a control image.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import sieve as S
from relations import HebbLayer, auc, pair_index, OFFSETS


def score_all(layer, images):
    return np.array([layer.score(x) for x in images])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40, help="crops per training photograph")
    ap.add_argument("--n-test", type=int, default=50, help="crops per test photograph")
    ap.add_argument("--folders", default=None)
    ap.add_argument("--name", default="natural")
    a = ap.parse_args()
    t0 = time.time()
    train = S.crops([S.photo(p) for p in S.TRAIN_PHOTOS], a.n_train, seed=1)
    test_src = S.load_folder(a.folders) if a.folders else [S.photo(p) for p in S.TEST_PHOTOS]
    test = S.crops(test_src, a.n_test, seed=2)
    print(f"{len(train)} training crops, {len(test)} test crops", flush=True)

    rng = np.random.default_rng(3)
    controls = {name: np.array([f(x, rng) for x in test]) for name, f in S.CONTROLS.items()}

    layers = {k: HebbLayer(k).fit(train) for k in ("rectified", "linear", "marginal")}
    real = {k: score_all(L, test) for k, L in layers.items()}
    res = {}
    for c, imgs in controls.items():
        res[c] = {}
        for k, L in layers.items():
            res[c][k] = auc(real[k], score_all(L, imgs))
    print(json.dumps(res, indent=2))

    # learning curve: how much experience the rectified layer needs
    curve = {}
    for n in (5, 10, 20, 50, 100, 200, len(train)):
        idx = np.random.default_rng(n).permutation(len(train))[:n]
        L = HebbLayer("rectified").fit(train[idx])
        curve[n] = {c: auc(score_all(L, test), score_all(L, controls[c])) for c in controls}

    g = {
        "A1_rectified_phase": res["phase"]["rectified"],
        "A1_linear_phase": res["phase"]["linear"],
        "A1_PASS": bool(res["phase"]["rectified"] >= .80 and .40 <= res["phase"]["linear"] <= .60),
        "A2_rectified_marginal": res["marginal"]["rectified"],
        "A2_marginalfeatures_marginal": res["marginal"]["marginal"],
        "A2_PASS": bool(res["marginal"]["rectified"] >= .70 and
                        res["marginal"]["rectified"] >= res["marginal"]["marginal"] + .10),
        "A3_report_shuffle": res["shuffle"],
    }
    print(json.dumps(g, indent=2))
    d = Path("results") / a.name
    d.mkdir(parents=True, exist_ok=True)
    L = layers["rectified"]
    out = {"auc": res, "gates": g, "learning_curve": curve, "n_train": len(train), "n_test": len(test),
           "W_rectified": L.W.tolist(), "sd_rectified": L.sd.tolist(),
           "pairs": [list(p) for p in pair_index(6)], "offsets": OFFSETS,
           "minutes": round((time.time() - t0) / 60, 2)}
    (d / "metrics.json").write_text(json.dumps(out, indent=2))
    import figures_a
    figures_a.all_figures(layers, test, controls, real, res, curve, d)


if __name__ == "__main__":
    main()
