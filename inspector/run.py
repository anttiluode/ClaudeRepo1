"""Run the inspector on MVTec LOCO AD (or any folder with the same layout).

    python -m inspector.run --root /path/to/mvtec_loco            # all five categories, GPU if present

Per category: PatchCore (unary memory) and the relation term are fitted on train/good (+ validation/good
for the relation self-test), then scored on test. Reports image AUROC for logical and structural
anomalies, for PatchCore alone, relations alone, and their sum (each standardised on validation/good),
plus whether the explanation of each logical anomaly points at the ground-truth region."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from . import data
from .patchcore import Backbone, PatchCore, PixelBackbone, extract
from .relations import Relations


def auroc(scores, kinds, kind):
    m = np.isin(kinds, ["good", kind]) if kind != "all" else np.ones(len(kinds), bool)
    y = (kinds[m] != "good").astype(int)
    return float(roc_auc_score(y, scores[m])) if 0 < y.sum() < len(y) else float("nan")


def upsample(mask_hw, size):
    t = torch.from_numpy(mask_hw.astype(np.float32))[None, None]
    return F.interpolate(t, size=(size, size), mode="nearest")[0, 0].numpy() > 0.5


def run_category(args, backbone, cat):
    t0 = time.time()
    train, val, test = data.split(args.root, cat)
    if args.limit:
        train, val = train[:args.limit], val[:max(4, args.limit // 3)]
    paths = [p for p, _ in test]
    kinds = np.array([k for _, k in test])
    ld = lambda p, s: data.load(p, s)
    ftr, hw = extract(backbone, train, args.size, ld)
    fva, _ = extract(backbone, val, args.size, ld)
    fte, _ = extract(backbone, paths, args.size, ld)
    t_feat = time.time() - t0

    pc = PatchCore(args.device, args.coreset).fit(ftr)
    pc_val, _ = pc.score(fva)
    pc_test, _ = pc.score(fte)
    t_pc = time.time() - t0 - t_feat

    rel = Relations(K=args.K, device=args.device, seed=args.seed).fit(ftr, fva, hw)
    rel_val, _, _, _ = rel.score(fva[1::2])           # the calibration half only
    rel_test, expl, locs, _ = rel.score(fte)
    t_rel = time.time() - t0 - t_feat - t_pc

    std = lambda s, ref: (s - ref.mean()) / max(ref.std(), 1e-6)
    z_pc, z_rel = std(pc_test, pc_val), std(rel_test, rel_val)
    combined = np.maximum(z_pc, z_rel) if args.combine == "max" else z_pc + z_rel
    np.savetxt(Path(args.out) / f"{cat}_scores.csv",
               np.c_[np.arange(len(paths)), z_pc, z_rel], delimiter=",", fmt="%.5f",
               header="index,patchcore_z,relation_z  (" + ";".join(f"{k}/{p.name}" for k, p in zip(kinds, paths)) + ")")
    kinds_path = Path(args.out) / f"{cat}_kinds.txt"
    kinds_path.write_text("\n".join(kinds))
    res = {"category": cat, "n_train": len(train), "n_val": len(val),
           "n_test": {k: int((kinds == k).sum()) for k in data.KINDS},
           "relations": {"part_types": args.K, "variables_live": rel.n_live, "candidates_found": rel.n_candidates, "branches_kept": len(rel.branches),
                         "top_branches": rel.describe(15)},
           "auroc": {}, "seconds": {"features": round(t_feat, 1), "patchcore": round(t_pc, 1), "relations": round(t_rel, 1)}}
    for name, s in [("patchcore", pc_test), ("relations", rel_test), ("combined", combined)]:
        res["auroc"][name] = {k: auroc(s, kinds, k) for k in ["logical_anomalies", "structural_anomalies", "all"]}

    # explanation check: does the flagged variable's location overlap the ground-truth mask?
    rng = np.random.default_rng(args.seed)
    hits, chance, examples = [], [], []
    labels_test = rel.labels(fte)
    for i in np.where(kinds == "logical_anomalies")[0]:
        gt = data.gt_mask(args.root, cat, paths[i], "logical_anomalies", args.size)
        if gt is None:
            continue
        hits.append(bool((upsample(locs[i], args.size) & gt).any()))
        v = int(rng.integers(rel.mu.shape[0]))
        chance.append(bool((upsample(rel.location(v, labels_test[i]), args.size) & gt).any()))
    order = np.argsort(-combined)
    for i in order[:12]:
        examples.append({"image": f"{kinds[i]}/{paths[i].name}", "combined_z": round(float(combined[i]), 2),
                         "patchcore_z": round(float(z_pc[i]), 2), "relation_z": round(float(z_rel[i]), 2),
                         "explanation": expl[i]["text"]})
    res["explanation"] = {"logical_with_masks": len(hits),
                          "flagged_location_overlaps_truth": float(np.mean(hits)) if hits else None,
                          "random_variable_overlaps_truth": float(np.mean(chance)) if chance else None}
    res["most_anomalous"] = examples
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--categories", default="all")
    ap.add_argument("--out", default="results/inspector")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--combine", choices=["max", "sum"], default="sum",
                    help="how the standardised PatchCore and relation scores are combined")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-pretrained", action="store_true", help="random backbone (smoke tests only)")
    ap.add_argument("--pixels", action="store_true", help="colour statistics instead of a CNN (smoke tests only)")
    ap.add_argument("--limit", type=int, default=0, help="cap training images (smoke tests only)")
    args = ap.parse_args()
    cats = data.CATEGORIES if args.categories == "all" else args.categories.split(",")
    torch.manual_seed(args.seed)
    backbone = PixelBackbone() if args.pixels else Backbone(args.device, pretrained=not args.no_pretrained)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for cat in cats:
        r = run_category(args, backbone, cat)
        results.append(r)
        (out / f"{cat}.json").write_text(json.dumps(r, indent=1))
        a = r["auroc"]
        print(f"{cat:22s} logical AUROC  patchcore {a['patchcore']['logical_anomalies']:.3f}  relations "
              f"{a['relations']['logical_anomalies']:.3f}  combined {a['combined']['logical_anomalies']:.3f}   "
              f"structural  patchcore {a['patchcore']['structural_anomalies']:.3f}  combined "
              f"{a['combined']['structural_anomalies']:.3f}   branches {r['relations']['branches_kept']}", flush=True)
    mean = lambda m, k: float(np.nanmean([r["auroc"][m][k] for r in results]))
    summary = {m: {k: mean(m, k) for k in ["logical_anomalies", "structural_anomalies", "all"]}
               for m in ["patchcore", "relations", "combined"]}
    hit = [r["explanation"]["flagged_location_overlaps_truth"] for r in results if r["explanation"]["flagged_location_overlaps_truth"] is not None]
    ch = [r["explanation"]["random_variable_overlaps_truth"] for r in results if r["explanation"]["random_variable_overlaps_truth"] is not None]
    summary["explanation_hit"] = float(np.mean(hit)) if hit else None
    summary["explanation_chance"] = float(np.mean(ch)) if ch else None
    summary["settings"] = {k: v for k, v in vars(args).items() if k != "root"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
