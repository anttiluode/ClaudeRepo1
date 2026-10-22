import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from relations import pair_index, OFFSETS

NAMES = ["r0-1", "r1-2", "r2-4", "r4-8", "r8-16", "r16-32"]
CTRL = {"phase": "phase-scrambled\n(same spectrum)", "marginal": "spectrum + band\nhistograms matched",
        "shuffle": "16px blocks shuffled\n(seams caveat)"}


def _off(ax):
    ax.set_xticks([]); ax.set_yticks([])


def aucs(res, path):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    kinds = [("rectified", "relations of |r| (Hebbian)", "#e8743b"), ("linear", "relations of r (Hebbian)", "#5b8def"),
             ("marginal", "per-band shape only", "#9aa5b1")]
    x = np.arange(len(CTRL))
    for i, (k, lab, col) in enumerate(kinds):
        v = [res[c][k] for c in CTRL]
        ax.bar(x + (i - 1) * .27, v, .27, color=col, label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + (i - 1) * .27, vi + .01, f"{vi:.2f}", ha="center", fontsize=7)
    ax.axhline(.5, color="k", lw=.6, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(list(CTRL.values()), fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("AUC: real vs control", fontsize=8)
    ax.legend(fontsize=7, loc="lower right"); ax.set_title("Which statistics can tell a photograph from its controls", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def weights(layer, path):
    """The learned links: cross-scale at zero offset, and how they fall off with distance."""
    W = layer.W
    idx = pair_index(6)
    M = np.full((6, 6), np.nan)
    for (k, j, oi), w in zip(idx, W):
        if oi == 0:
            M[k, j] = M[j, k] = w
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    im = axes[0].imshow(M, cmap="magma", vmin=0, vmax=1)
    axes[0].set_xticks(range(6)); axes[0].set_yticks(range(6))
    axes[0].set_xticklabels(NAMES, fontsize=7, rotation=45); axes[0].set_yticklabels(NAMES, fontsize=7)
    for k in range(6):
        for j in range(6):
            axes[0].text(j, k, f"{M[k, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if M[k, j] < .6 else "k")
    axes[0].set_title("learned co-activation of |r| across scales\n(same place)", fontsize=8)
    plt.colorbar(im, ax=axes[0], fraction=.046)
    for k in range(6):
        d0 = 1.
        ys = [d0]
        for d in (2, 6):
            vals = [w for (kk, j, oi), w in zip(idx, W) if kk == k and j == k and OFFSETS[oi] in ((0, d), (d, 0))]
            ys.append(np.mean(vals))
        axes[1].plot([0, 2, 6], ys, marker="o", label=NAMES[k])
    axes[1].set_xlabel("lateral distance (px)", fontsize=8); axes[1].set_ylabel("learned co-activation, same band", fontsize=8)
    axes[1].legend(fontsize=6); axes[1].set_title("lateral links: coarser bands reach further", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def telling_links(layer, test, controls, path, top=12):
    """Which learned links does a control break most? Mean z-deviation over test images."""
    idx = pair_index(6)
    real = np.mean([layer.deviation(x) for x in test], 0)
    fig, axes = plt.subplots(1, len(controls), figsize=(12, 3.6))
    for ax, (c, imgs) in zip(axes, controls.items()):
        dev = np.mean([layer.deviation(x) for x in imgs], 0) - real
        order = np.argsort(np.abs(dev))[::-1][:top]
        labels = [f"{NAMES[idx[i][0]]}x{NAMES[idx[i][1]]} @{OFFSETS[idx[i][2]]}" for i in order]
        ax.barh(range(top), dev[order], color=["#c0392b" if v < 0 else "#5b8def" for v in dev[order]])
        ax.set_yticks(range(top)); ax.set_yticklabels(labels, fontsize=6); ax.invert_yaxis()
        ax.set_title(f"links broken by: {c}", fontsize=8); ax.set_xlabel("shift vs real images (sd units)", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def examples(layer, test, controls, path, n=4):
    fig, axes = plt.subplots(n, 1 + len(controls), figsize=(7.5, 1.9 * n))
    for r in range(n):
        x = test[r * (len(test) // n)]
        row = [("real", x)] + [(c, controls[c][r * (len(test) // n)]) for c in controls]
        for ax, (lab, im) in zip(axes[r], row):
            ax.imshow(im, cmap="gray", interpolation="nearest"); _off(ax)
            ax.set_title(f"{lab}\nscore {layer.score(im):.2f}", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def curve(curve, path):
    fig, ax = plt.subplots(figsize=(5, 3))
    ns = sorted(curve)
    for c, col in zip(CTRL, ("#e8743b", "#c0392b", "#9aa5b1")):
        ax.plot(ns, [curve[n][c] for n in ns], marker="o", color=col, label=c)
    ax.set_xscale("log"); ax.axhline(.5, color="k", lw=.6, ls=":")
    ax.set_xlabel("natural training crops seen", fontsize=8); ax.set_ylabel("AUC (rectified relations)", fontsize=8)
    ax.legend(fontsize=7); ax.set_title("how much experience the Hebbian layer needs", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def all_figures(layers, test, controls, real, res, crv, d):
    L = layers["rectified"]
    aucs(res, d / "auc.png")
    weights(L, d / "learned_links.png")
    telling_links(L, test, controls, d / "broken_links.png")
    examples(L, test, controls, d / "examples.png")
    curve(crv, d / "learning_curve.png")
