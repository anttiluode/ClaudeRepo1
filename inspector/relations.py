"""Learned relations between the parts of an image, kept only if they predict held-out good images.

The world.html mechanism, moved onto real features:

1. Part types. k-means on the backbone's patch features from good training images (K types). Every
   patch of an image gets a type, giving a small label map.
2. Variables, each readable by a person: how much of type k lies in each of the 3x3 regions, and how
   many separate pieces (connected blobs) of type k the image has.
3. Branches. A branch predicts one variable from one or two others by nearest neighbours over the good
   training images. Candidates are searched with leave-one-out prediction on training images: for each
   variable the best one- or two-variable predictor (greedy), plus random triples whose two predictors
   together beat each alone by joint_gain (the three-way case that pairs cannot see). A candidate is
   kept only if it ALSO predicts the variable on held-out good images (the self-test: R^2 >= keep on
   the selection half of validation/good), so a relation found by searching many candidates on the
   same images has to survive images it has never seen.
4. Scoring. For a new image every kept branch reports how far the observed value lies from its
   prediction, in units of its own held-out error; every variable also reports how far it lies from
   its usual range. The image's relation score is the largest of these, and that term is its
   explanation: 'given A and B, C should be about x; it is y'.

Validation images are split: one half selects the branches, the other half sets each branch's error
scale and the score normalisation, so the scale is not fitted on the images that chose the branch."""
import numpy as np
import torch
from scipy import ndimage

REGIONS = ["top-left", "top", "top-right", "left", "centre", "right", "bottom-left", "bottom", "bottom-right"]


def kmeans(x, K, iters=30, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    x = x.to(device)
    c = x[torch.randint(len(x), (1,), generator=g).item()][None]
    for _ in range(K - 1):                                     # k-means++ seeding
        d = torch.cdist(x, c).min(1).values ** 2
        i = torch.multinomial((d / d.sum()).cpu(), 1, generator=g).item()
        c = torch.cat([c, x[i][None]])
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for k in range(K):
            m = a == k
            if m.any():
                c[k] = x[m].mean(0)
    return c


def knn_predict(Atr, ytr, Aq, k):
    d = ((Aq[:, None, :] - Atr[None, :, :]) ** 2).sum(-1)
    k = min(k, Atr.shape[0])
    idx = np.argpartition(d, k - 1, axis=1)[:, :k]
    return ytr[idx].mean(1)


class Relations:
    def __init__(self, K=12, grid=3, knn=5, keep=0.5, joint_gain=0.2, n_random=3000, min_blob=3,
                 area_floor=0.02, count_floor=0.5, err_floor=0.25, seed=0, device="cpu"):
        self.__dict__.update(K=K, grid=grid, knn=knn, keep=keep, joint_gain=joint_gain, n_random=n_random,
                             min_blob=min_blob, area_floor=area_floor, count_floor=count_floor,
                             err_floor=err_floor, seed=seed, device=device)

    # ---- part types and variables ------------------------------------------------------------------
    def labels(self, feats):
        out = []
        for f in feats.split(64):
            d = torch.cdist(f.to(self.device), self.centroids)
            out.append(d.argmin(-1).cpu())
        return torch.cat(out).reshape(-1, *self.hw).numpy()

    def variables(self, labels):
        N, h, w = labels.shape
        G = self.grid
        rs = [slice(round(i * h / G), round((i + 1) * h / G)) for i in range(G)]
        cs = [slice(round(j * w / G), round((j + 1) * w / G)) for j in range(G)]
        X = np.zeros((N, self.K * (G * G + 1)), np.float32)
        for n in range(N):
            for k in range(self.K):
                m = labels[n] == k
                base = k * (G * G + 1)
                for r in range(G):
                    for c in range(G):
                        X[n, base + r * G + c] = m[rs[r], cs[c]].mean()
                lab, nb = ndimage.label(m)
                if nb:
                    sizes = np.bincount(lab.ravel())[1:]
                    X[n, base + G * G] = (sizes >= self.min_blob).sum()
        return X

    def var_info(self, v):
        G2 = self.grid * self.grid
        k, j = divmod(v, G2 + 1)
        if j < G2:
            return k, "area", j, f"part type {k} in the {REGIONS[j] if self.grid == 3 else 'region ' + str(j)}"
        return k, "count", None, f"number of type {k} pieces"

    def location(self, v, labels_one):
        """Where a variable lives in the image (h x w bool): its region, or the pieces of its type."""
        k, kind, j, _ = self.var_info(v)
        h, w = labels_one.shape
        if kind == "area":
            G = self.grid
            r, c = divmod(j, G)
            m = np.zeros((h, w), bool)
            m[round(r * h / G):round((r + 1) * h / G), round(c * w / G):round((c + 1) * w / G)] = True
            return m
        return ndimage.binary_dilation(labels_one == k, iterations=1)

    # ---- learning ------------------------------------------------------------------------------------
    def fit(self, train_feats, val_feats, hw):
        self.hw = hw
        rng = np.random.default_rng(self.seed)
        flat = train_feats.reshape(-1, train_feats.shape[-1])
        sample = flat[torch.from_numpy(rng.choice(len(flat), min(len(flat), 60000), replace=False))]
        self.centroids = kmeans(sample, self.K, seed=self.seed, device=self.device)
        Xtr = self.variables(self.labels(train_feats))
        Xva = self.variables(self.labels(val_feats))
        self.mu = Xtr.mean(0)
        floors = np.array([self.count_floor if self.var_info(v)[1] == "count" else self.area_floor
                           for v in range(Xtr.shape[1])], np.float32)
        self.sd = np.maximum(Xtr.std(0), floors)
        Z = lambda X: (X - self.mu) / self.sd
        self.Ztr = Z(Xtr)
        Zsel, Zcal = Z(Xva[0::2]), Z(Xva[1::2])
        live = np.where(Xtr.std(0) > 1e-6)[0]
        self.n_live = len(live)

        def r2(t, S, Zq):                                    # held-out: memory = train, queries = Zq
            yhat = knn_predict(self.Ztr[:, S], self.Ztr[:, t], Zq[:, S], self.knn)
            return 1 - np.mean((Zq[:, t] - yhat) ** 2) / max(np.mean(Zq[:, t] ** 2), 1e-6)

        Zs = self.Ztr[rng.choice(len(self.Ztr), min(len(self.Ztr), 200), replace=False)]

        def r2_loo(t, S):                                    # search: leave-one-out on training images
            A = Zs[:, S]
            d = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d, np.inf)
            k = min(self.knn, len(A) - 1)
            yhat = Zs[np.argpartition(d, k - 1, axis=1)[:, :k], t].mean(1)
            return 1 - np.mean((Zs[:, t] - yhat) ** 2) / max(np.mean(Zs[:, t] ** 2), 1e-6)

        candidates = {}
        for t in live:                                       # greedy: best 1- or 2-variable predictor
            others = [j for j in live if j != t]
            singles = sorted(((r2_loo(t, [j]), j) for j in others), reverse=True)
            if not singles:
                continue
            best = (singles[0][0], (singles[0][1],))
            for _, j in singles[:5]:
                for m in others:
                    if m != j:
                        s = r2_loo(t, [j, m])
                        if s > best[0] + 0.05:               # a second predictor must earn its place
                            best = (s, tuple(sorted((j, m))))
            if best[0] >= self.keep:
                candidates[(t, best[1])] = best[0]
        single = {}
        for _ in range(self.n_random):                       # random triples: joint relations only
            t, j, m = (int(v) for v in rng.choice(live, 3, replace=False))
            S = tuple(sorted((j, m)))
            if (t, S) in candidates:
                continue
            s = r2_loo(t, list(S))
            if s < self.keep:
                continue
            a = single.setdefault((t, j), r2_loo(t, [j]))
            b = single.setdefault((t, m), r2_loo(t, [m]))
            if s - max(a, b) >= self.joint_gain:
                candidates[(t, S)] = s
        self.n_candidates = len(candidates)
        branches = {}
        for (t, S), s in candidates.items():                 # self-test on images never searched over
            h = r2(t, list(S), Zsel)
            if h >= self.keep:
                branches[(t, S)] = h
        self.branches = []
        for (t, S), s in sorted(branches.items(), key=lambda kv: -kv[1]):
            yhat = knn_predict(self.Ztr[:, list(S)], self.Ztr[:, t], Zcal[:, list(S)], self.knn)
            err = max(np.sqrt(np.mean((Zcal[:, t] - yhat) ** 2)), self.err_floor)
            self.branches.append({"target": int(t), "given": [int(v) for v in S], "r2": float(s), "err": float(err)})
        self.marg_scale = np.maximum(np.sqrt(np.mean(Zcal ** 2, 0)), 1.0)
        return self

    # ---- scoring and explanation -----------------------------------------------------------------------
    def score(self, feats):
        L = self.labels(feats)
        X = self.variables(L)
        Z = (X - self.mu) / self.sd
        N = len(Z)
        best = np.full(N, -np.inf)
        which = [None] * N
        for b in self.branches:
            S, t = b["given"], b["target"]
            yhat = knn_predict(self.Ztr[:, S], self.Ztr[:, t], Z[:, S], self.knn)
            z = np.abs(Z[:, t] - yhat) / b["err"]
            for n in np.where(z > best)[0]:
                best[n] = z[n]
                which[n] = ("relation", b, float(yhat[n]))
        zm = np.abs(Z) / self.marg_scale
        for n in range(N):
            v = int(zm[n].argmax())
            if zm[n, v] > best[n]:
                best[n] = zm[n, v]
                which[n] = ("marginal", v, 0.0)
        explanations, locations = [], []
        for n in range(N):
            kind, item, yhat = which[n]
            if kind == "relation":
                t = item["target"]
                expected = yhat * self.sd[t] + self.mu[t]
                text = ("given " + ", ".join(f"{self.var_info(g)[3]} = {X[n, g]:.2f}" for g in item["given"])
                        + f": {self.var_info(t)[3]} should be about {expected:.2f}, it is {X[n, t]:.2f}")
            else:
                t = item
                text = f"{self.var_info(t)[3]} is {X[n, t]:.2f}; in good images it is {self.mu[t]:.2f} ± {self.sd[t]:.2f}"
            explanations.append({"kind": kind, "target": int(t), "z": float(best[n]), "text": text})
            locations.append(self.location(t, L[n]))
        return best, explanations, np.stack(locations), L

    def describe(self, top=20):
        lines = []
        for b in self.branches[:top]:
            lines.append(f"R2 {b['r2']:.2f}: " + " + ".join(self.var_info(g)[3] for g in b["given"])
                         + f"  ->  {self.var_info(b['target'])[3]}")
        return lines
