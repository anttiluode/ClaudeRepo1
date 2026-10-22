"""The relation layer: which configurations across the bands occur together.

For one image, each band r_k becomes an activity map. Two versions:

  rectified a_k = |r_k|        (a nonlinearity first: 'is something happening here, at this scale')
  linear    s_k =  r_k         (the raw signed coefficient)

each standardised per image and band (zero mean, unit variance), so a relation is a
correlation, i.e. observed co-activation minus what the two activities' own means
predict. The relation tensor of an image is

  R[k, k', delta] = mean_p  a_k(p) * a_k'(p + delta)

over band pairs (k, k') and spatial offsets delta (torus). Same scale + offset =
lateral; different scale, zero offset = across the sieve; both = across scale and space.

Hebbian learning: one weight per (k, k', delta), updated image by image

  W <- W + eta * (R_image - W),   eta = 1/n

i.e. the running average of the co-activations seen in natural images. This is the
covariance (observed-minus-expected) Hebb rule; with eta = 1/n it is exactly the mean.

What the two versions can see: on a torus each spatially averaged linear-version relation here is a
second-order statistic, fixed by the power spectrum. The rectified version is nonlinear in the image (not literally a fourth-order
polynomial because it uses absolute values): it can see whether things at one scale
happen in the same place at another.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import kurtosis
from sieve import sieve

OFFSETS = [(0, 0)] + [o for d in (2, 6) for o in ((0, d), (d, 0), (d, d), (d, -d))]


def activities(x, kind="rectified"):
    bands, _ = sieve(x)
    out = []
    for b in bands:
        a = np.abs(b) if kind == "rectified" else b
        out.append((a - a.mean()) / (a.std() + 1e-12))
    return np.array(out)                                   # (K, H, W)


def pair_index(K, offsets=OFFSETS):
    """(k, k', offset_index) for every relation kept (symmetric duplicates dropped at delta=0)."""
    idx = []
    for oi, o in enumerate(offsets):
        for k in range(K):
            for j in range(K):
                if o == (0, 0) and j < k:
                    continue
                idx.append((k, j, oi))
    return idx


def relation_vector(x, kind="rectified", offsets=OFFSETS):
    a = activities(x, kind)
    K = len(a)
    shifted = {o: np.roll(a, shift=(-o[0], -o[1]), axis=(1, 2)) for o in offsets}
    return np.array([float(np.mean(a[k] * shifted[offsets[oi]][j])) for k, j, oi in pair_index(K, offsets)])


def marginal_vector(x):
    """No relations at all: each band's own shape (kurtosis, mean |z|, skew of |r|)."""
    bands, _ = sieve(x)
    f = []
    for b in bands:
        z = (b - b.mean()) / (b.std() + 1e-12)
        f += [float(kurtosis(z.ravel())), float(np.mean(np.abs(z))), float(np.mean(np.abs(z) ** 3))]
    return np.array(f)


FEATURES = {"rectified": lambda x: relation_vector(x, "rectified"),
            "linear": lambda x: relation_vector(x, "linear"),
            "marginal": marginal_vector}


@dataclass
class HebbLayer:
    """W learned online from natural images; scores how well an image's own
    co-activations agree with the learned ones (z-scored squared distance, negated)."""
    feature: str = "rectified"
    W: np.ndarray = None
    M2: np.ndarray = None
    n: int = 0
    history: list = field(default_factory=list)

    def update(self, x):
        r = FEATURES[self.feature](x)
        if self.W is None:
            self.W = np.zeros_like(r)
            self.M2 = np.zeros_like(r)
        self.n += 1
        eta = 1. / self.n
        delta = r - self.W
        self.W = self.W + eta * delta                       # Hebb: move toward observed co-activation
        self.M2 = self.M2 + delta * (r - self.W)            # spread of co-activations (Welford)
        return self

    def fit(self, images, log_every=0):
        for i, x in enumerate(images):
            self.update(x)
            if log_every and (i + 1) % log_every == 0:
                self.history.append(self.W.copy())
        return self

    @property
    def sd(self):
        return np.sqrt(self.M2 / max(self.n - 1, 1)) + 1e-9

    def score(self, x):
        r = FEATURES[self.feature](x)
        return -float(np.mean(((r - self.W) / self.sd) ** 2))

    def deviation(self, x):
        return (FEATURES[self.feature](x) - self.W) / self.sd


def auc(pos, neg):
    """P(score of a real image > score of a control), ties count half."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + .5 * eq)
