"""The Sihti sieve and the three kinds of 'same ingredients, different structure' controls.

Sieve: repeated linear diffusion x <- x + a * Laplacian(x) on a torus (wrap-around
boundary). Saving the departures between octave depths gives bands

    r_k = x_{d_k} - x_{d_{k+1}},   d = 0, 1, 2, 4, 8, 16, 32,   core = x_32,
    x = sum_k r_k + core   exactly.

The torus matters: with it the sieve is a convolution, so every second-order
statistic of every band (energy, signed correlation between bands at any offset)
is fixed by the image's power spectrum alone.

Controls (all keep the image's power spectrum; they differ in what else they keep):
  phase    : random Fourier phases. Every second-order statistic identical, all
             higher-order structure gone. Freeman et al. 2013's spectrally matched noise.
  marginal : phase-scrambled, then alternately each band's coefficient histogram is
             matched to the real image's band and the amplitude spectrum re-imposed
             (Heeger & Bergen 1995). Band energies AND band histograms ~matched;
             what is left different is how coefficients co-occur across scale and space.
  shuffle  : 16x16 blocks of the image permuted. Local pieces kept, arrangement broken.
             Caveat: block seams are new edges.
"""
import numpy as np
from scipy.ndimage import laplace

DEPTHS = (0, 1, 2, 4, 8, 16, 32)
A = .2


def sieve(x, depths=DEPTHS, a=A):
    """Bands r_0..r_{K-1} (finest first) and the core. sum(bands) + core == x."""
    lv, cur, d = [], np.asarray(x, float), 0
    for t in depths:
        while d < t:
            cur = cur + a * laplace(cur, mode="wrap")
            d += 1
        lv.append(cur.copy())
    return [lv[i] - lv[i + 1] for i in range(len(lv) - 1)], lv[-1]


def unsieve(bands, core):
    return np.sum(bands, axis=0) + core


# ---------------------------------------------------------------- controls
def phase_scramble(x, rng):
    F = np.fft.fft2(x)
    phase = np.angle(np.fft.fft2(rng.standard_normal(x.shape)))   # Hermitian-symmetric random phase
    G = np.abs(F) * np.exp(1j * phase)
    G[0, 0] = F[0, 0]
    return np.real(np.fft.ifft2(G))


def _match_hist(src, ref):
    out = np.empty_like(src)
    out.ravel()[np.argsort(src.ravel())] = np.sort(ref.ravel())
    return out


def marginal_matched(x, rng, iters=8):
    """Heeger-Bergen style: band histograms and amplitude spectrum both ~ the real image's."""
    ref_bands, ref_core = sieve(x)
    amp = np.abs(np.fft.fft2(x))
    y = phase_scramble(x, rng)
    for _ in range(iters):
        bands, core = sieve(y)
        bands = [_match_hist(b, rb) for b, rb in zip(bands, ref_bands)]
        y = unsieve(bands, _match_hist(core, ref_core))
        F = np.fft.fft2(y)
        y = np.real(np.fft.ifft2(amp * np.exp(1j * np.angle(F))))
    bands, core = sieve(y)                       # finish on the histograms
    return unsieve([_match_hist(b, rb) for b, rb in zip(bands, ref_bands)], _match_hist(core, ref_core))


def block_shuffle(x, rng, block=16):
    n = x.shape[0] // block
    tiles = [x[i*block:(i+1)*block, j*block:(j+1)*block] for i in range(n) for j in range(n)]
    order = rng.permutation(len(tiles))
    while n > 1 and np.all(order == np.arange(len(tiles))):
        order = rng.permutation(len(tiles))
    rows = [np.concatenate([tiles[order[i*n + j]] for j in range(n)], 1) for i in range(n)]
    return np.concatenate(rows, 0)


CONTROLS = {"phase": phase_scramble, "marginal": marginal_matched, "shuffle": block_shuffle}


# ---------------------------------------------------------------- data
TRAIN_PHOTOS = ["astronaut", "chelsea", "coins", "moon", "rocket", "immunohistochemistry",
                "hubble_deep_field", "grass", "gravel", "brick", "retina", "page", "text",
                "colorwheel", "microaneurysms"]
TEST_PHOTOS = ["camera", "coffee", "clock"]


def photo(name):
    import skimage.data
    from skimage.color import rgb2gray, rgba2rgb
    im = getattr(skimage.data, name)()
    if im.ndim == 3 and im.shape[2] == 4:
        im = rgba2rgb(im)
    if im.ndim == 3:
        im = rgb2gray(im)
    im = np.asarray(im, float)
    return im / 255. if im.max() > 1.5 else im


def crops(images, n_per, size=64, seed=0, min_std=.05, scales=(.35, .5, .75)):
    from skimage.transform import rescale
    rng = np.random.default_rng(seed)
    out = []
    for img in images:
        scaled = [rescale(img, s, anti_aliasing=True) for s in scales]
        got, tries = 0, 0
        while got < n_per and tries < 5000:
            tries += 1
            s = scaled[int(rng.integers(len(scaled)))]
            if min(s.shape) < size:
                continue
            y = int(rng.integers(0, s.shape[0] - size + 1))
            x = int(rng.integers(0, s.shape[1] - size + 1))
            c = s[y:y + size, x:x + size]
            if c.std() >= min_std:
                out.append(c.astype(float))
                got += 1
    return np.array(out)


def load_folder(folder, max_images=None):
    from pathlib import Path
    from PIL import Image
    root = Path(folder).expanduser().resolve()
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    if max_images:
        paths = paths[:max_images]
    out = []
    for p in paths:
        with Image.open(p) as im:
            g = im.convert("L")
            g.thumbnail((512, 512))
            out.append(np.asarray(g, float) / 255.)
    if not out:
        raise ValueError(f"no images under {root}")
    return out
