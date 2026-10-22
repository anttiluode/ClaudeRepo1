"""MVTec LOCO AD layout:  <root>/<category>/{train/good, validation/good, test/{good,logical_anomalies,
structural_anomalies}, ground_truth/{logical_anomalies,structural_anomalies}/<image id>/*.png}.
A test image's ground truth may be several mask files; their union is used."""
from pathlib import Path
import numpy as np
from PIL import Image

CATEGORIES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
KINDS = ["good", "logical_anomalies", "structural_anomalies"]


def split(root, category):
    d = Path(root) / category
    train = sorted((d / "train" / "good").glob("*.png"))
    val = sorted((d / "validation" / "good").glob("*.png"))
    test = [(p, k) for k in KINDS for p in sorted((d / "test" / k).glob("*.png"))]
    if not train or not val or not test:
        raise FileNotFoundError(f"{d}: expected train/good, validation/good and test/* with .png images")
    return train, val, test


def load(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def gt_mask(root, category, path, kind, size):
    """Union of the ground-truth masks for one test image, resized to size x size (bool), or None."""
    if kind == "good":
        return None
    folder = Path(root) / category / "ground_truth" / kind / Path(path).stem
    files = sorted(folder.glob("*.png"))
    if not files:
        return None
    m = np.zeros((size, size), bool)
    for f in files:
        m |= np.asarray(Image.open(f).convert("L").resize((size, size), Image.NEAREST)) > 0
    return m
