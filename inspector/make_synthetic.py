"""A tiny LOCO-shaped dataset for smoke tests (not evidence of anything on real data).

A tray with two compartments: a round fruit on the left, a square box on the right, three dots along the
top. Rule: the fruit and the box have the same colour (red or green), and there are always three dots.
Logical anomalies: colours differ, or a dot is missing. Structural anomaly: a dark scratch.

    python -m inspector.make_synthetic out_dir"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLORS = {"red": (200, 50, 45), "green": (60, 160, 70)}


def scene(rng, mismatch=False, missing_dot=False, scratch=False, size=256):
    im = Image.new("RGB", (size, size), (225, 222, 214))
    d = ImageDraw.Draw(im)
    gt = Image.new("L", (size, size), 0)
    g = ImageDraw.Draw(gt)
    d.rectangle([12, 60, 244, 236], outline=(90, 80, 70), width=6)
    d.line([128, 60, 128, 236], fill=(90, 80, 70), width=6)
    jit = lambda: rng.integers(-8, 9)
    c1 = rng.choice(list(COLORS))
    c2 = ([c for c in COLORS if c != c1][0]) if mismatch else c1
    shade = lambda c: tuple(int(np.clip(v + rng.integers(-15, 16), 0, 255)) for v in COLORS[c])
    cx, cy = 70 + jit(), 148 + jit()
    d.ellipse([cx - 32, cy - 32, cx + 32, cy + 32], fill=shade(c1))
    bx, by = 186 + jit(), 148 + jit()
    d.rectangle([bx - 30, by - 30, bx + 30, by + 30], fill=shade(c2))
    if mismatch:
        g.rectangle([bx - 32, by - 32, bx + 32, by + 32], fill=255)
    gone = rng.integers(3) if missing_dot else -1
    for i, x in enumerate([60, 128, 196]):
        x += jit() // 2
        if i == gone:
            g.ellipse([x - 14, 18, x + 14, 46], fill=255)
            continue
        d.ellipse([x - 10, 22, x + 10, 42], fill=(40, 40, 50))
    if scratch:
        x0, y0 = rng.integers(40, 200), rng.integers(80, 220)
        ang, length = rng.uniform(0, np.pi), rng.uniform(30, 50)
        x1, y1 = x0 + length * np.cos(ang), y0 + length * np.sin(ang)
        d.line([x0, y0, x1, y1], fill=(30, 25, 25), width=4)
        g.line([x0, y0, x1, y1], fill=255, width=8)
    a = np.asarray(im).astype(np.float32) + rng.normal(0, 4, (size, size, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), gt


def main(out, cat="tray", seed=0, n=(80, 30, 30, 30, 20)):
    rng = np.random.default_rng(seed)
    root = Path(out) / cat
    plan = [("train/good", n[0], {}), ("validation/good", n[1], {}), ("test/good", n[2], {}),
            ("test/logical_anomalies", n[3], "logical"), ("test/structural_anomalies", n[4], {"scratch": True})]
    for folder, count, kw in plan:
        (root / folder).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            if kw == "logical":
                kwi = {"mismatch": True} if i % 2 == 0 else {"missing_dot": True}
            else:
                kwi = kw
            im, gt = scene(rng, **kwi)
            im.save(root / folder / f"{i:03d}.png")
            if folder.startswith("test/") and folder != "test/good":
                kind = folder.split("/")[1]
                gd = root / "ground_truth" / kind / f"{i:03d}"
                gd.mkdir(parents=True, exist_ok=True)
                gt.save(gd / "000.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic_loco")
