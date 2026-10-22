"""Audit final control images; default is an offline synthetic diagnostic."""
import argparse
import json
from pathlib import Path
import numpy as np
import sieve as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', help='Optional photographs; otherwise synthetic diagnostic only')
    ap.add_argument('--output', default='results/astra/control_audit.json')
    args = ap.parse_args()
    if args.folder:
        images = S.crops(S.load_folder(args.folder, max_images=4), 2, seed=221026)
        source = 'folder diagnostic (up to four images, two crops each)'
    else:
        y, x = np.mgrid[:64, :64]
        images = [.2+.4*(x > y)+.2*((x-35)**2+(y-25)**2 < 100),
                  .5+.2*np.cos(x*.4)*np.cos(y*.2),
                  .3+.4*((x//8+y//8) % 2)]
        source = 'three synthetic fixtures; not the photograph benchmark'
    rng = np.random.default_rng(221026)
    rows = [{name: S.control_errors(img, make(img, rng)) for name, make in S.CONTROLS.items()}
            for img in images]
    out = {'source': source, 'definition': 'relative L2 Fourier amplitude and sorted reanalysed coefficients',
           'rows': rows}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
