"""Parse OCT_P / OCT_N text files into a numpy cache.
Output: work/cache_OCT_{S}.npz with C[cfg, field_idx(0..3 for f=1,2,4,8), sign(0=MINUS tag,1=PLUS tag), kappa, t, comp], cfg."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import numpy as np, os, glob, sys
DATA = str(OCTET)
WORK = str(WORK)
FIELDS = [1, 2, 4, 8]

def parse(path):
    out = np.full((4, 2, 6, 48, 2), np.nan)
    nfill = 0
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            lab = p[0]
            if 'ZERO' in lab:
                continue
            # OCT_P_ETA04_MINUS_02
            parts = lab.split('_')
            f = int(parts[2][3:]); tag = parts[3]; k = int(parts[4])
            t = int(p[1])
            out[FIELDS.index(f), 0 if tag == 'MINUS' else 1, k, t, 0] = float(p[2])
            out[FIELDS.index(f), 0 if tag == 'MINUS' else 1, k, t, 1] = float(p[3])
            nfill += 1
    return out, nfill

for S in ['P', 'N']:
    files = sorted(glob.glob(f'{DATA}/OCT_{S}/cfly_2448_*.OCT_{S}'))
    cfg = np.array([int(os.path.basename(f).split('_')[2].split('.')[0]) for f in files])
    C = np.empty((len(files), 4, 2, 6, 48, 2))
    for i, f in enumerate(files):
        C[i], n = parse(f)
        assert n == 4 * 2 * 6 * 48, (f, n)
    assert np.isfinite(C).all()
    np.savez(f'{WORK}/cache_OCT_{S}.npz', C=C, cfg=cfg)
    print(S, C.shape, 'neg frac', (C < 0).mean())
