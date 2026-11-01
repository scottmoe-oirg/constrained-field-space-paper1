"""Repackage the raw decuplet correlator text files into the arrays read by the pipeline.

Raw input:  <raw>/DEC_{ST}/DATA/cfly_2448_NNN.DEC_{ST}, lines
            DEC_{ST}_ETA{FF}_{PLUS|MINUS}_{kk}  t  c1 ... c8
            where the eight columns are four complex numbers whose real parts are
            G11(+parity), G22(+parity), G11(-parity), G22(-parity); the raw MINUS tag is physical +B.
Output:     <out>/s_{ST}_k{k}.npz with C[cfg, label, t, comp] (t = 0..31; comp 0 = Re G11, 1 = Re G22,
            positive parity), labels (01_PLUS ... 08_MINUS) and cfg (configuration numbers).
Usage:      python build_decuplet_inputs.py --raw <dir with DEC_DP, DEC_DPP> --out <dir>
"""
import argparse, glob, os, re, numpy as np

LABELS = [f'{f:02d}_{s}' for f in (1, 2, 4, 8) for s in ('PLUS', 'MINUS')]
NT = 32

def build(raw, st, out):
    files = sorted(glob.glob(f'{raw}/DEC_{st}/DATA/cfly_2448_*.DEC_{st}'),
                   key=lambda p: int(re.search(r'_(\d+)\.DEC', p).group(1)))
    pat = re.compile(rf'DEC_{st}_ETA(\d\d)_(PLUS|MINUS)_(\d\d)\s+(\d+)\s+(.*)')
    cfg, arr = [], []
    for p in files:
        A = np.full((6, len(LABELS), NT, 2), np.nan)
        for line in open(p):
            m = pat.match(line)
            if not m: continue
            t = int(m.group(4))
            if t >= NT: continue
            v = np.array(m.group(5).split(), float)
            A[int(m.group(3)), LABELS.index(f'{m.group(1)}_{m.group(2)}'), t] = (v[0], v[2])
        cfg.append(int(re.search(r'_(\d+)\.DEC', p).group(1))); arr.append(A)
    A = np.array(arr)
    if np.isnan(A).any(): raise SystemExit(f'{st}: {int(np.isnan(A).sum())} missing entries')
    for k in range(6):
        np.savez_compressed(f'{out}/s_{st}_k{k}.npz', C=A[:, k], cfg=np.array(cfg), labels=np.array(LABELS))
    print(st, A.shape)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--out', required=True); ap.add_argument('--states', default='DP,DPP,D0,DM')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for st in a.states.split(','): build(a.raw, st, a.out)
