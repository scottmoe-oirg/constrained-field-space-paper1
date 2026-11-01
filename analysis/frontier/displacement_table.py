"""Table of BLUE-O4 minus fixed-O4 displacements (unrestricted O4-constrained BLUE, Sigma_fix and Sigma_nest).
Prints mu_N displacement [marginal sigma of fixed O4; paired sigma] and writes displacement_table.csv.
Usage: displacement_table.py [DERIVED_DIR]   (needs nested_{P,N}{k}.npz, frontier.csv, frontier_nested.csv)"""
import sys, pathlib, numpy as np, pandas as pd
D = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2] / 'data' / 'derived'
CONV = 0.88658; W4 = np.array([4/3, -1/3, 0, 0]); MPI = [338, 362, 384, 405, 444, 693]
rows = []
for p in 'PN':
    for k in range(6):
        z = np.load(D / f'nested_{p}{k}.npz'); y = z['y']; r = dict(particle=p, k=k, mpi=MPI[k])
        for lab, fn, C in (('fix', 'frontier.csv', z['C']), ('nest', 'frontier_nested.csv', z['Cn'])):
            f = pd.read_csv(D / fn); f = f[(f.particle == p) & (f.k == k) & (f.tau.astype(str) == 'unrestricted')].iloc[0]
            w = f[['w1', 'w2', 'w3', 'w4']].to_numpy(float); dw = w - W4
            r[f'd_{lab}'] = CONV * dw @ y; r[f'zmarg_{lab}'] = dw @ y / np.sqrt(W4 @ C @ W4); r[f'zpair_{lab}'] = dw @ y / np.sqrt(dw @ C @ dw)
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv('displacement_table.csv', index=False); print(df.round(3).to_string(index=False))
