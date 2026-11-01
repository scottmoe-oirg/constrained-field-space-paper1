"""Hard-leverage frontier (paper Table VII, Figs. 2-3).

Minimize w^T C w subject to the O4 constraints sum(w)=1, sum(w x)=0 and
|L4| = |w.x^2| <= 4 tau, |L6| = |w.x^3| <= 20 tau, for tau in TAUS and unrestricted.
C is the field-space jackknife covariance with the temporal covariance held fixed
(the production convention), and the weights are held fixed in the jackknife.

Input: per-ensemble files <data>/nested_{P,N}{k}.npz with y (central responses, 4 fields)
and C (fixed-temporal-covariance jackknife covariance).
"""
import argparse, itertools, numpy as np, pandas as pd

X = np.array([1., 4., 16., 64.])
W4 = np.array([4/3, -1/3, 0, 0])
NULL = np.array([[4., 20.], [-5., -21.], [1., 0.], [0., 1.]])   # O4 null space
TAUS = [1, 2, 5, 10, 30, 100, 300, 1000, None]                  # None = unrestricted

# w = W4 + NULL c ; (L4, L6) = L0 + M c  ->  c = M^{-1}(L - L0): parametrize by L = (L4, L6)
L0 = np.array([W4 @ X**2, W4 @ X**3])                            # (-4, -20)
M = np.array([NULL.T @ X**2, NULL.T @ X**3])                     # 2x2
Minv = np.linalg.inv(M)
B = NULL @ Minv                                                  # w = W4 + B (L - L0)

def weights(L): return W4 + B @ (np.asarray(L) - L0)

def solve(C, tau):
    """Exact 2-D box-constrained QP over L=(L4,L6): check interior optimum, 4 edges, 4 corners."""
    Q = B.T @ C @ B; g = B.T @ C @ (W4 - B @ L0)                 # var(L) = L^T Q L + 2 g^T L + const
    var = lambda L: float(weights(L) @ C @ weights(L))
    Lfree = np.linalg.solve(Q, -g)
    if tau is None: return Lfree
    lim = np.array([4.0 * tau, 20.0 * tau])
    if np.all(np.abs(Lfree) <= lim): return Lfree
    cands = []
    for i in (0, 1):                                             # fix coordinate i at +/- lim[i], optimize other
        j = 1 - i
        for s in (-1, 1):
            Li = s * lim[i]
            Lj = -(g[j] + Q[j, i] * Li) / Q[j, j]
            Lj = np.clip(Lj, -lim[j], lim[j])
            L = np.empty(2); L[i] = Li; L[j] = Lj; cands.append(L)
    for s0, s1 in itertools.product((-1, 1), repeat=2):
        cands.append(np.array([s0 * lim[0], s1 * lim[1]]))
    return min(cands, key=var)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='../../data/derived')
    ap.add_argument('--out', default='frontier.csv')
    ap.add_argument('--cov', choices=['fixed', 'nested'], default='fixed',
                    help='fixed: temporal covariance held fixed (production); nested: noise-inclusive (nested jackknife)')
    a = ap.parse_args()
    rows = []
    for p in 'PN':
        for k in range(6):
            d = np.load(f'{a.data}/nested_{p}{k}.npz'); y, C = d['y'], (d['C'] if a.cov == 'fixed' else d['Cn'])
            s4 = np.sqrt(W4 @ C @ W4)
            for tau in TAUS:
                L = solve(C, tau); w = weights(L); dlt = w - W4
                rows.append(dict(particle=p, k=k, tau='unrestricted' if tau is None else tau,
                                 L4=L[0], L6=L[1], Lambda=max(abs(L[0]) / 4, abs(L[1]) / 20),
                                 se_ratio=np.sqrt(w @ C @ w) / s4,
                                 z_disp=(dlt @ y) / np.sqrt(dlt @ C @ dlt) if np.any(np.abs(dlt) > 1e-12) else 0.0,
                                 w1=w[0], w2=w[1], w3=w[2], w4=w[3]))
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    med = df.groupby(['tau', 'particle'], sort=False)['se_ratio'].median().unstack()
    print(med.to_string(float_format='%.6f'))

if __name__ == '__main__': main()
