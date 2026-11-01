"""Validation of the fitter, stencils, null bases and the full pipeline on synthetic data
with a known answer."""
import numpy as np
from scipy import optimize, stats
import bfa

rng = np.random.default_rng(12345)
ok = True

def check(name, cond):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name)
    ok &= bool(cond)

# 1. stencils: exact on polynomials of the advertised degree
W = bfa.stencils()
for name, deg in [('O4', 1), ('O6', 2), ('O8', 3)]:
    w = W[name]
    for p in range(deg + 1):
        target = 1.0 if p == 0 else 0.0
        check(f'{name} w.x^{p} = {target}', abs(w @ bfa.X ** p - target) < 1e-12)
check('O4 support f=1,2', np.all(W['O4'][2:] == 0))
check('O6 support f=1,2,4', W['O6'][3] == 0)
print('weights:', {k: v.tolist() for k, v in W.items()})
NB = bfa.null_bases()
for k, Nb in NB.items():
    P = 2 if k == 'O4' else 3
    for p in range(P):
        check(f'null {k} annihilates x^{p}', np.allclose(Nb.T @ bfa.X ** p, 0, atol=1e-14))
    check(f'null {k} rank', np.linalg.matrix_rank(Nb) == Nb.shape[1])
print('null bases:', {k: v.T.tolist() for k, v in NB.items()})

# 2. fitter on noiseless data
t = np.arange(10, 19.)
cov = 0.01 ** 2 * (0.5 * np.eye(9) + 0.5 * np.exp(-np.abs(t[:, None] - t[None]) / 3))
R = 1.03 * np.exp(-0.0123 * t)
r = bfa.fit_exp(R[None], t, cov)
check('noiseless fit exact', abs(r.m[0] - 0.0123) < 1e-12 and abs(r.A[0] - 1.03) < 1e-12 and r.conv[0])

# 3. fitter vs scipy on noisy correlated data, many vectors; also starting far away
L = np.linalg.cholesky(cov)
Rn = R[None] + (L @ rng.standard_normal((9, 500))).T
r = bfa.fit_exp(Rn, t, cov)
check('all 500 noisy fits converged', r.conv.all())
Linv = np.linalg.inv(L)
maxd = 0
for i in range(0, 500, 25):
    f = lambda p: Linv @ (Rn[i] - p[0] * np.exp(-p[1] * t))
    s = optimize.least_squares(f, [1, 0], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    maxd = max(maxd, abs(s.x[1] - r.m[i]) / r.sig_m[i])
check(f'agreement with scipy least_squares (max |dm|/sigma = {maxd:.2e})', maxd < 1e-6)
pull = (r.m - 0.0123) / r.sig_m
check(f'pulls mean {pull.mean():.3f} sd {pull.std():.3f}', abs(pull.mean()) < 0.15 and abs(pull.std() - 1) < 0.1)
chi = r.chi2
check(f'chi2 mean {chi.mean():.2f} (dof 7)', abs(chi.mean() - 7) < 0.5)
r2 = bfa.fit_exp(Rn, t, cov, p0=np.array([3.0, -0.2]))
check('bad start converges to same minimum', r2.conv.all() and np.allclose(r2.m, r.m, atol=1e-12))

# 4. end-to-end synthetic ensemble: per-config correlators with known mu and field-even shifts,
#    field-odd cubic term, and gauge noise shared across fields (as in the real data).
def synth(N, y0, c1, c3, Ntime=32, M=0.6, noise=0.25, seed=0):
    g = np.random.default_rng(seed)
    tt = np.arange(Ntime)
    C = np.empty((N, 4, 2, Ntime, 2))  # cfg, field, tag(0=+B,1=-B), t, comp
    # common gauge fluctuation (correlated in t) + small field-dependent part
    base = noise * np.cumsum(g.standard_normal((N, Ntime)), axis=1) / np.sqrt(Ntime)
    for fi, f in enumerate(bfa.FIELDS):
        eta = bfa.ETA0 * f
        # m(B) = 12 eta y(f), y(f) = y0 + c1 x + c3 x^2  -> spin splitting dE = m/4 per spin
        yf = y0 + c1 * f ** 2 + c3 * f ** 4
        mB = 12 * eta * yf
        even = 0.3 * eta ** 2 * 1e3   # field-even shift
        for si, s in enumerate([+1, -1]):
            E11 = M + even + s * mB / 4   # chosen so that R ~ exp(-m(B) t) with m(B) = 12 eta y
            E22 = M + even - s * mB / 4
            loc = 0.05 * g.standard_normal((N, Ntime, 2))
            C[:, fi, si, :, 0] = np.exp(-E11 * tt)[None] * np.exp(base + loc[..., 0])
            C[:, fi, si, :, 1] = np.exp(-E22 * tt)[None] * np.exp(base + loc[..., 1])
    return C

y0, c1, c3 = 1.7, -2e-3, 1e-5
C = synth(800, y0, c1, c3)
mean, reps = bfa.jk_means(C)
yc = np.empty(4); yr = np.empty((800, 4)); nfail = 0
tt = np.arange(32)
for fi, f in enumerate(bfa.FIELDS):
    Rc = bfa.double_ratio(mean[fi, 0, :, 0], mean[fi, 0, :, 1], mean[fi, 1, :, 0], mean[fi, 1, :, 1])
    Rr = bfa.double_ratio(reps[:, fi, 0, :, 0], reps[:, fi, 0, :, 1], reps[:, fi, 1, :, 0], reps[:, fi, 1, :, 1])
    fw = bfa.fit_window(Rc, Rr, tt, 10, 18)
    nfail += fw['nfail'] + (not fw['conv'])
    yc[fi] = bfa.m_to_y(fw['m'], f); yr[:, fi] = bfa.m_to_y(fw['m_reps'], f)
truth_y = y0 + c1 * bfa.X + c3 * bfa.X ** 2
print('synthetic y(f):', yc, 'truth', truth_y, 'SE', bfa.jk_se(yr))
check('synthetic: no fit failures', nfail == 0)
for name in ['O2', 'O4', 'O6', 'O8']:
    w = W[name]
    est = w @ yc; se = bfa.jk_se(yr @ w)
    print(f'  {name}: {est:.6f} +- {se:.6f}  (truth y0={y0}, w.truth={w @ truth_y:.6f})')
    check(f'synthetic {name} matches w.truth within 3 SE', abs(est - w @ truth_y) < 3 * se)
check('synthetic O6 recovers y0 within 3 SE', abs(W['O6'] @ yc - y0) < 3 * bfa.jk_se(yr @ W['O6']))
T2, p = bfa.hotelling(NB['O6'], yc, yr)
print(f'  O6 null test T2={T2:.3f} p={p:.3f} (true y quadratic in x: null holds)')
T2_4, p4 = bfa.hotelling(NB['O4'], yc, yr)
print(f'  O4 null test T2={T2_4:.3f} p={p4:.3g} (c3 x^2 term present: null false)')

# 5. jackknife SE of a mean agrees with the textbook SE
z = rng.standard_normal(1000)
_, zr = bfa.jk_means(z)
check('jackknife SE of mean == sd/sqrt(N)', abs(bfa.jk_se(zr) - z.std(ddof=1) / np.sqrt(1000)) < 1e-12)

# 6. BLUE: constraint satisfaction and optimality vs fixed stencil
S = np.cov(rng.standard_normal((4, 50)))
for P in (1, 2, 3):
    w = bfa.blue_weights(S, P)
    check(f'BLUE P={P} constraints', np.allclose([w @ bfa.X ** p for p in range(P)], np.eye(P)[0]))
w6 = bfa.blue_weights(S, 3)
check('BLUE-O6 variance <= fixed O6 variance', w6 @ S @ w6 <= W['O6'] @ S @ W['O6'] + 1e-15)

print('\nALL PASS' if ok else '\nSOME CHECKS FAILED')

# 7. multi-seed pull study of the full pipeline (unbiasedness and SE calibration)
pulls = {k: [] for k in ['O2', 'O4', 'O6', 'O8']}; nf = 0; T2s = []
for seed in range(1, 41):
    C = synth(400, y0, c1, c3, seed=seed)
    mean, reps = bfa.jk_means(C)
    yc = np.empty(4); yr = np.empty((400, 4))
    for fi, f in enumerate(bfa.FIELDS):
        Rc = bfa.double_ratio(mean[fi, 0, :, 0], mean[fi, 0, :, 1], mean[fi, 1, :, 0], mean[fi, 1, :, 1])
        Rr = bfa.double_ratio(reps[:, fi, 0, :, 0], reps[:, fi, 0, :, 1], reps[:, fi, 1, :, 0], reps[:, fi, 1, :, 1])
        fw = bfa.fit_window(Rc, Rr, tt, 10, 18)
        nf += fw['nfail'] + (not fw['conv'])
        yc[fi] = bfa.m_to_y(fw['m'], f); yr[:, fi] = bfa.m_to_y(fw['m_reps'], f)
    truth_y = y0 + c1 * bfa.X + c3 * bfa.X ** 2
    for k in pulls:
        pulls[k].append((W[k] @ yc - W[k] @ truth_y) / bfa.jk_se(yr @ W[k]))
    T2s.append(bfa.hotelling(NB['O6'], yc, yr)[0])
print('multi-seed (40 ensembles x 400 cfgs): fit failures', nf)
for k, v in pulls.items():
    v = np.array(v)
    print(f'  {k}: pull mean {v.mean():+.3f} +- {v.std(ddof=1)/np.sqrt(len(v)):.3f}, pull sd {v.std(ddof=1):.3f}')
    check(f'multi-seed {k} pull mean consistent with 0', abs(v.mean()) < 3 * v.std(ddof=1) / np.sqrt(len(v)) + 0.05)
print(f'  O6 null-test T2 mean {np.mean(T2s):.3f} (expect ~1 under the null)')
print('\nALL PASS' if ok else '\nSOME CHECKS FAILED')
