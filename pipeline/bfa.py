"""Background-field analysis core: jackknife, double ratio, correlated exp fitter,
field-space stencils, null-space Hotelling tests and BLUE weights.

Written from SPEC.md only (clean room)."""
import numpy as np
from scipy import stats, linalg, optimize

FIELDS = np.array([1, 2, 4, 8], dtype=float)
X = FIELDS ** 2                      # x = f^2 = 1, 4, 16, 64
ETA0 = 0.001364
MU_N_FACTOR = 0.88658


# --------------------------------------------------------------------------------------
# Jackknife
# --------------------------------------------------------------------------------------
def jk_means(C):
    """Delete-one jackknife replicas of the mean along axis 0.  Returns (mean, replicas)."""
    N = C.shape[0]
    tot = C.sum(axis=0)
    mean = tot / N
    reps = (tot[None] - C) / (N - 1)
    return mean, reps


def jk_cov(reps):
    """Jackknife covariance of replica rows: (N-1)/N * sum_j (r_j - rbar)(r_j - rbar)^T.
    reps: (N, d) -> (d, d)."""
    N = reps.shape[0]
    d = reps - reps.mean(axis=0)
    return (N - 1) / N * d.T @ d


def jk_se(reps):
    """Jackknife SE along axis 0 (any trailing shape)."""
    N = reps.shape[0]
    return np.sqrt((N - 1) / N * ((reps - reps.mean(axis=0)) ** 2).sum(axis=0))


# --------------------------------------------------------------------------------------
# Double ratio
# --------------------------------------------------------------------------------------
def double_ratio(G11p, G22p, G11m, G22m):
    """R = G11(+B) G22(-B) / [G11(-B) G22(+B)].  'p' = physical +B, 'm' = physical -B."""
    return (G11p * G22m) / (G11m * G22p)


# --------------------------------------------------------------------------------------
# Correlated fit of R(t) = A exp(-m t) with a fixed covariance.
# Vectorised Levenberg-Marquardt over many data vectors (replicas) sharing one covariance.
# --------------------------------------------------------------------------------------
class FitResult:
    pass


def _gn_step(J0, J1, r):
    a00 = (J0 * J0).sum(1); a01 = (J0 * J1).sum(1); a11 = (J1 * J1).sum(1)
    b0 = (J0 * r).sum(1); b1 = (J1 * r).sum(1)
    det = a00 * a11 - a01 ** 2
    return np.sqrt(a11 / det), np.sqrt(a00 / det), (a11 * b0 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det


def fit_exp(R, t, cov, p0=None, maxit=200, tol_sig=1e-6):
    """R: (M, nt) data vectors, t: (nt,), cov: (nt, nt) held fixed.
    p0: (M, 2) or (2,) initial (A, m).  Returns FitResult with A, m, chi2, conv (bool), nit.

    Convergence (checked per vector, and re-checked at exit): the full Gauss-Newton step from
    the final point is smaller than tol_sig times the parameter standard error."""
    R = np.atleast_2d(np.asarray(R, float))
    M, nt = R.shape
    t = np.asarray(t, float)
    L = np.linalg.cholesky(cov)                       # raises if not PD
    Linv = linalg.solve_triangular(L, np.eye(nt), lower=True)
    d = R @ Linv.T                                    # whitened data, (M, nt)

    if p0 is None:
        # unweighted log-linear start
        p0 = np.empty((M, 2))
        lr = np.log(np.abs(R))
        tt = t - t.mean()
        slope = (lr * tt).sum(1) / (tt ** 2).sum()
        p0[:, 1] = -slope
        p0[:, 0] = np.exp(lr.mean(1) - slope * t.mean()) * np.sign(R.mean(1))
    p = np.array(np.broadcast_to(p0, (M, 2)), float)

    def resid_jac(p):
        e = np.exp(-np.outer(p[:, 1], t))             # (M, nt)
        g = p[:, :1] * e
        r = d - g @ Linv.T
        J0 = e @ Linv.T                               # d g / dA  (whitened)
        J1 = (-t[None] * g) @ Linv.T                  # d g / dm
        return r, J0, J1

    r, J0, J1 = resid_jac(p)
    chi2 = (r ** 2).sum(1)
    lam = np.full(M, 1e-6)
    conv = np.zeros(M, bool)
    nit = np.zeros(M, int)
    for it in range(maxit):
        act = ~conv
        if not act.any():
            break
        a00 = (J0 * J0).sum(1); a01 = (J0 * J1).sum(1); a11 = (J1 * J1).sum(1)
        b0 = (J0 * r).sum(1); b1 = (J1 * r).sum(1)
        A00 = a00 * (1 + lam); A11 = a11 * (1 + lam)
        det = A00 * A11 - a01 ** 2
        dA = (A11 * b0 - a01 * b1) / det
        dm = (A00 * b1 - a01 * b0) / det
        pn = p.copy()
        pn[act, 0] += dA[act]; pn[act, 1] += dm[act]
        rn, J0n, J1n = resid_jac(pn)
        chi2n = (rn ** 2).sum(1)
        acc = act & (chi2n <= chi2 * (1 + 1e-15) + 1e-300)
        p[acc] = pn[acc]; r[acc] = rn[acc]; J0[acc] = J0n[acc]; J1[acc] = J1n[acc]
        chi2[acc] = chi2n[acc]
        lam[acc] = np.maximum(lam[acc] / 10, 1e-12)
        lam[act & ~acc] *= 10
        nit[act] += 1
        # convergence: the undamped Gauss-Newton step at the current point is < tol_sig
        # in units of the parameter standard errors (inverse-Hessian), i.e. we sit at the
        # chi^2 minimum to far better precision than any statistical scale.
        sdA, sdm, dA0, dm0 = _gn_step(J0, J1, r)
        conv |= act & (np.abs(dA0) < tol_sig * sdA) & (np.abs(dm0) < tol_sig * sdm)
    res = FitResult()
    res.A = p[:, 0]; res.m = p[:, 1]; res.chi2 = chi2; res.conv = conv; res.nit = nit
    res.dof = nt - 2
    sdA, sdm, dA0, dm0 = _gn_step(J0, J1, r)
    res.gradmax = np.maximum(np.abs(dA0) / sdA, np.abs(dm0) / sdm)   # residual Newton step / sigma
    res.conv = conv & (res.gradmax < tol_sig)
    # parameter errors from the (fixed-covariance) Hessian, for the central fit
    a00 = (J0 * J0).sum(1); a01 = (J0 * J1).sum(1); a11 = (J1 * J1).sum(1)
    det = a00 * a11 - a01 ** 2
    res.sig_m = np.sqrt(a00 / det)
    return res


def fit_window(Rc, Rreps, t_all, t0, t1):
    """Central + jackknife-replica fits of R over window t0..t1 (inclusive) with the
    jackknife covariance computed once and held fixed.  Returns dict."""
    sl = slice(t0, t1 + 1)
    t = np.asarray(t_all, float)[sl]
    Rw = Rc[sl]; Rr = Rreps[:, sl]
    cov = jk_cov(Rr)
    ev = np.linalg.eigvalsh(cov)
    c = fit_exp(Rw[None], t, cov)
    p0 = np.array([c.A[0], c.m[0]])
    r = fit_exp(Rr, t, cov, p0=p0)
    # Fallback for any replica not converged at tol 1e-6 sigma: refit with scipy least_squares
    # (trust-region, independent code path) and accept only if the residual Newton step is
    # < 1e-4 sigma.  Both counts are reported.
    nfail_primary = int((~r.conv).sum())
    m_reps = r.m.copy(); conv_reps = r.conv.copy()
    if nfail_primary:
        L = np.linalg.cholesky(cov)
        Linv = linalg.solve_triangular(L, np.eye(len(t)), lower=True)
        for i in np.where(~r.conv)[0]:
            fun = lambda p, i=i: Linv @ (Rr[i] - p[0] * np.exp(-p[1] * t))
            s = optimize.least_squares(fun, p0, xtol=1e-15, ftol=1e-15, gtol=1e-15, method='lm')
            chk = fit_exp(Rr[i:i + 1], t, cov, p0=s.x[None], maxit=0)
            if s.success and chk.gradmax[0] < 1e-4:
                m_reps[i] = s.x[1]; conv_reps[i] = True
    return dict(m=c.m[0], A=c.A[0], chi2=c.chi2[0], dof=c.dof, conv=bool(c.conv[0]),
                sig_m_hess=c.sig_m[0],
                m_reps=m_reps, conv_reps=conv_reps, nfail=int((~conv_reps).sum()), nfail_primary=nfail_primary,
                gradmax=max(c.gradmax[0], r.gradmax.max()),
                cov_mineig=ev.min(), cov_cond=ev.max() / ev.min())


# --------------------------------------------------------------------------------------
# Field space
# --------------------------------------------------------------------------------------
def lagrange_x0_weights(xs):
    """Weights w_i with sum_i w_i p(x_i) = p(0) for every polynomial of degree < len(xs):
    w_i = prod_{j!=i} (0 - x_j) / (x_i - x_j)."""
    xs = np.asarray(xs, float)
    w = np.ones(len(xs))
    for i in range(len(xs)):
        for j in range(len(xs)):
            if j != i:
                w[i] *= (0 - xs[j]) / (xs[i] - xs[j])
    return w


def stencils():
    W = {}
    W['O2'] = np.array([1., 0, 0, 0])
    W['O4'] = np.r_[lagrange_x0_weights(X[:2]), 0, 0]
    W['O6'] = np.r_[lagrange_x0_weights(X[:3]), 0]
    W['O8'] = lagrange_x0_weights(X)
    return W


def null_bases():
    """Columns span {n : n.x^p = 0, p < P} on the four fields.
    O6 (P=3, 1-dim): third divided-difference coefficients n_i = 1/prod_{j!=i}(x_i-x_j).
    O4 (P=2, 2-dim): the O6 vector plus the second divided difference on f=1,2,4 (zero on f=8)."""
    def dd(xs):
        xs = np.asarray(xs, float)
        return np.array([1 / np.prod([xs[i] - xs[j] for j in range(len(xs)) if j != i])
                         for i in range(len(xs))])
    n3 = dd(X)                      # annihilates degree <= 2
    n2 = np.r_[dd(X[:3]), 0]        # annihilates degree <= 1
    N4 = np.column_stack([n2, n3])
    N6 = n3[:, None]
    return {'O4': N4, 'O6': N6}


def hotelling(Nb, y_c, y_reps):
    """T^2 = u^T V^-1 u, u = N^T y (central), V = jackknife cov of N^T y_j.
    F = T^2 (J-q)/(q(J-1)) ~ F(q, J-q)."""
    u = Nb.T @ y_c
    U = y_reps @ Nb
    V = jk_cov(U)
    T2 = float(u @ np.linalg.solve(V, u))
    J = y_reps.shape[0]; q = Nb.shape[1]
    F = T2 * (J - q) / (q * (J - 1))
    p = float(stats.f.sf(F, q, J - q))
    return T2, p


def blue_weights(Sigma, P):
    A = np.column_stack([X ** p for p in range(P)])
    SiA = np.linalg.solve(Sigma, A)
    M = A.T @ SiA
    e1 = np.zeros(P); e1[0] = 1
    return SiA @ np.linalg.solve(M, e1)


def m_to_y(m, f):
    return m / (12 * ETA0 * f)
