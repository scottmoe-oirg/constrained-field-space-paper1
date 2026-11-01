"""Deliverables B, C, D: Delta+ / Delta++ charge-ratio control.
Writes decuplet_ratio.csv, decuplet_blue.csv, decuplet_window_scan.csv, decuplet_y.csv."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import numpy as np, pandas as pd
import bfa

WORK = str(WORK)
DATA = str(DECUPLET)
STATES = ['DP', 'DPP']
MAIN_WINDOWS = [(10, 18), (12, 20)]
SCAN_WINDOWS = [(a, 18) for a in range(8, 15)] + [(a, 20) for a in range(10, 17)] + [(a, 22) for a in range(12, 17)]
ALL_WINDOWS = sorted(set(MAIN_WINDOWS + SCAN_WINDOWS), key=lambda w: (w[1], w[0]))
W = bfa.stencils(); NB = bfa.null_bases()
SIGN = 1.0      # global sign factor fixed by the proton anchor in run_nucleon.py (raw proton y > 0)
LABELS = ['01_PLUS', '01_MINUS', '02_PLUS', '02_MINUS', '04_PLUS', '04_MINUS', '08_PLUS', '08_MINUS']


def load(k):
    d = {S: np.load(f'{DATA}/s_{S}_k{k}.npz') for S in STATES}
    for S in STATES:
        assert list(d[S]['labels']) == LABELS, d[S]['labels']
    common = np.intersect1d(d['DP']['cfg'], d['DPP']['cfg'])
    C = {}
    for S in STATES:
        cfg = d[S]['cfg']
        assert len(np.unique(cfg)) == len(cfg)
        idx = np.searchsorted(cfg, common) if np.all(np.diff(cfg) > 0) else \
            np.array([np.where(cfg == c)[0][0] for c in common])
        assert np.array_equal(cfg[idx], common)
        C[S] = d[S]['C'][idx]
        assert np.isfinite(C[S]).all()
    return common, C


def per_state_y(C, windows):
    """Return yc[win, field], yr[win, N, field], diag list."""
    N = C.shape[0]
    mean, reps = bfa.jk_means(C)                     # [.., label, t, comp]
    tt = np.arange(C.shape[2])
    yc = np.empty((len(windows), 4)); yr = np.empty((len(windows), N, 4)); diag = []
    for fi, f in enumerate(bfa.FIELDS):
        lp = LABELS.index(f'{int(f):02d}_MINUS')     # physical +B
        lm = LABELS.index(f'{int(f):02d}_PLUS')      # physical -B
        Rc = bfa.double_ratio(mean[lp, :, 0], mean[lp, :, 1], mean[lm, :, 0], mean[lm, :, 1])
        Rr = bfa.double_ratio(reps[:, lp, :, 0], reps[:, lp, :, 1], reps[:, lm, :, 0], reps[:, lm, :, 1])
        for wi, (t0, t1) in enumerate(windows):
            fw = bfa.fit_window(Rc, Rr, tt, t0, t1)
            G = np.stack([mean[l, t0:t1 + 1, c] for l in (lp, lm) for c in (0, 1)])
            fw['Gbar_sign_mixed'] = not (np.all(G > 0) or np.all(G < 0))
            fw.update(f=int(f), window=f'{t0}..{t1}')
            diag.append(fw)
            yc[wi, fi] = SIGN * bfa.m_to_y(fw['m'], f)
            yr[wi, :, fi] = SIGN * bfa.m_to_y(fw['m_reps'], f)
    return yc, yr, diag


def ratio_rows(k, N, wname, yc, yr):
    rows = []
    T2 = {S: bfa.hotelling(NB['O6'], yc[S], yr[S]) for S in STATES}
    for name, w in W.items():
        e = {S: w @ yc[S] for S in STATES}
        er = {S: yr[S] @ w for S in STATES}
        r = e['DP'] / e['DPP']; rr = er['DP'] / er['DPP']
        se = bfa.jk_se(rr)
        rows.append(dict(window=wname, kappa_idx=k, Ncfg=N, stencil=name,
                         mu_DP_muN=bfa.MU_N_FACTOR * e['DP'], se_DP_muN=bfa.MU_N_FACTOR * bfa.jk_se(er['DP']),
                         mu_DPP_muN=bfa.MU_N_FACTOR * e['DPP'], se_DPP_muN=bfa.MU_N_FACTOR * bfa.jk_se(er['DPP']),
                         r=r, r_se=se, r_minus_half=r - 0.5, sigma=(r - 0.5) / se,
                         r_jk_mean=rr.mean(),
                         T2_O6_DP=T2['DP'][0], p_O6_DP=T2['DP'][1],
                         T2_O6_DPP=T2['DPP'][0], p_O6_DPP=T2['DPP'][1]))
    return rows


def reopt_weights(yr, P):
    """Per-replica BLUE weights from the jackknife covariance of the other N-1 replicas
    (rank-one downdate of the scatter matrix; scale is irrelevant for the weights)."""
    N = yr.shape[0]
    D = yr - yr.mean(0)
    S = D.T @ D
    Sj = S[None] - (N / (N - 1)) * np.einsum('ji,jk->jik', D, D)
    A = np.column_stack([bfa.X ** p for p in range(P)])
    SiA = np.linalg.solve(Sj, np.broadcast_to(A, (N, 4, P)))        # (N,4,P)
    M = np.einsum('ip,jiq->jpq', A, SiA)
    e1 = np.zeros(P); e1[0] = 1
    lam = np.linalg.solve(M, np.broadcast_to(e1, (N, P))[..., None])[..., 0]
    return np.einsum('jip,jp->ji', SiA, lam)


def blue_rows(k, N, wname, yc, yr):
    rows = []
    Sig = {S: bfa.jk_cov(yr[S]) for S in STATES}
    cond = {S: np.linalg.cond(Sig[S]) for S in STATES}
    mineig = {S: np.linalg.eigvalsh(Sig[S]).min() for S in STATES}
    se6 = {S: bfa.jk_se(yr[S] @ W['O6']) for S in STATES}
    wfix = {S: {P: bfa.blue_weights(Sig[S], P) for P in (1, 2, 3)} for S in STATES}
    for P, ename in [(1, 'BLUE-1'), (2, 'BLUE-O4'), (3, 'BLUE-O6')]:
        for mode in ['fixed', 'reopt']:
            e = {S: wfix[S][P] @ yc[S] for S in STATES}
            if mode == 'fixed':
                er = {S: yr[S] @ wfix[S][P] for S in STATES}
            else:
                er = {S: np.einsum('ji,ji->j', reopt_weights(yr[S], P), yr[S]) for S in STATES}
            r = e['DP'] / e['DPP']; rr = er['DP'] / er['DPP']; se = bfa.jk_se(rr)
            row = dict(window=wname, kappa_idx=k, Ncfg=N, estimator=ename, weights=mode,
                       mu_DP_muN=bfa.MU_N_FACTOR * e['DP'], mu_DPP_muN=bfa.MU_N_FACTOR * e['DPP'],
                       r=r, r_minus_half=r - 0.5, r_se=se, sigma=(r - 0.5) / se)
            for S in STATES:
                se_S = bfa.jk_se(er[S])
                row[f'se_{S}_muN'] = bfa.MU_N_FACTOR * se_S
                row[f'err_ratio_vs_O6_{S}'] = se_S / se6[S]
            for i, f in enumerate(bfa.FIELDS):
                row[f'w_DP_f{int(f)}'] = wfix['DP'][P][i]
            for i, f in enumerate(bfa.FIELDS):
                row[f'w_DPP_f{int(f)}'] = wfix['DPP'][P][i]
            row['cond_Sigma_DP'] = cond['DP']; row['cond_Sigma_DPP'] = cond['DPP']
            row['mineig_Sigma_DP'] = mineig['DP']; row['mineig_Sigma_DPP'] = mineig['DPP']
            rows.append(row)
    return rows


def main():
    ratio, scan, blue, ytab, diag_rows = [], [], [], [], []
    for k in range(6):
        common, C = load(k)
        N = len(common)
        res = {S: per_state_y(C[S], ALL_WINDOWS) for S in STATES}
        for S in STATES:
            for d in res[S][2]:
                diag_rows.append(dict(state=S, kappa_idx=k, window=d['window'], f=d['f'], m=SIGN * d['m'],
                                      A=d['A'], chi2=d['chi2'], dof=d['dof'], conv=d['conv'],
                                      nfail_reps=d['nfail'], nfail_reps_primary_LM=d['nfail_primary'],
                                      max_newton_step_over_sigma=d['gradmax'], cov_mineig=d['cov_mineig'],
                                      cov_cond=d['cov_cond'], Gbar_sign_mixed=d['Gbar_sign_mixed']))
        for wi, (t0, t1) in enumerate(ALL_WINDOWS):
            wname = f'{t0}..{t1}'
            yc = {S: res[S][0][wi] for S in STATES}
            yr = {S: res[S][1][wi] for S in STATES}
            for S in STATES:
                ytab.append(dict(window=wname, kappa_idx=k, state=S,
                                 **{f'y_f{int(f)}': yc[S][i] for i, f in enumerate(bfa.FIELDS)},
                                 **{f'y_f{int(f)}_se': bfa.jk_se(yr[S][:, i]) for i, f in enumerate(bfa.FIELDS)}))
            rr = ratio_rows(k, N, wname, yc, yr)
            if (t0, t1) in SCAN_WINDOWS:
                scan += rr
            if (t0, t1) in MAIN_WINDOWS:
                ratio += rr
                blue += blue_rows(k, N, wname, yc, yr)
        print('kappa', k, 'done, Ncommon =', N, flush=True)
    order = {f'{a}..{b}': i for i, (a, b) in enumerate(MAIN_WINDOWS)}
    df = pd.DataFrame(ratio); df['_o'] = df.window.map(order)
    df.sort_values(['_o', 'kappa_idx']).drop(columns='_o').to_csv(f'{WORK}/decuplet_ratio.csv', index=False, float_format='%.10g')
    db = pd.DataFrame(blue); db['_o'] = db.window.map(order)
    db.sort_values(['_o', 'kappa_idx'], kind='stable').drop(columns='_o').to_csv(f'{WORK}/decuplet_blue.csv', index=False, float_format='%.10g')
    pd.DataFrame(scan).to_csv(f'{WORK}/decuplet_window_scan.csv', index=False, float_format='%.10g')
    pd.DataFrame(ytab).to_csv(f'{WORK}/decuplet_y.csv', index=False, float_format='%.10g')
    pd.DataFrame(diag_rows).to_csv(f'{WORK}/decuplet_fitdiag.csv', index=False, float_format='%.6g')


if __name__ == '__main__':
    main()
