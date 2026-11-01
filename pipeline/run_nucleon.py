"""Deliverable A: nucleon (proton, neutron) background-field moments -> work/nucleon.csv"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'lib')); from paths import OCTET, DECUPLET, DERIVED, WORK
import numpy as np, pandas as pd, json
import bfa

WORK = str(WORK)
WINDOWS = {'P': (12, 20), 'N': (17, 26)}
W = bfa.stencils(); NB = bfa.null_bases()
# Global sign factor applied to y: fixed by the proton anchor (see DECISIONS.md).
# +B (physical) = file tag MINUS = cache index 0;  -B = PLUS = index 1.
SIGN = None


def analyse_state(S):
    d = np.load(f'{WORK}/cache_OCT_{S}.npz')
    C = d['C']                                   # cfg, field, tag(0=MINUS=+B,1=PLUS=-B), k, t, comp
    N = C.shape[0]
    mean, reps = bfa.jk_means(C)
    t0, t1 = WINDOWS[S]
    tt = np.arange(48)
    out = []
    for k in range(6):
        yc = np.empty(4); yr = np.empty((N, 4)); diag = []
        for fi, f in enumerate(bfa.FIELDS):
            g = lambda a, tag, comp: a[..., fi, tag, k, :, comp]
            Rc = bfa.double_ratio(g(mean, 0, 0), g(mean, 0, 1), g(mean, 1, 0), g(mean, 1, 1))
            Rr = bfa.double_ratio(g(reps, 0, 0), g(reps, 0, 1), g(reps, 1, 0), g(reps, 1, 1))
            # sign health of the averaged correlators in the window
            neg = int((np.stack([g(mean, a, b) for a in (0, 1) for b in (0, 1)])[:, t0:t1 + 1] <= 0).sum())
            fw = bfa.fit_window(Rc, Rr, tt, t0, t1)
            fw['neg_Gbar_points'] = neg
            fw['f'] = int(f)
            diag.append(fw)
            yc[fi] = bfa.m_to_y(fw['m'], f); yr[:, fi] = bfa.m_to_y(fw['m_reps'], f)
        out.append((k, N, yc, yr, diag))
    return out


def main():
    raw = {S: analyse_state(S) for S in ['P', 'N']}
    # sign anchor: proton must be positive; take the sign of the raw proton O2 at every kappa
    s_raw = np.sign([o[2][0] for o in raw['P']])
    assert np.all(s_raw == s_raw[0]), 'proton raw sign not uniform across kappa'
    sign = float(s_raw[0])
    rows = []; fitdiag = []
    for S in ['P', 'N']:
        for k, N, yc, yr, diag in raw[S]:
            yc = sign * yc; yr = sign * yr
            row = dict(state=S, kappa_idx=k, Ncfg=N, window=f'{WINDOWS[S][0]}..{WINDOWS[S][1]}',
                       sign_factor=sign)
            for fi, f in enumerate(bfa.FIELDS):
                row[f'y_f{int(f)}'] = yc[fi]; row[f'y_f{int(f)}_se'] = bfa.jk_se(yr[:, fi])
            for name, w in W.items():
                row[f'{name}_muN'] = bfa.MU_N_FACTOR * (w @ yc)
                row[f'{name}_se'] = bfa.MU_N_FACTOR * bfa.jk_se(yr @ w)
            for name in ['O4', 'O6']:
                T2, p = bfa.hotelling(NB[name], yc, yr)
                row[f'T2_{name}'] = T2; row[f'p_{name}'] = p
            row['nfit_fail_central'] = sum(not d['conv'] for d in diag)
            row['nfit_fail_reps'] = sum(d['nfail'] for d in diag)
            row['nfit_fail_reps_primary_LM'] = sum(d['nfail_primary'] for d in diag)
            rows.append(row)
            for d in diag:
                fitdiag.append(dict(state=S, kappa_idx=k, f=d['f'], m=d['m'] * sign, A=d['A'],
                                    chi2=d['chi2'], dof=d['dof'], conv=d['conv'], nfail_reps=d['nfail'], nfail_reps_primary_LM=d['nfail_primary'],
                                    max_newton_step_over_sigma=d['gradmax'],
                                    cov_mineig=d['cov_mineig'], cov_cond=d['cov_cond'],
                                    neg_Gbar_points=d['neg_Gbar_points']))
    pd.DataFrame(rows).to_csv(f'{WORK}/nucleon.csv', index=False, float_format='%.10g')
    pd.DataFrame(fitdiag).to_csv(f'{WORK}/nucleon_fitdiag.csv', index=False, float_format='%.6g')
    print('sign factor applied to raw y:', sign)
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(df[['state', 'kappa_idx', 'O2_muN', 'O2_se', 'O4_muN', 'O4_se', 'O6_muN', 'O6_se', 'O8_muN', 'O8_se',
              'T2_O4', 'p_O4', 'T2_O6', 'p_O6', 'nfit_fail_central', 'nfit_fail_reps', 'nfit_fail_reps_primary_LM']].to_string())
    fd = pd.DataFrame(fitdiag)
    print(fd.to_string())


if __name__ == '__main__':
    main()
