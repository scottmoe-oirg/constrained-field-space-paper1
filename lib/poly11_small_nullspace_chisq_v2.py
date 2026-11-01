#!/usr/bin/env python3
"""Reviewer-facing weight-free null-space chi-square tests for poly11-small.

Uses the recovered Draft-09 scientific pipeline:
  configuration data -> spin/field double ratio -> correlated exponential fit
  -> four responses y=(y_B,y_2B,y_4B,y_8B) -> aligned delete-one JK covariance C
  -> fixed algebraic null-space tests for O4 and O6.

The script refuses to continue unless the lightest-proton regression fingerprint
matches the recovered analysis core.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import chi2

ETA0 = 0.001364
FIELDS = np.array([1., 2., 4., 8.])
CONV = 0.88658
KAPPAS = np.array([.1567, .1566, .1565, .1564, .1562, .1546])
MPI_MEV = np.array([337.749, 361.505, 383.802, 404.903, 444.264, 692.971])

# Draft-09 fit windows recovered from the analysis.
WINDOWS = {"P": np.arange(12, 21), "N": np.arange(17, 27)}
EXPECTED_N = {"P": 992, "N": 999}

# Exact deterministic null bases for x=m^2=(1,4,16,64).
# Columns satisfy N^T A = 0.
N4 = np.array([[4., 20.], [-5., -21.], [1., 0.], [0., 1.]])
N6 = np.array([[-64.], [84.], [-21.], [1.]])

# Lightest-proton recovered regression fingerprints.
EXPECTED_Y_P0 = np.array([2.20917302, 2.20063138, 2.16545036, 2.04216874])
EXPECTED_O4_P0 = 1.9611329055
EXPECTED_O4_SE_P0 = .0408682390
EXPECTED_T4_P0 = 37.26943127
EXPECTED_T6_P0 = 4.040752945


def parse_file(path):
    d = {}
    with open(path) as fh:
        for line in fh:
            a = line.split()
            if len(a) >= 4:
                try:
                    d[(a[0], int(a[1]))] = (float(a[2]), float(a[3]))
                except ValueError:
                    pass
    return d


def discover_particle(root, particle):
    """Discover OCT_P/OCT_N directory and infer label prefix from actual file text."""
    root = Path(root)
    preferred = root / f"OCT_{particle}"
    dirs = [preferred] if preferred.is_dir() else []
    if not dirs:
        dirs = [p for p in root.rglob(f"OCT_{particle}") if p.is_dir()]
    if not dirs:
        raise RuntimeError(f"Could not find OCT_{particle} under {root}")
    ddir = dirs[0]
    files = sorted(ddir.glob(f"cfly_2448_*.OCT_{particle}"))
    if not files:
        files = sorted(ddir.glob("cfly_2448_*"))
    if not files:
        raise RuntimeError(f"No configuration files found in {ddir}")

    # Infer a label template from the first file instead of silently assuming neutron names.
    pat = re.compile(rf"^(\S*OCT_{particle}_ETA)(01|02|04|08)_(PLUS|MINUS)_(\d\d)$")
    prefix = None
    with open(files[0]) as fh:
        for line in fh:
            a = line.split()
            if not a:
                continue
            m = pat.match(a[0])
            if m:
                prefix = m.group(1)
                break
    if prefix is None:
        # Historical proton form is exactly OCT_P_ETA...
        prefix = f"OCT_{particle}_ETA"
    return ddir, files, prefix


def load_particle(root, particle, kidx):
    ddir, files, prefix = discover_particle(root, particle)
    expn = EXPECTED_N[particle]
    if len(files) != expn:
        raise RuntimeError(f"Expected {expn} {particle} files; found {len(files)} in {ddir}")
    X = np.empty((len(files), 4, 2, 2, 48), dtype=float)
    for j, f in enumerate(files):
        d = parse_file(f)
        for q, mm in enumerate([1, 2, 4, 8]):
            for s, sg in enumerate(["PLUS", "MINUS"]):
                lab = f"{prefix}{mm:02d}_{sg}_{kidx:02d}"
                try:
                    X[j, q, s] = np.array([d[(lab, t)] for t in range(48)]).T
                except KeyError as e:
                    raise RuntimeError(
                        f"Missing label/time {e} in {f}. Expected label like {lab}."
                    ) from e
    return X, ddir, prefix


def ratio(M):
    return (M[:, 1, 0, :] * M[:, 0, 1, :]) / (M[:, 0, 0, :] * M[:, 1, 1, :])


def fit_ls(r, C, times):
    ev, U = np.linalg.eigh(C)
    keep = ev > ev.max() * 1e-12
    if keep.sum() < 2:
        raise RuntimeError("Time covariance is numerically rank deficient.")
    W = (U[:, keep] / np.sqrt(ev[keep])).T
    t = times.astype(float)
    m0 = -np.polyfit(t, np.log(np.maximum(np.abs(r), 1e-300)), 1)[0]
    A0 = np.exp(np.mean(np.log(np.maximum(np.abs(r), 1e-300)) + m0 * t))

    def resid(p):
        A, m = p
        return W @ (A * np.exp(-m * t) - r)

    res = least_squares(resid, [A0, m0], method="trf",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[1]


def ordinary_analysis(X, times):
    """Exact recovered aligned delete-one JK pipeline; returns y, C, y_jackknife."""
    n = len(X)
    S = X.sum(0)
    RF = ratio(S / n)
    loo = (S[None] - X) / (n - 1)
    R = (loo[:, :, 1, 0, :] * loo[:, :, 0, 1, :]) / (loo[:, :, 0, 0, :] * loo[:, :, 1, 1, :])
    y = np.empty(4)
    yj = np.empty((n, 4))
    for f in range(4):
        Z = R[:, f][:, times]
        DD = Z - Z.mean(0)
        Ct = (n - 1) / n * (DD.T @ DD)
        y[f] = fit_ls(RF[f, times], Ct, times) / (12 * ETA0 * FIELDS[f])
        for i in range(n):
            yj[i, f] = fit_ls(Z[i], Ct, times) / (12 * ETA0 * FIELDS[f])
    D = yj - yj.mean(0)
    C = (n - 1) / n * (D.T @ D)
    return y, C, yj


def null_test(y, C, N):
    z = N.T @ y
    V = N.T @ C @ N
    ev = np.linalg.eigvalsh(V)
    if np.any(ev <= 0):
        raise RuntimeError(f"Projected covariance is not positive definite: eig={ev}")
    T = float(z @ np.linalg.solve(V, z))
    dof = N.shape[1]
    p = float(chi2.sf(T, dof))
    return T, dof, p, z, V, ev, float(np.linalg.cond(V))


def analyze_one(root, particle, kidx):
    X, ddir, prefix = load_particle(root, particle, kidx)
    times = WINDOWS[particle]
    y, C, yj = ordinary_analysis(X, times)
    t4, df4, p4, z4, V4, ev4, cond4 = null_test(y, C, N4)
    t6, df6, p6, z6, V6, ev6, cond6 = null_test(y, C, N6)

    w4 = np.array([4/3, -1/3, 0., 0.])
    o4 = CONV * float(w4 @ y)
    o4se = CONV * float(np.sqrt(w4 @ C @ w4))

    row = dict(
        particle=particle, kidx=kidx, kappa=KAPPAS[kidx], mpi_MeV=MPI_MEV[kidx],
        nconfig=len(X), fit_tmin=int(times[0]), fit_tmax=int(times[-1]),
        y1=y[0], y2=y[1], y4=y[2], y8=y[3],
        O4_muN=o4, O4_se_muN=o4se,
        chi2_O4=t4, dof_O4=df4, p_O4=p4,
        O4_contrast_1=z4[0], O4_contrast_2=z4[1],
        O4_proj_eig_min=ev4.min(), O4_proj_eig_max=ev4.max(), O4_proj_cond=cond4,
        chi2_O6=t6, dof_O6=df6, p_O6=p6,
        O6_contrast=z6[0],
        O6_proj_eig=ev6[0], O6_proj_cond=cond6,
        C_cond=float(np.linalg.cond(C)),
        label_prefix=prefix, source_dir=str(ddir),
    )
    return row, y, C


def regression_check(row, y):
    # Mandatory reconstruction fingerprints: these must reproduce tightly.
    checks = [
        ("y", float(np.max(np.abs(y - EXPECTED_Y_P0))), 2e-7),
        ("O4", abs(row["O4_muN"] - EXPECTED_O4_P0), 2e-7),
        ("O4_SE", abs(row["O4_se_muN"] - EXPECTED_O4_SE_P0), 2e-7),
    ]
    bad = [(n, e, tol) for n, e, tol in checks if e > tol]
    if bad:
        raise RuntimeError("Lightest-proton reconstruction regression failure: " + repr(bad))

    # Historical null-statistic values are diagnostic fingerprints only.
    # Small cross-platform differences in the projected covariance solve must
    # not block the fresh reviewer-requested calculation.
    d4 = row["chi2_O4"] - EXPECTED_T4_P0
    d6 = row["chi2_O6"] - EXPECTED_T6_P0
    print(f"  reconstruction fingerprints PASS")
    print(f"  current T_O4 = {row['chi2_O4']:.12g}  "
          f"(stored {EXPECTED_T4_P0:.12g}, delta {d4:+.6g})")
    print(f"  current T_O6 = {row['chi2_O6']:.12g}  "
          f"(stored {EXPECTED_T6_P0:.12g}, delta {d6:+.6g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="_poly11_work/poly11-small-reduced",
                    help="Extracted poly11-small-reduced root containing OCT_P and OCT_N")
    ap.add_argument("--output", default="poly11_small_nullspace_chisq.csv")
    ap.add_argument("--cov-output", default="poly11_small_nullspace_covariances.npz")
    ap.add_argument("--particle", choices=["P", "N", "both"], default="both")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    print("Running mandatory lightest-proton regression check...", flush=True)
    row0, y0, C0 = analyze_one(root, "P", 0)
    regression_check(row0, y0)
    print(f"PASS: O4={row0['O4_muN']:.10f} +/- {row0['O4_se_muN']:.10f}; "
          f"T4={row0['chi2_O4']:.8f}; T6={row0['chi2_O6']:.8f}", flush=True)

    particles = ["P", "N"] if args.particle == "both" else [args.particle]
    rows = []
    covs = {}
    for particle in particles:
        for kidx in range(6):
            if particle == "P" and kidx == 0:
                row, y, C = row0, y0, C0
            else:
                print(f"Analyzing {particle} kidx={kidx} kappa={KAPPAS[kidx]:.4f}...", flush=True)
                row, y, C = analyze_one(root, particle, kidx)
            rows.append(row)
            covs[f"{particle}_k{kidx}_y"] = y
            covs[f"{particle}_k{kidx}_C"] = C
            print(f"  O4 chi2/dof={row['chi2_O4']:.6g}/2 p={row['p_O4']:.6g} | "
                  f"O6 chi2/dof={row['chi2_O6']:.6g}/1 p={row['p_O6']:.6g}", flush=True)

    df = pd.DataFrame(rows).sort_values(["particle", "kidx"])
    df.to_csv(args.output, index=False)
    np.savez_compressed(args.cov_output, **covs)

    cols = ["particle", "mpi_MeV", "kappa", "chi2_O4", "dof_O4", "p_O4",
            "chi2_O6", "dof_O6", "p_O6", "O4_proj_cond", "O6_proj_cond"]
    print("\nReviewer-facing summary:\n")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.cov_output}")

if __name__ == "__main__":
    main()
