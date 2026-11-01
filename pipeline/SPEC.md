> Historical document from the independent clean-room reimplementation. Paths below refer to the original working environment; in this package they are set in `lib/paths.py`.

# Clean-room specification: background-field magnetic-moment analysis (paper 1)

You are implementing an analysis from a written description only. There is no reference code and
no reference results. Write your own code in `/home/claude/cleanroom/work/`. Read data only from
`/home/claude/cleanroom/data/`. Do not open, list or search any other directory under `/home/claude`,
`/mnt` or the home folder. Other directories hold an existing implementation and its outputs, and
looking at them defeats the purpose. Use Python 3 with numpy/scipy/pandas (install anything else you need).

Where this spec is ambiguous, pick the most defensible reading, and record each choice in `work/DECISIONS.md`.
If you think the spec is wrong or internally inconsistent, say so in DECISIONS.md. Do not silently work around it.

## 1. Physical setup

- Quenched Wilson lattice, 24^3 x 48, a = 0.09325 fm, a^-1 = 2.116 GeV.
- Six quark masses, kappa index k = 0..5 (kappa = 0.1567, 0.1566, 0.1565, 0.1564, 0.1562, 0.1546;
  m_pi ~ 338, 362, 384, 405, 444, 693 MeV).
- Uniform background magnetic field along z, applied to the valence quarks only. There are four field
  magnitudes f = 1, 2, 4, 8 (in units of a base field), each at both signs. The field parameter is
  eta = eta0 * f with eta0 = 0.001364, where eta = q_d B a^2 and q_d = -e/3 is the d-quark charge.
- **Sign convention of the archive:** the file tag `MINUS` is the physical +B and `PLUS` is the
  physical -B.
- Baryon two-point correlators G(t), t = 0..47, source at t = 0. For each configuration, field and kappa
  there are two real spin-projected components, called G11 and G22 below (the two spin states along
  the field axis).

## 2. Data

### 2.1 Nucleon (octet), plain text
`data/OCT_P/cfly_2448_NNN.OCT_P` (992 configurations) and `data/OCT_N/cfly_2448_NNN.OCT_N` (999).
Each line: `LABEL  t  Re(G11)  Re(G22)`, with LABEL like `OCT_P_ETA04_MINUS_02`
(field f = 4, tag MINUS, kappa index 02). The zero-field labels (`..._ETA_ZERO_kk`) are not needed here.

### 2.2 Delta (decuplet), numpy
`data/decuplet/s_{ST}_k{k}.npz` for ST in {DP (Delta+), DPP (Delta++)} and k = 0..5. Keys:
- `C`: float64 array `[cfg, label, t, comp]` with t = 0..31 and comp 0 = G11, comp 1 = G22 (positive parity);
- `labels`: the 8 label strings `01_PLUS, 01_MINUS, 02_PLUS, 02_MINUS, 04_PLUS, 04_MINUS, 08_PLUS, 08_MINUS`;
- `cfg`: configuration numbers.

DPP has 999 configurations and DP has 892. **Use only the configurations common to both (match by
configuration number)** so that the two states are analysed on identical gauge ensembles.
These arrays are a lossless re-packaging of the raw text files (spot-checked against the raw text).

## 3. Per-field response (time direction)

For each state, kappa and field f:

1. **Double ratio** built from configuration averages Gbar:
   R(t; B) = Gbar11(t; +B) Gbar22(t; -B) / [ Gbar11(t; -B) Gbar22(t; +B) ].
   With spin energies E_{up/down} = M -/+ mu B + (field-even terms), R ~ A exp(-m(B) t), with
   m(B) = 4 a mu B + O(B^3). All field-even shifts cancel.
2. **Fit windows** (inclusive): proton t = 12..20, neutron t = 17..26. Delta: two windows,
   t = 10..18 and t = 12..20.
3. **Fit:** model A exp(-m t), two parameters (A, m). Correlated chi^2 over the window, with the
   covariance of R over the window as the weight.
4. **Covariance and resampling:** use delete-one jackknife replicas. Replica j is built from the averages
   over all configurations except j, and the deletion is aligned across all fields (and, for the Delta,
   across both states). The covariance of R over the window is the jackknife covariance of the delete-one
   replicas of R, (N-1)/N * sum_j (R_j - mean)(R_j - mean)^T. It is computed **once** per
   (state, kappa, field, window) and **held fixed** in the central fit and every replica fit.
5. **Reduced response:** y(f) = m(B) / (12 eta0 f), where m is in lattice units. Signs: with the
   conventions above, a positive magnetic moment must give positive y. Use this as the sign anchor
   (proton and Delta++ have positive moments, the neutron a negative one); state in DECISIONS.md how
   you fixed the sign.
6. Conversion to nuclear magnetons: mu/mu_N = 0.88658 * y.

## 4. Field space

Write x = f^2 = (1, 4, 16, 64). Since m(B) is odd in B, y(f) = y0 + c1 x + c2 x^2 + c3 x^3 + ...
The target is y0. Estimators are linear combinations w . y with weights fixed by the field hierarchy:
- O2: w = (1, 0, 0, 0)
- O4: removes the x term (w.1 = 1, w.x = 0, support on f = 1, 2)
- O6: removes x and x^2 (w.1 = 1, w.x = w.x^2 = 0, support on f = 1, 2, 4)
- O8: removes x, x^2, x^3 (all four fields)

Derive the weights yourself. Errors are delete-one jackknife errors computed from the replica y-vectors:
SE = sqrt((N-1)/N * sum_j (w.y_j - mean)^2).

**Null tests (Hotelling).** For O4, the null space is the 2-dimensional space of vectors n with
n . 1 = n . x = 0 on the four fields. For O6, it is the 1-dimensional space with n . 1 = n . x = n . x^2 = 0.
Form u = N^T y, where the columns of N span the null space. Use the jackknife covariance V of N^T y_j and
compute T^2 = u^T V^-1 u. Report T^2 and the p-value from the Hotelling F distribution:
F = T^2 (J - q) / (q (J - 1)) with q = null-space dimension, J = number of jackknife replicas,
and F ~ F(q, J - q).

## 5. Deliverables (write CSVs to `work/`)

**A. Nucleon (`nucleon.csv`):** for P and N and each kappa, report:
- y(f) for f = 1, 2, 4, 8;
- O2/O4/O6/O8 in mu_N with jackknife SE;
- T^2 and p for the O4 and O6 null tests.

**B. Delta charge-ratio control (`decuplet_ratio.csv`):** with exactly degenerate u/d quarks and the field
on the valence quarks only, the linear response is proportional to the total quark charge, so
mu(Delta+)/mu(Delta++) = 1/2 exactly in this setup. For each window, kappa and stencil (O2, O4, O6, O8),
report:
- each state's moment and SE;
- r = mu_O(Delta+)/mu_O(Delta++), where the subscript O means the stencil estimate;
- the jackknife SE of r, computed replica by replica from the ratio of replica estimates;
- (r - 1/2)/SE;
- the O6 null-test T^2 for each state.

**C. Covariance-optimised estimators on the control (`decuplet_blue.csv`):** for each state separately,
let Sigma be the 4x4 jackknife covariance of the replica y-vectors. BLUE weights minimise w^T Sigma w
subject to w . x^p = delta_{p0} for p = 0..P-1:
- BLUE-1: P = 1, normalisation only;
- BLUE-O4: P = 2;
- BLUE-O6: P = 3.

The closed form is w = Sigma^-1 A (A^T Sigma^-1 A)^-1 e1, where A has columns x^p.

For each window and kappa, report:
- r - 1/2, its SE and (r - 1/2)/SE;
- the ratio of each state's BLUE marginal error to its fixed-O6 marginal error;
- the BLUE-1 and BLUE-O6 weight vectors for Delta+;
- cond(Sigma).

State how you treat the weights inside the jackknife (fixed from the full sample, or re-optimised per
replica) and why. If time permits, do both.

**D. Window scan (`decuplet_window_scan.csv`):** repeat B for the windows
- t = tmin..18 for tmin = 8..14;
- t = tmin..20 for tmin = 10..16;
- t = tmin..22 for tmin = 12..16.

**E. `work/REPORT.md`:** summarise your results. Point out anything that looked suspicious in the data or
the method, such as:
- non-converged fits;
- non-positive-definite covariances;
- the sign of the response in each state;
- whether any assumption in this spec is contradicted by the data.

Performance note: there are about 1000 replicas x 4 fields x 6 kappas x several windows of 2-parameter
fits. Use a good initial guess for each replica fit (the central-fit result) and vectorise where possible.
Multiprocessing is allowed. The machine has a few cores, and each full pass should take minutes, not hours.
