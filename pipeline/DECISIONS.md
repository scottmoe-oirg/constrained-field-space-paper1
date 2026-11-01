> Historical document from the independent clean-room reimplementation. Paths below refer to the original working environment; in this package they are set in `lib/paths.py`.

# DECISIONS.md: choices made where SPEC.md is ambiguous or incomplete

## Isolation
- Data were read only through `/home/claude/cleanroom/data/`. No other location was opened, listed or searched.
- **Observation, not something I chose:** `ls -la data/` shows that `data/OCT_P` and `data/OCT_N` are **symlinks** to
  `/home/claude/audit/data/injection_recovery_mocks/_poly11_work/poly11-small-reduced/OCT_{P,N}`. I read the files only
  through the cleanroom path and did not look at the target directory. The target path suggests that the nucleon
  inputs are **injection-recovery mocks** (perhaps with an injected polynomial in field), not the raw lattice data the
  spec describes. See REPORT.md.

## Data handling
1. **Labels and tags.** Physical +B = file tag `MINUS` and physical -B = `PLUS`, as the spec says. In the double ratio
   I use G11(+B) = G11[MINUS], G22(-B) = G22[PLUS], and so on.
2. **Nucleon ensembles.** P (992 cfgs) and N (999 cfgs) are analysed separately, each on all of its own configurations.
   The spec asks for configuration matching only for the Delta. The P set is missing configurations
   032, 154, 191, 445, 491, 675, 848 and 956; N is missing 675.
3. **Delta ensembles.** DP and DPP are restricted to the intersection of their `cfg` arrays, which has 892
   configurations (every DP configuration is also in DPP). I matched them by configuration number, checked that the
   arrays are identical after matching, and reordered both into the same order, so that delete-one replica j removes
   the same gauge configuration from both states and from all 8 field labels.
4. **Overall sign of the Delta correlators.** All Delta correlators are negative (an overall phase convention). The
   double ratio is invariant under G -> -G, so I applied no correction.
5. Zero-field labels are parsed and then ignored (they are not needed).

## Sign convention (spec §3.5)
- **Method.** I compute the raw m from the double ratio exactly as the spec writes it, with +B = MINUS, and set
  y = m/(12 eta0 f). The sign anchor is the proton: the global sign factor s is chosen so that the proton y > 0. The
  same s is then applied, without change, to N, DP and DPP. Those states serve as independent checks.
- **Result.** The raw proton y is positive at every kappa, so s = +1 and no flip is applied. With s = +1 the neutron
  comes out negative and the Delta++ positive (Delta+ is positive as well), at every kappa and every window. So the
  anchor is consistent across all four states.
- **Spec inconsistency (minor).** Take E_up = M - mu B and E_down = M + mu B, and identify G11 with spin up. Then the
  literal double ratio gives R ~ exp(+4 mu B t), that is, m = -4 mu B. The spec instead says m = +4 a mu B. The
  difference is absorbed by the sign of eta (q_d < 0), or equivalently by which tag counts as +B, or by which
  component is spin up. Because of this the sign cannot be derived from first principles out of the spec, and the
  empirical anchor is needed. The result does not depend on this point.

## Fit
6. **Fitter.** My own vectorised Levenberg–Marquardt on the whitened residuals L^-1 (R - A e^{-mt}), where
   cov = L L^T is the jackknife covariance of R over the window. It fits 2 parameters (A, m) directly. The covariance
   is computed **once** per (state, kappa, field, window) from the delete-one replicas of R, with the (N-1)/N factor,
   and is held fixed in the central fit and in every replica fit.
7. **Starting values.** The central fit starts from an unweighted log-linear fit. Every replica fit starts from the
   central-fit result.
8. **Convergence criterion.** It is checked per fit, explicitly: the undamped Gauss–Newton step from the final point
   must be smaller than 1e-6 times the parameter's standard error (inverse Hessian), for both A and m.
   - A fit that misses this criterion within 200 iterations is **counted** (`nfail_reps_primary_LM`). It is then refit
     with `scipy.optimize.least_squares` (method `lm`), which is an independent code path, and accepted only if its
     residual Newton step is < 1e-4 sigma.
   - Fits that still fail would be counted in `nfail_reps` / `conv`. None did.
   - All counts are in `*_fitdiag.csv`.
9. **Validation before use** (`validate.py`, all checks pass):
   - noiseless recovery;
   - agreement with scipy `least_squares` to 5e-8 sigma;
   - pull mean and width and chi^2/dof on 500 noisy correlated vectors;
   - a bad starting point converges to the same minimum;
   - a full end-to-end synthetic ensemble (per-config correlators with known y0, field-even shifts, cubic field
     dependence, and gauge noise shared across fields), followed by a 40-ensemble pull study of O2..O8. The pull
     means are consistent with 0, and the pull widths are 1.07.
10. **Bias correction.** The central value is the fit to the full-sample R. No jackknife bias correction is applied,
    because the spec gives SE formulas only. The `r_jk_mean` column shows that the replica-mean r agrees with the
    central r to < 3e-6 (max 2.4e-6), far below every SE.

## Field-space estimators (spec §4)
11. **Stencils.** Each stencil is Lagrange extrapolation to x = 0 over its support:
    w_i = prod_{j != i} (0 - x_j)/(x_i - x_j).
    - O4 = (4/3, -1/3, 0, 0)
    - O6 = (64/45, -4/9, 1/45, 0) = (1.42222, -0.44444, 0.02222, 0)
    - O8 = (1.444797, -0.474074, 0.0296296, -3.52734e-4)
    - All constraints were verified numerically, to 1e-12.
12. **Null bases.** I used divided-difference vectors.
    - O6: n3_i = 1/prod_{j != i}(x_i - x_j) over all four x.
    - O4: the pair {second divided difference on f = 1, 2, 4 padded with 0, n3}.
    - T^2 does not depend on which basis spans the null space (u -> M^T u, V -> M^T V M). I verified rank and the
      annihilation conditions.
13. **Hotelling statistic.** u uses the central y. V is the jackknife covariance of N^T y_j, including the (N-1)/N
    factor. J = N replicas. F = T^2 (J - q)/(q (J - 1)), with p from F(q, J - q), exactly as specified.
    - Strictly speaking, V estimates the covariance of the *mean*, so T^2 behaves as a chi^2_q-like statistic and the
      F correction is negligible at J ~ 1000. I followed the spec.
14. **Moments.** mu/mu_N = 0.88658 * (w . y), applied to all states, including the Delta.
    - Check: 0.88658 = 2 a M_N^phys = 2 * 0.93827/2.116, which is consistent with m = 4 mu B and eB a^2 = 3|eta|.

## Delta ratio (spec §5B)
15. r is computed from central estimates, r = (w.y_DP)/(w.y_DPP). SE(r) is the jackknife SE of the replica ratios
    r_j = (w.y_DP,j)/(w.y_DPP,j). sigma = (r - 1/2)/SE.
16. **Moment normalisation for spin 3/2 (ambiguity).** The spec's formula E = M ∓ mu B is exact only if G11 and G22
    are the s_z = ±3/2 states. If they are the s_z = ±1/2 projections, the energy splitting is mu B/3, and the
    reported Delta "moments" are mu/3. The ratio r is unaffected either way. The data favour s_z = ±1/2 (see
    REPORT.md). I report the moments exactly as the spec defines them, without any factor of 3.

## BLUE (spec §5C)
17. **Covariance.** Sigma is the 4x4 jackknife covariance of the replica y-vectors, per state, with the (N-1)/N
    factor. The weights do not depend on the overall scale.
18. **Primary choice: weights fixed from the full sample.** The full-sample weights are applied to every replica, so
    the SE is the jackknife SE of w.y_j, which equals sqrt(w^T Sigma w). This is the standard choice and isolates the
    statistical error of a fixed linear estimator. However, it **ignores the uncertainty that comes from estimating
    the weights**, and it is optimistic when Sigma is ill-conditioned.
19. **Secondary choice: re-optimised weights (also reported, `weights = reopt`).** For replica j the weights come from
    Sigma_(j), the jackknife covariance of the other N - 1 replicas, computed as an exact rank-one downdate of the
    scatter matrix. A true double jackknife (refitting every config pair, about N^2 fits) is not affordable, so this
    approximation is what is used. The central value is the same in both modes.
20. **Columns.** I report BLUE-1, BLUE-O4 and BLUE-O6. The error ratio is SE(BLUE)/SE(fixed O6), computed per state
    with the same replica treatment. Weights are listed for DP, and for DPP as well. The condition number is the
    2-norm condition number of Sigma.

## Windows
21. The windows in B (10..18, 12..20) also appear in the scan (D). They are computed once and written to both files.
22. **Possible spec issue.** The neutron window 17..26 sits far out in t, where the nucleon signal-to-noise is poor.
    The averaged correlators are still positive at every point of the window, at every field and kappa, so the window
    was used as specified.
