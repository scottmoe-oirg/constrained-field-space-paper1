# Reproducibility package: Constrained field-space estimators for background-field lattice QCD observables

S. Moerschbacher and F. X. Lee, Phys. Rev. D (submitted). arXiv: [ARXIV ID]

This package regenerates every table and figure of the paper from the archived background-field
correlators (quenched Wilson, 24^3x48, poly11-small ensemble). The correlators themselves are available
from the authors on request; everything downstream of the per-replica field responses is included here.

## Layout

| Folder | Contents |
| --- | --- |
| `run_all.sh` | Runs everything, stage by stage (see the header for stages and runtimes) |
| `lib/` | Shared code: `paths.py` (all file locations), the field-space fitting core, constants |
| `pipeline/` | `build_decuplet_inputs.py` (raw decuplet text -> arrays) and the clean-room pipeline: correlators -> double ratios -> correlated fits -> aligned delete-one jackknife -> field responses y(B) and fixed stencils. `SPEC.md` is the specification it was written from; `DECISIONS.md` records every choice. |
| `analysis/nested/` | Nested jackknife (temporal-covariance noise), null-space tests, fit-window scan |
| `analysis/bootstrap/` | Full-pipeline configuration bootstrap (500 replicates) |
| `analysis/decuplet/` | Delta+/Delta++ ratio control: fixed stencils, BLUE (fixed and reoptimized weights), joint optimization, 19-window scan |
| `analysis/nucleon_blue/` | Nucleon BLUE with weights re-derived per replica |
| `analysis/frontier/` | Hard-leverage frontier |
| `analysis/controls/` | Absolute decuplet controls (Delta0 = 0, Delta-/Delta++ = -1/2) and the Delta++ BLUE-O4 shift |
| `analysis/regularization/` | Shrinkage of the field-space covariance; uncorrelated and shrunk temporal fits |
| `analysis/learning/` | Learning curve (30 subsets per N; seed 20261008) |
| `analysis/synthetic/` | Known-covariance synthetic study and train/evaluate splits |
| `analysis/identity/` | Configuration-level decuplet identity check |
| `analysis/poly0/` | Reproduction of the original 90-configuration poly0 notebook fits |
| `data/derived/` | Derived outputs (CSV/JSON) behind the tables |
| `paper/` | LaTeX source and bibliography |

## Table and figure map (draft 20)

| Item | Script | Output (in `data/derived/`) |
| --- | --- | --- |
| Table I (O2-O8 moments) | `pipeline/run_nucleon.py` | `nucleon.csv` |
| Table II (fit windows) | `analysis/nested/window_scan_moments.py`, `analysis/nested/nested_windows.py` | `window_scan_{P,N}.json`, `nested_windows.csv` |
| Table III (null-space tests) | `analysis/nested/nested_all.py` | `nested_vs_fixed.csv`, `nested_*.npz` |
| Table IV (O6 test calibration) | `analysis/bootstrap/fullcal.py` | `fullcal_*.npz/json`, `bootstrap_validation.json` |
| Sec. VI A (uncorrelated and shrunk temporal fits) | `analysis/regularization/temporal_regularization.py`, `nested_tempreg.py` | `tempreg.csv`, `nested_tempreg.csv` |
| Table V, Sec. VI A (two-state temporal fits) | `analysis/twostate/two_state_nested.py` | `two_state_nested.csv` |
| Table VI (BLUE − fixed displacements) | `analysis/frontier/displacement_table.py` | `displacement_table.csv` |
| Table VII, Fig. 1 (leverage frontier) | `analysis/frontier/frontier.py`, `plot_frontier.py` | `frontier.csv`, `frontier_nested.csv` |
| Sec. VII C (train/evaluate splits) | `analysis/synthetic/splits.py` | `splits.csv` |
| Fig. 2 (learning curve) | `analysis/learning/learning_curve.py`, `plot_learning_curve.py` | `poly11_small_learning_curve_exact_*.csv` |
| Table VIII and Sec. VII E (synthetic: displacement, pre-test power, population optimum, field design) | `analysis/synthetic/synthetic.py`, `synthetic2.py` | `synthetic_results.csv`, `pretest_power.csv`, `population_optimum.csv`, `field_design.csv` |
| Table IX, Fig. 3 (Delta+/Delta++) | `analysis/decuplet/dec_nested_scan.py`, `post.py`, `fig_nested.py` | `dec_scan_nested_summary.csv` |
| Table X (Delta0) and the Delta- control | `analysis/controls/dec_extra_controls.py`, `post_extra.py` | `dec_extra_summary.csv` |
| Table XI (decuplet BLUE), joint optimization | `analysis/decuplet/post.py`, `joint_nested.py` | `dec_blue_nested.csv`, `joint_nested.csv` |
| Delta++ BLUE-O4 shift (Sec. VIII D) | `analysis/controls/dpp_blue_o4.py` | `dpp_blue_o4.csv` |
| Shrinkage (Sec. VIII D) | `analysis/regularization/shrink_field_cov.py` | `shrinkage.csv` |
| Table XII (identity check) | `analysis/identity/corr_identity.py` | `corr_identity.csv` |
| Appendix A (poly0 reproduction) | `analysis/poly0/poly0_repro.py` | |

## Running

```sh
export PAPER1_CORRELATORS=/path/to/correlators     # octet/OCT_N, octet/OCT_P; decuplet/ is built by stage 0
export RAW_DECUPLET=/path/to/raw                   # folder containing DEC_DP/ and DEC_DPP/
./run_all.sh            # all stages
./run_all.sh 6 7        # stages that need only data/derived (no correlators)
```

Each stage writes to `$PAPER1_WORK/<stage>/` (default `./work`). `data/derived/` holds the outputs used in the paper,
so tables can be checked against it.

## Verification status (2026-11-01, draft 20)

Every stage has now been rerun from this package against the raw correlators, and the outputs compared with `data/derived`:

- Stage 0: rebuilt decuplet arrays agree with the arrays used in the paper (checksums to 1e-15).
- Stage 1: `nucleon.csv` and all decuplet outputs are byte-identical.
- Stage 2 (nested jackknife): `nested_vs_fixed`, `nested_windows`, window scans and nested `.npz` files are byte-identical; `nuc_oos.csv` added from this rerun.
- Stage 3 (full-pipeline bootstrap): `fullcal_{P0,P0_1,P5,N0,N5}.npz/.json` reproduce exactly (max |difference| = 0 in every array; JSON identical apart from run time). Roughly 30 min per 500 replicates on one core.
- Stage 4 (post-processing), stage 6, stage 8: identical.
- Stage 5 (learning curve): subsets and summary are byte-identical; the figure is `fig_learning_curve_leverage_v4`.
- Stage 7 (synthetic study and splits): `field_design`, `population_optimum`, `pretest_power`, splits, `synthetic_leverage`, `synthetic_results` are byte-identical.
- Stages 9 and 10: run from these scripts for drafts 17–20, including `analysis/regularization/nested_tempreg.py` (draft 19) and `analysis/twostate/two_state_nested.py` (draft 20).
- `displacement_table.py` reproduces Table VI exactly from `frontier*.csv` and `nested_*.npz`.

## Requirements

Python 3.10+, numpy, scipy, matplotlib.

## Licence

Code: MIT. Derived data: CC BY 4.0.
