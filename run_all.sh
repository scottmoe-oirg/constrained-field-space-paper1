#!/bin/sh
# Regenerate every table and figure of the paper.
# Requires: PAPER1_CORRELATORS pointing at a directory with
#   octet/OCT_N, octet/OCT_P   (cfly_2448_NNN.OCT_N / .OCT_P text files)
#   decuplet/                  (built below from the raw DEC_DP, DEC_DPP files: set RAW_DECUPLET)
# Outputs go to $PAPER1_WORK (default ./work), one subfolder per stage.
# Approximate runtimes on 2 cores: stage 1 ~1 min; 2 ~2 h; 3 ~2 days (bootstraps); 4 ~3 h; 5 ~6 h; 6-8 minutes; 9 ~40 min; 10 ~10 min.
# Stages 6 and 7 need only data/derived and can be run without the correlators: ./run_all.sh 6 7
# (stage 7 then skips the train/evaluate splits). See README.md for the table and figure map.
set -e
REPO=$(cd "$(dirname "$0")" && pwd); WORK=${PAPER1_WORK:-$REPO/work}; export PAPER1_WORK=$WORK
STAGES=${*:-"0 1 2 3 4 5 6 7 8 9 10"}
run() { mkdir -p "$WORK/$1"; (cd "$WORK/$1" && shift && "$@"); }
for s in $STAGES; do case $s in
 0) [ -n "$RAW_DECUPLET" ] && python3 "$REPO/pipeline/build_decuplet_inputs.py" --raw "$RAW_DECUPLET" --out "$PAPER1_CORRELATORS/decuplet" ;;
 1) # nucleon moments (Table I) and the fixed-covariance decuplet control (clean-room pipeline)
    run pipeline python3 "$REPO/pipeline/validate.py"
    run pipeline python3 "$REPO/pipeline/load_octet.py"
    run pipeline python3 "$REPO/pipeline/run_nucleon.py"
    run pipeline python3 "$REPO/pipeline/run_decuplet.py" ;;
 2) # nested jackknife, null-space tests, fit windows; nucleon BLUE with reoptimized weights
    run nested python3 "$REPO/analysis/nested/nested_all.py"
    run nested python3 "$REPO/analysis/nested/nested_windows.py"
    run nested python3 "$REPO/analysis/nested/window_scan_moments.py" P "$PAPER1_CORRELATORS/octet/OCT_P" 12 20 10,11,12,13,14,15,16
    run nested python3 "$REPO/analysis/nested/window_scan_moments.py" N "$PAPER1_CORRELATORS/octet/OCT_N" 17 26 14,15,16,17,18,19,20
    run nucleon_blue python3 "$REPO/analysis/nucleon_blue/nuc_oos.py" ;;
 3) # full-pipeline bootstraps (500 replicates each)
    # the published lightest-proton bootstrap was run as 200 + 300 replicates (offset 1); reproduce those seeds exactly
    run bootstrap python3 "$REPO/analysis/bootstrap/fullcal.py" P 0 200
    run bootstrap python3 "$REPO/analysis/bootstrap/fullcal.py" P 0 300 1
    for pk in "P 5" "N 0" "N 5"; do run bootstrap python3 "$REPO/analysis/bootstrap/fullcal.py" $pk 500; done ;;
 4) # decuplet Delta+/Delta++ control with temporal-covariance noise, Fig. 3
    run decuplet python3 "$REPO/analysis/decuplet/dec_nested_scan.py"
    run decuplet python3 "$REPO/analysis/decuplet/post.py"
    run decuplet python3 "$REPO/analysis/decuplet/joint_nested.py"
    run decuplet python3 "$REPO/analysis/decuplet/fig_nested.py" ;;
 5) # learning curve (Fig. 2); needs the correlators
    run learning python3 "$REPO/analysis/learning/learning_curve.py"
    run learning python3 "$REPO/analysis/learning/plot_learning_curve.py" poly11_small_learning_curve_exact_subsets.csv "$REPO/data/derived/frontier.csv" ;;
 6) # leverage frontier (Fig. 1), displacement table, Fig. 2 (all from data/derived)
    run frontier python3 "$REPO/analysis/frontier/frontier.py" --data "$REPO/data/derived" --cov fixed --out frontier.csv
    run frontier python3 "$REPO/analysis/frontier/frontier.py" --data "$REPO/data/derived" --cov nested --out frontier_nested.csv
    run frontier python3 "$REPO/analysis/frontier/plot_frontier.py" frontier.csv frontier_nested.csv
    run frontier python3 "$REPO/analysis/frontier/displacement_table.py" "$REPO/data/derived"
    # Fig. 2 from the stage-5 subsets if present, otherwise from the archived copy in data/derived
    LC="$WORK/learning/poly11_small_learning_curve_exact_subsets.csv"; [ -f "$LC" ] || LC="$REPO/data/derived/poly11_small_learning_curve_exact_subsets.csv"
    run learning python3 "$REPO/analysis/learning/plot_learning_curve.py" "$LC" "$WORK/frontier/frontier.csv" ;;
 7) # synthetic study (from data/derived) and train/evaluate splits (need correlators)
    run synthetic python3 "$REPO/analysis/synthetic/synthetic.py"
    run synthetic python3 "$REPO/analysis/synthetic/synthetic2.py" "$REPO/data/derived" "$REPO/data/derived"
    if [ -n "$PAPER1_CORRELATORS" ]; then run synthetic python3 "$REPO/analysis/synthetic/splits.py"; fi ;;
 8) # configuration-level decuplet identity
    run identity python3 "$REPO/analysis/identity/corr_identity.py" ;;
 9) # Delta0 (absolute) and Delta- controls, Delta++ BLUE-O4 shift; needs stage 4 for the Delta+ arrays
    run controls python3 "$REPO/analysis/controls/dec_extra_controls.py" D0
    run controls python3 "$REPO/analysis/controls/dec_extra_controls.py" DM
    run controls python3 "$REPO/analysis/controls/post_extra.py"
    run controls python3 "$REPO/analysis/controls/dpp_blue_o4.py" "$WORK/decuplet" ;;
 10) # Regularization: field-space shrinkage on the controls; uncorrelated / shrunk temporal fits for the nucleon
    run regularization python3 "$REPO/analysis/regularization/shrink_field_cov.py" "$WORK/decuplet" "$WORK/controls"
    run regularization python3 "$REPO/analysis/regularization/temporal_regularization.py"
    run regularization python3 "$REPO/analysis/regularization/nested_tempreg.py"
    # two-state temporal fits with nested-jackknife field-space covariance (all 12 nucleon ensembles)
    run twostate python3 "$REPO/analysis/twostate/two_state_nested.py" P0,N0,P1,N1,P2,N2,P3,N3,P4,N4,P5,N5 2 ;;
esac; done
echo "done: outputs in $WORK"
