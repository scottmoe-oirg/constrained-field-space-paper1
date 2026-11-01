"""Locations used by every script. Override with environment variables:
  PAPER1_CORRELATORS  directory holding octet/OCT_N, octet/OCT_P (cfly_2448_*.OCT_* text files)
                      and decuplet/ (s_*.npz from pipeline/build_decuplet_inputs.py)
  PAPER1_WORK         scratch/output directory for heavy intermediate arrays (default: <repo>/work)
"""
import os, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
CORR = pathlib.Path(os.environ.get('PAPER1_CORRELATORS', REPO / 'correlators'))
OCTET = CORR / 'octet'
DECUPLET = CORR / 'decuplet'
DERIVED = REPO / 'data' / 'derived'
WORK = pathlib.Path(os.environ.get('PAPER1_WORK', REPO / 'work')); WORK.mkdir(parents=True, exist_ok=True)
LIB = REPO / 'lib'
