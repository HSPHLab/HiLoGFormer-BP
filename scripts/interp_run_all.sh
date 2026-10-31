#!/usr/bin/env bash
# Final, consistent regeneration of every interpretability table and figure from the frozen caches (same scripts, same seeds),
# then re-finalisation into paper/final with a logged reason. Usage: bash scripts/interp_run_all.sh <gpu>
set -euo pipefail
GPU=${1:-2}
REASON=${2:-"final consistent regeneration of all interpretability outputs"}
cd "$(dirname "$0")/.."
for f in runs/interpretability/pairs_notime.csv.gz runs/interpretability/pulsedb_v1notime.npz data_processed/features_pwa/aurora_v1.csv; do
  [ -f "$f" ] || { echo "missing input: $f"; exit 1; }
done
mkdir -p runs/logs
python scripts/interp_A_trust.py        > runs/logs/final_A.out 2>&1          && echo "A done"
python scripts/interp_B_drift.py        > runs/logs/final_B.out 2>&1          && echo "B done"
python scripts/interp_C_explain.py      > runs/logs/final_C.out 2>&1          && echo "C1/C2 done"
python scripts/interp_C3_saliency.py --gpu "$GPU" > runs/logs/final_C3.out 2>&1 && echo "C3 done"
python scripts/interp_D_subgroups.py    > runs/logs/final_D.out 2>&1          && echo "D done"
python scripts/interp_table_I1.py       > runs/logs/final_table_I1.out 2>&1   && echo "Table I1 done"
python scripts/finalize_outputs.py --replace-all --reason "$REASON"
