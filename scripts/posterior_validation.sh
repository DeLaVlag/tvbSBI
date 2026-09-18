#!/usr/bin/env bash
# Run inside the SBI environment, or through scripts/posterior_validation_width_recovery.sbatch.
set -euo pipefail
if (( $# < 3 )); then
    printf 'Usage: %s CHECKPOINT TRAIN_N_TIME TRAIN_DT [validation/simulator arguments...]\n' "$0" >&2
    exit 2
fi
checkpoint="$1"
training_n_time="$2"
training_dt="$3"
shift 3
if [[ -z "${PROJECT:-}" ]]; then
    PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export PROJECT
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$PROJECT"
if [[ ! -r "$checkpoint" ]]; then
    printf 'Checkpoint is not readable: %s\n' "$checkpoint" >&2
    exit 2
fi
# Small in-sample demonstration; not held-out calibration. CLI flags may override these counts.
exec "${PYTHON:-python}" -u -m tvbgpu.Posterior_tests \
    --checkpoint "$checkpoint" --tests width,recovery \
    --width-items 2 --width-samples 100 \
    --recovery-items 2 --recovery-samples 100 \
    -n "$training_n_time" -dt "$training_dt" "$@"
