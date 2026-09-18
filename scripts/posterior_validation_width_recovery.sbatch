#!/usr/bin/env bash
# Submit from the repository root; supply site account/partition as sbatch options.
#SBATCH --job-name=sbi-postval-width
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --output=postval-width-%j.out
#SBATCH --error=postval-width-%j.err
set -euo pipefail
export PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:?Submit from the repository root or set PROJECT}}"
: "${IMAGE:?Set IMAGE to the external SBI Apptainer image}"
if [[ "$PROJECT" != /* || "$IMAGE" != /* ]]; then
    printf 'PROJECT and IMAGE must be absolute paths\n' >&2
    exit 2
fi
if [[ ! -f "$PROJECT/scripts/posterior_validation.sh" || ! -r "$IMAGE" ]]; then
    printf 'Check PROJECT and IMAGE: launcher or container missing\n' >&2
    exit 2
fi
cache_root="${TMPDIR:-/tmp}/sbi_postval_${USER}_${SLURM_JOB_ID}"
mkdir -p "$cache_root/pycuda" "$cache_root/xdg" "$cache_root/mpl" "$cache_root/numba"
srun --mpi=none --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    apptainer exec --cleanenv --no-home --nv \
    --bind "${PROJECT}:${PROJECT}" --bind "${cache_root}:${cache_root}" --pwd "$PROJECT" \
    --env "PROJECT=${PROJECT}" --env "PYTHONPATH=${PROJECT}" \
    --env "PYTHON=/opt/sbi/bin/python" --env "SLURM_LOCALID=0" \
    --env "PYCUDA_CACHE_DIR=${cache_root}/pycuda" \
    --env "XDG_CACHE_HOME=${cache_root}/xdg" \
    --env "MPLCONFIGDIR=${cache_root}/mpl" \
    --env "NUMBA_CACHE_DIR=${cache_root}/numba" \
    "$IMAGE" bash "$PROJECT/scripts/posterior_validation.sh" "$@"
