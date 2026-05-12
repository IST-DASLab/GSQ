#!/usr/bin/env bash
# ============================================================================
# GSQ — Model assembly (Slurm)
# ============================================================================
# Purpose : Assemble per-layer quantized shards into a full HuggingFace model.
#           Runs save_model.py on a single node (CPU-only, no GPU needed).
#
# Note    : Uses --mpi=pmix on the srun line (matching run.sbatch.sh) to
#           avoid Pyxis/enroot PMIx overlay mount errors.
#
# Usage   : sbatch scripts/save_model.sbatch.sh
#
# Before submitting:
#   1. Ensure a training run has completed (progress.json exists).
#   2. Edit the "User configuration" block below.
# ============================================================================

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=gsq-save
#SBATCH --account=a-g200
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --no-requeue
#SBATCH --output=logs/slurm/gsq_save_%j.out
#SBATCH --error=logs/slurm/gsq_save_%j.err

set -euo pipefail

# ============================================================================
# User configuration
# ============================================================================

CONFIG_FILE=${CONFIG_FILE:-"configs/kimi-k2.5/kimi_k2.5_2bit_gptq_gsq.yaml"}

# Run ID to export. Leave empty to export the latest completed run.
RUN_ID=${RUN_ID:-}

# Output directory. Leave empty to use the default (<run_dir>/assembled).
OUT_DIR=${OUT_DIR:-}

# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EDF_FILE="${SCRIPT_ROOT}/scripts/gsq-cuda.toml"
SCRATCH="${SCRATCH:-"${SCRIPT_ROOT}/runtime"}"
if [[ "${EDF_FILE}" == *"gsq-cuda"* ]]; then
    VENV_PATH="${SCRATCH}/gsq/venv-gsq-cuda"
else
    VENV_PATH="${SCRATCH}/gsq/venv-gsq"
fi

# ── Load secrets ─────────────────────────────────────────────────────────────
if [ -f "${SCRIPT_ROOT}/.env" ]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' "${SCRIPT_ROOT}/.env" | xargs)
fi

# HuggingFace cache (needed for AutoConfig.from_pretrained)
export HF_HOME="${SCRATCH}/.hf"
export HF_DATASETS_CACHE="${SCRATCH}/.hf/datasets"

# ── Directories ──────────────────────────────────────────────────────────────
mkdir -p "${SCRIPT_ROOT}/logs/slurm"

# ── Job info ─────────────────────────────────────────────────────────────────
echo "=========================================="
echo "GSQ Model Assembly"
echo "Start time   : $(date)"
echo "Host         : $(hostname)"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Config       : ${CONFIG_FILE}"
echo "Run ID       : ${RUN_ID:-<latest>}"
echo "Output dir   : ${OUT_DIR:-<default>}"
echo "Venv         : ${VENV_PATH}"
echo "=========================================="

ulimit -c 0

cd "${SCRIPT_ROOT}"

# Build CLI arguments
SAVE_ARGS="--config ${SCRIPT_ROOT}/${CONFIG_FILE}"
if [ -n "${RUN_ID}" ]; then
    SAVE_ARGS="${SAVE_ARGS} --run-id ${RUN_ID}"
fi
if [ -n "${OUT_DIR}" ]; then
    SAVE_ARGS="${SAVE_ARGS} --out-dir ${OUT_DIR}"
fi

# ── Launch with container ─────────────────────────────────────────────────────
srun -ul --mpi=pmix --environment="${EDF_FILE}" bash -c "
set -euo pipefail
source ${VENV_PATH}/bin/activate

unset SSL_CERT_FILE
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
else
    export SSL_CERT_FILE=\$(python -c 'import certifi; print(certifi.where())')
fi

python ${SCRIPT_ROOT}/save_model.py ${SAVE_ARGS}
"

echo "=========================================="
echo "End time: $(date)"
echo "Model assembly completed."
echo "=========================================="
