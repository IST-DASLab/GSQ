#!/usr/bin/env bash
# ============================================================================
# GSQ — Benchmark evaluation (Slurm)
# ============================================================================
# Purpose : Run lm-evaluation-harness against a running vLLM server and log
#           results to WandB. This is a lightweight client job (no GPUs).
#
# Usage   : sbatch scripts/eval_model.sbatch.sh
#
# Before submitting:
#   1. Start the vLLM server: sbatch scripts/serve_model.sbatch.sh
#   2. Check the serve job's Slurm output for the server URL.
#   3. Set VLLM_URL below to that URL.
# ============================================================================

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=gsq-eval
#SBATCH --account=a-g200          # <-- replace with your project account
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --no-requeue
#SBATCH --output=logs/slurm/gsq_eval_%j.out
#SBATCH --error=logs/slurm/gsq_eval_%j.err

set -euo pipefail

# ============================================================================
# User configuration — edit these before submitting
# ============================================================================

# vLLM server completions URL (from serve_model.sbatch.sh output)
VLLM_URL="http://CHANGE_ME:8000/v1/completions"

# Training config file (used to resolve model path and WandB run ID).
# Available configs:
#   configs/kimi-k2.5/kimi_k2.5_2bit_gptq_gsq.yaml — Kimi-K2.5 (384 experts)
#   configs/qwen3/qwen3_235B_A22B.yaml — Qwen3-235B-A22B (128 experts)
#   configs/qwen35/qwen35_397B_A17B.yaml — Qwen3.5-397B-A17B (512 experts, hybrid attn)
CONFIG_FILE="configs/kimi-k2.5/kimi_k2.5_2bit_gptq_gsq.yaml"

# Training run ID. Leave empty to use the latest completed run.
RUN_ID=""

# Benchmark tasks (comma-separated lm-eval task names)
TASKS="gsm8k,arc_challenge,arc_easy,winogrande,piqa"

# Number of concurrent requests to the vLLM server
NUM_CONCURRENT=8

# Output directory for lm-eval results. Leave empty for default.
OUTPUT_DIR=""

# Set to "--no-wandb" to skip WandB logging
WANDB_FLAG=""

# ============================================================================
# Paths — do not edit unless your workspace layout differs
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

# ── Load secrets (.env is gitignored; contains WANDB_API_KEY) ────────────────
if [ -f "${SCRIPT_ROOT}/.env" ]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' "${SCRIPT_ROOT}/.env" | xargs)
fi

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p "${SCRIPT_ROOT}/logs/slurm"

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ "${VLLM_URL}" == *"CHANGE_ME"* ]]; then
    echo "ERROR: VLLM_URL has not been set. Edit the script with the server URL"
    echo "       from serve_model.sbatch.sh output before submitting."
    exit 1
fi

# ── Job info ──────────────────────────────────────────────────────────────────
echo "=========================================="
echo "GSQ Benchmark Evaluation"
echo "Start time   : $(date)"
echo "Host         : $(hostname)"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "vLLM URL     : ${VLLM_URL}"
echo "Config       : ${CONFIG_FILE}"
echo "Run ID       : ${RUN_ID:-<latest>}"
echo "Tasks        : ${TASKS}"
echo "Concurrent   : ${NUM_CONCURRENT}"
echo "=========================================="

ulimit -c 0
cd "${SCRIPT_ROOT}"

# Build CLI arguments
EVAL_ARGS="--config ${SCRIPT_ROOT}/${CONFIG_FILE}"
EVAL_ARGS="${EVAL_ARGS} --base-url ${VLLM_URL}"
EVAL_ARGS="${EVAL_ARGS} --tasks ${TASKS}"
EVAL_ARGS="${EVAL_ARGS} --num-concurrent ${NUM_CONCURRENT}"
if [ -n "${RUN_ID}" ]; then
    EVAL_ARGS="${EVAL_ARGS} --run-id ${RUN_ID}"
fi
if [ -n "${OUTPUT_DIR}" ]; then
    EVAL_ARGS="${EVAL_ARGS} --output-dir ${OUTPUT_DIR}"
fi
if [ -n "${WANDB_FLAG}" ]; then
    EVAL_ARGS="${EVAL_ARGS} ${WANDB_FLAG}"
fi

# ── Launch with Container Engine ─────────────────────────────────────────────
srun -ul --mpi=pmix --environment="${EDF_FILE}" bash -c "
set -euo pipefail

source ${VENV_PATH}/bin/activate

# SSL fix
unset SSL_CERT_FILE
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
else
    export SSL_CERT_FILE=\$(python -c 'import certifi; print(certifi.where())')
fi

python ${SCRIPT_ROOT}/eval_model.py ${EVAL_ARGS}
"

echo "=========================================="
echo "End time: $(date)"
echo "Evaluation completed."
echo "=========================================="
