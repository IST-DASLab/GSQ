#!/usr/bin/env bash
# ============================================================================
# GSQ — Production Slurm job
# ============================================================================
# Purpose : Run Gumbel Softmax Quantization on a target LLM.
#           Supports single-node (LLaMA, OPT) and multi-node multi-GPU
#           (Kimi K2.5, Qwen3-MoE) configurations.
#
# Usage   : sbatch scripts/run.sbatch.sh
#
# Smoke test (full e2e pipeline, minimal compute):
#   SMOKE_TEST=1 sbatch --partition=debug --nodes=1 --ntasks-per-node=1 \
#       --time=00:30:00 --mem=50G scripts/run.sbatch.sh
#
# Before submitting:
#   1. Edit the "User configuration" block below.
#   2. Ensure the target model is pre-downloaded to $HF_HOME.
#   3. Run a smoke test first:
#        SMOKE_TEST=1 sbatch --partition=debug --nodes=1 --ntasks-per-node=1 \
#            --time=00:30:00 --mem=50G scripts/run.sbatch.sh
#
# Architecture notes:
#   - Dense models (LLaMA, OPT) : NODES=1, NTASKS_PER_NODE=1
#   - MoE models (Kimi K2, Qwen3, Qwen3.5): NODES>=1, NTASKS_PER_NODE=4
#     Expert parallelism distributes MoE layers across GPUs automatically via
#     KimiK2DistributedWrapper / Qwen3MoeDistributedWrapper / Qwen35MoeDistributedWrapper.
#   - Kimi-K2.5 (384 experts): 2+ nodes recommended for reasonable runtime.
#     With 8 GPUs the model fits; 16-32 GPUs reduce wall-clock time linearly.
#   - Qwen3-235B-A22B (128 experts, 94 layers):
#     2 nodes (8 GPUs) — 16 experts/GPU; 4 nodes — 8/GPU; 8 nodes — 4/GPU
#   - Qwen3.5-397B-A17B (512 experts, 60 layers, hybrid attention):
#     4 nodes (16 GPUs) — 32 experts/GPU; 8 nodes — 16/GPU; 16 nodes — 8/GPU
#     Requires transformers >= 5.3.0 for Qwen3.5 MoE model support.
#
# Checkpoint behavior:
#   GSQ writes one .safetensors shard per layer to checkpoint_dir.
#   Use --resume to restart from the last completed layer.
# ============================================================================

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=gsq-run
#SBATCH --account=a-g200          # <-- replace with your project account
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=8                 # <-- Kimi-K2.5: 2-8 nodes recommended (8-32 GPUs)
#SBATCH --ntasks-per-node=4       # <-- 4 for MoE (one per GPU); 1 for dense models
#SBATCH --gpus-per-task=1         # always 1: ensures correct CUDA_VISIBLE_DEVICES per rank
#SBATCH --cpus-per-task=16
# Note: --exclusive already allocates all node memory; no --mem flag needed here
#SBATCH --exclusive               # prevent resource contention on multi-node jobs
#SBATCH --no-requeue
#SBATCH --signal=SIGUSR2@600      # warn 10 min before wall time (for graceful shutdown)
# Performance constraints for LLM workloads on GH200:
#   thp_never          : disable Transparent Hugepages (reduces fragmentation)
#   nvidia_vboost_enabled : enable GPU voltage boost for higher sustained throughput
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --output=logs/slurm/gsq_run_%j.out
#SBATCH --error=logs/slurm/gsq_run_%j.err

set -euo pipefail

# ============================================================================
# User configuration — edit these before submitting
# ============================================================================

# Config file relative to project root
# Cluster config with scratch paths (see README_CLARIDEN.md)
CONFIG_FILE=${CONFIG_FILE:-"configs/kimi-k2.5/kimi_k2.5_2bit_gptq_gsq.yaml"}

# Optional: resume from a previous run's checkpoint directory on scratch
# Set to "" to start fresh
RESUME_FROM="${RESUME_FROM:-}"

# WandB run name prefix (leave empty to use config defaults)
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

# ── Smoke test ───────────────────────────────────────────────────────────────
# Set SMOKE_TEST=1 to run the full pipeline end-to-end with minimal compute:
#   - Uses LLaMA-3.2-1B instead of the production model
#   - 16 samples, 1 epoch, 2 layers only
#   - WandB disabled
#
# Usage:
#   SMOKE_TEST=1 sbatch --partition=debug --nodes=1 --ntasks-per-node=1 \
#       --time=00:30:00 --mem=50G scripts/run.sbatch.sh
SMOKE_TEST="${SMOKE_TEST:-0}"

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

# ── Load secrets (.env is gitignored; contains HF_TOKEN, WANDB_API_KEY) ─────
if [ -f "${SCRIPT_ROOT}/.env" ]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' "${SCRIPT_ROOT}/.env" | xargs)
fi

# ── Validate required variables ───────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN is not set. Gated models (Kimi K2.5, LLaMA) will fail to download."
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is not set. WandB logging will be disabled or fail."
fi

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p "${SCRIPT_ROOT}/logs/slurm"

# Scratch layout:
#   $SCRATCH/gsq/
#     checkpoints/   ← per-run checkpoint shards (managed by main.py run_id logic)
#     logs/          ← Python training logs
#     torch-dist/    ← torch distributed debug logs, one subdir per job
#     smoke/         ← ephemeral smoke-test outputs
OUTPUT_DIR="${SCRATCH}/gsq/torch-dist/job${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH}/gsq/checkpoints"
mkdir -p "${SCRATCH}/gsq/logs"

# Disable core dumps (each core file can be hundreds of GB)
ulimit -c 0

cd "${SCRIPT_ROOT}"

# Build optional resume argument
RESUME_ARG=""
if [ -n "${RESUME_FROM}" ]; then
    RESUME_ARG="--resume ${RESUME_FROM}"
fi

# Resolve CONFIG_FILE to absolute path
[[ "${CONFIG_FILE}" != /* ]] && CONFIG_FILE="${SCRIPT_ROOT}/${CONFIG_FILE}"

# ── Smoke-test config override ───────────────────────────────────────────────
EXTRA_ARGS=""
if [ "${SMOKE_TEST}" = "1" ]; then
    SMOKE_DIR="${SCRATCH}/gsq/smoke/job${SLURM_JOB_ID}"
    mkdir -p "${SMOKE_DIR}"
    CONFIG_FILE="${SMOKE_DIR}/config_smoke.yaml"  # generated from configs/config_smoke.yaml
    # Expand $SCRATCH and $SLURM_JOB_ID into the template; all other variables
    # (e.g. $USER inside the yaml) are left untouched.
    envsubst '${SCRATCH} ${SLURM_JOB_ID}' \
        < "${SCRIPT_ROOT}/configs/config_smoke.yaml" \
        > "${CONFIG_FILE}"
    EXTRA_ARGS="--max-layers 2"
fi

# ── Job info ─────────────────────────────────────────────────────────────────
echo "=========================================="
if [ "${SMOKE_TEST}" = "1" ]; then
    echo "GSQ Smoke Test"
else
    echo "GSQ Production Job"
fi
echo "Start time   : $(date)"
echo "Host         : $(hostname)"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Nodes        : ${SLURM_NNODES} x ${SLURM_NTASKS_PER_NODE} tasks x 1 GPU"
echo "Total GPUs   : $((SLURM_NNODES * SLURM_NTASKS_PER_NODE))"
echo "Config       : ${CONFIG_FILE}"
echo "Resume from  : ${RESUME_FROM:-<fresh start>}"
echo "Output dir   : ${OUTPUT_DIR}"
if [ "${SMOKE_TEST}" = "1" ]; then
    echo "Smoke test   : ON (LLaMA-3.2-1B, 16 samples, 1 epoch, 2 layers)"
fi
echo "=========================================="

# ── Launch with Container Engine ─────────────────────────────────────────────
srun -ul --mpi=pmix --environment="${EDF_FILE}" --label bash -c "
set -euo pipefail

# ── Activate venv ─────────────────────────────────────────────────────────
source ${VENV_PATH}/bin/activate

# NCCL, Libfabric, and PyTorch env vars are set in gsq.toml [env].

# ── OMP threads: one per CPU allocated per task ───────────────────────────
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}

# ── Triton JIT cache: keep off Lustre ────────────────────────────────────
# /dev/shm is a per-node RAM disk; avoids Lustre metadata pressure from JIT.
export TRITON_HOME=/dev/shm/triton_\${SLURM_JOB_ID}

# ── SSL fix ───────────────────────────────────────────────────────────────
unset SSL_CERT_FILE
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
else
    export SSL_CERT_FILE=\$(python -c 'import certifi; print(certifi.where())')
fi

# ── Resolve master address (first node in the Slurm allocation) ───────────
MASTER_ADDR=\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\" | head -n1)
echo \"Master addr  : \$MASTER_ADDR\"
echo \"This rank    : \$SLURM_PROCID / \$SLURM_NTASKS (local \$SLURM_LOCALID)\"
echo \"GPUs on node : \$SLURM_GPUS_ON_NODE\"

# ── Distributed env (one process per srun task; each task has 1 GPU) ─────
# With --gpus-per-task=1, each task sees a single GPU as cuda:0. We run
# main.py once per task (no torchrun) and set rank env vars from Slurm.
export RANK=\${SLURM_PROCID}
export LOCAL_RANK=\${SLURM_LOCALID}
export WORLD_SIZE=\${SLURM_NTASKS}
export MASTER_ADDR=\${MASTER_ADDR}
export MASTER_PORT=29500

# ── Launch one process per GPU ───────────────────────────────────────────
python ${SCRIPT_ROOT}/main.py \\
    --config ${CONFIG_FILE} \\
    ${RESUME_ARG} ${EXTRA_ARGS}

EXIT=\$?
if [ \$EXIT -ne 0 ]; then
    echo 'GSQ exited with status '\$EXIT
    if [ -d \"${OUTPUT_DIR}/torch_logs\" ]; then
        echo '--- Torch distributed logs ---'
        find \"${OUTPUT_DIR}/torch_logs\" -type f -name '*.log' \\
            -exec echo '--- {} ---' \\; -exec cat {} \\;
    fi
    exit \$EXIT
fi
"

echo "=========================================="
echo "End time: $(date)"
echo "Run completed."
echo ""
echo "Checkpoint shards : ${SCRATCH}/gsq/checkpoints/"
echo "Training logs     : ${SCRATCH}/gsq/logs/"
echo "Torch dist logs   : ${OUTPUT_DIR}"
echo ""
echo "To archive completed checkpoints (example — copy to your long-term storage):"
echo "  rsync -av ${SCRATCH}/gsq/checkpoints/ \"${SCRIPT_ROOT}/archives/checkpoints/\""
echo "=========================================="
