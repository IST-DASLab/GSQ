#!/usr/bin/env bash
# ============================================================================
# GSQ — vLLM serving + optional evaluation (Slurm)
# ============================================================================
# Purpose : Launch a vLLM OpenAI-compatible server for inference/evaluation.
#           Supports multi-node serving via Ray + tensor/pipeline parallelism.
#           Optionally runs lm-eval benchmarks once the server is healthy, then
#           keeps the server up until the job times out or is cancelled.
#
# Usage   : sbatch scripts/serve_model.sbatch.sh
#
# Before submitting:
#   1. Ensure the model has been assembled (run save_model.py first).
#   2. Edit the "User configuration" block below.
#   3. Set EVAL=1 to automatically run benchmarks after the server is ready.
#   4. After the job starts, check the Slurm output for the server URL.
#
# To monitor: tail -f logs/slurm/gsq_serve_<job_id>.out
#   The serve script waits for all Ray nodes and polls /health until the server is
#   up, then runs a quick completions test. Check the .out log for "Server is up."
#
# To stop : scancel <job_id>
#
# Serving node guidance (4 GPUs/node, TP spans all GPUs via Ray):
#   Kimi K2.5          — 2 nodes ( 8 GPUs, TP=8)
#   Qwen3-235B-A22B    — 2 nodes ( 8 GPUs, TP=8)
#   Qwen3.5-397B-A17B  — 4 nodes (16 GPUs, TP=16)
# ============================================================================

# ── Slurm directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=gsq-serve
#SBATCH --account=a-g200          
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=4                 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G                   
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --output=logs/slurm/gsq_serve_%j.out
#SBATCH --error=logs/slurm/gsq_serve_%j.err

set -euo pipefail

RUN_ID=${RUN_ID:-}
MODEL_PATH=${MODEL_PATH:-}

PORT=8000

MAX_MODEL_LEN="4096"

# --tokenizer-mode hf               : prevents garbled output on extended serving (vLLM issue #35718)
# --mm-encoder-tp-mode data         : run vision encoder in data-parallel (required for Kimi-K2.5;
#                                      ViT dims are not evenly divisible by TP, causing cuBLAS errors)
EXTRA_VLLM_ARGS="--gpu-memory-utilization 0.85 --tokenizer-mode hf --mm-encoder-tp-mode data --max-num-seqs 4"

# ── Evaluation (optional) ─────────────────────────────────────────────────────
# Set EVAL=1 to automatically run lm-eval benchmarks once the server is healthy.
# The server stays running after eval completes (until the job times out).
EVAL=${EVAL:-0}

# Training config file (used to resolve model path and WandB run ID).
# Available configs:
#   configs/kimi-k2.5/kimi_k2.5_2bit_gptq_gsq.yaml — Kimi-K2.5 (384 experts)
#   configs/qwen3/qwen3_235B_A22B.yaml — Qwen3-235B-A22B
#   configs/qwen35/qwen35_397B_A17B.yaml — Qwen3.5-397B-A17B
EVAL_CONFIG_FILE="configs/kimi-k2.5/kimi_k2.5_2bit_gptq_gsq.yaml"

# Benchmark tasks (comma-separated lm-eval task names)
EVAL_TASKS="gsm8k,arc_challenge,arc_easy,winogrande,piqa"

# Number of concurrent requests to the vLLM server
EVAL_NUM_CONCURRENT=8

# Output directory for lm-eval results and model write-outs.
# Leave empty to use <model_path>/evals (set after MODEL_PATH is resolved below).
EVAL_OUTPUT_DIR=""

# Set to "--no-wandb" to skip WandB logging for eval
EVAL_WANDB_FLAG=""

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

if [ -f "${SCRIPT_ROOT}/.env" ]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' "${SCRIPT_ROOT}/.env" | xargs)
fi

mkdir -p "${SCRIPT_ROOT}/logs/slurm"

if [ -z "${MODEL_PATH}" ]; then
    if [ -z "${RUN_ID}" ]; then
        echo "ERROR: MODEL_PATH is unset and RUN_ID is not specified."
        exit 1
    fi
    CANDIDATE_DIR=$(find "${SCRATCH}/gsq/checkpoints" -type d -path "*/${RUN_ID}/assembled" | head -n1)
    if [ -z "$CANDIDATE_DIR" ] || [ ! -d "$CANDIDATE_DIR" ]; then
        echo "ERROR: Could not find assembled model directory for RUN_ID: ${RUN_ID}"
        exit 1
    else
        MODEL_PATH="$CANDIDATE_DIR"
    fi
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}"
    exit 1
fi

# Default eval output dir: <model_path>/evals (matches eval_model.py default)
if [ -z "${EVAL_OUTPUT_DIR}" ]; then
    EVAL_OUTPUT_DIR="${MODEL_PATH}/evals"
fi

# Resolve the WandB run ID from progress.json so eval can resume the training run.
# RUN_ID is the checkpoint run ID (filesystem key, e.g. 20260309-051218_968da2).
# WANDB_RUN_ID is the actual WandB project run ID stored in progress.json under
# "wandb_run_id" — this is what wandb.init(id=...) needs to resume the correct run.
# Only attempted when RUN_ID is known and no explicit WANDB_RUN_ID override is provided.
if [ -z "${WANDB_RUN_ID:-}" ] && [ -n "${RUN_ID:-}" ]; then
    PROGRESS_JSON=$(find "${SCRATCH}/gsq/checkpoints" -path "*/${RUN_ID}/progress.json" | head -n1)
    if [ -f "${PROGRESS_JSON}" ]; then
        WANDB_RUN_ID=$(uv run python3 -c "import json; d=json.load(open('${PROGRESS_JSON}')); print(d.get('wandb_run_id',''))" 2>/dev/null || true)
    fi
fi

NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=$(echo "$NODELIST" | head -n1)
HEAD_ADDR=$(srun --mpi=pmix --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    --environment="${EDF_FILE}" hostname -i | tail -n1)
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4

echo "=========================================="
echo "GSQ vLLM Serving"
echo "Start time   : $(date)"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Nodes        : ${NUM_NODES} (${GPUS_PER_NODE} GPUs each)"
echo "Total GPUs   : $((NUM_NODES * GPUS_PER_NODE))"
echo "Head node    : ${HEAD_NODE} (${HEAD_ADDR})"
echo "Model path   : ${MODEL_PATH}"
echo "Port         : ${PORT}"
echo "=========================================="

ulimit -c 0
cd "${SCRIPT_ROOT}"

RAY_PORT=6379

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
VLLM_ARGS="${MODEL_PATH}"
VLLM_ARGS="${VLLM_ARGS} --tensor-parallel-size ${TOTAL_GPUS}"
if [ "${NUM_NODES}" -gt 1 ]; then
    VLLM_ARGS="${VLLM_ARGS} --distributed-executor-backend ray"
fi
VLLM_ARGS="${VLLM_ARGS} --trust-remote-code"
VLLM_ARGS="${VLLM_ARGS} --host 0.0.0.0"
VLLM_ARGS="${VLLM_ARGS} --port ${PORT}"
if [ -n "${MAX_MODEL_LEN}" ]; then
    VLLM_ARGS="${VLLM_ARGS} --max-model-len ${MAX_MODEL_LEN}"
fi
if [ -n "${EXTRA_VLLM_ARGS}" ]; then
    VLLM_ARGS="${VLLM_ARGS} ${EXTRA_VLLM_ARGS}"
fi

PREAMBLE_SCRIPT="${SCRATCH}/gsq/logs/_serve_preamble.sh"
mkdir -p "$(dirname "${PREAMBLE_SCRIPT}")"
cat > "${PREAMBLE_SCRIPT}" <<PREAMBLE_EOF
set -euo pipefail
source ${VENV_PATH}/bin/activate
unset SSL_CERT_FILE
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
else
    export SSL_CERT_FILE=\$(python -c 'import certifi; print(certifi.where())')
fi
export TRITON_CACHE_DIR=${SCRATCH}/.triton_cache
export TRITON_HOME=${SCRATCH}/.triton
export TORCHINDUCTOR_CACHE_DIR=${SCRATCH}/.inductor_cache
export TMPDIR=${SCRATCH}/.tmp
export RAY_TMPDIR=/tmp
export RAY_raylet_start_wait_time_s=60
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
mkdir -p \${TRITON_CACHE_DIR} \${TRITON_HOME} \${TORCHINDUCTOR_CACHE_DIR} \${TMPDIR}
PREAMBLE_EOF


# Start Ray workers on non-head nodes (background; sleep infinity keeps container alive)
if [ "${NUM_NODES}" -gt 1 ]; then
    WORKER_NODES=$(echo "$NODELIST" | tail -n +2)
    for WORKER in $WORKER_NODES; do
        echo "Starting Ray worker on ${WORKER}..."
        srun --mpi=pmix --nodes=1 --ntasks=1 -w "$WORKER" \
            --mpi=pmix --environment="${EDF_FILE}" bash -c "
source ${PREAMBLE_SCRIPT}
WORKER_IP=\$(hostname -i)
export VLLM_HOST_IP=\${WORKER_IP}
echo \"[\${HOSTNAME}] Waiting for Ray head at ${HEAD_ADDR}:${RAY_PORT}...\"
for i in \$(seq 1 60); do
    if python -c \"import socket; s=socket.socket(); s.settimeout(2); s.connect(('${HEAD_ADDR}', ${RAY_PORT})); s.close()\" 2>/dev/null; then
        echo \"[\${HOSTNAME}] Ray head reachable after \${i}s\"
        break
    fi
    sleep 1
done
ray start --address=${HEAD_ADDR}:${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --node-ip-address=\${WORKER_IP}
echo \"[\${HOSTNAME}] Ray worker started (WORKER_IP=\${WORKER_IP})\"
sleep infinity
" &
    done
fi

echo ""
echo "=========================================="
echo "  vLLM Server URL: http://${HEAD_ADDR}:${PORT}"
echo ""
echo "  Completions : http://${HEAD_ADDR}:${PORT}/v1/completions"
echo "  Health check: http://${HEAD_ADDR}:${PORT}/health"
echo ""
echo "  Use this URL in eval_model.sbatch.sh or eval_model.py:"
echo "    --base-url http://${HEAD_ADDR}:${PORT}/v1/completions"
echo "=========================================="
echo ""

# Head node: start Ray head, then launch vLLM (all in one srun / one container).
# This avoids GPU double-allocation and keeps the Ray head daemon alive.
# Run in the background so the outer shell can run eval once the server is healthy.
echo "Starting Ray head + vLLM server on ${HEAD_NODE}..."
srun --mpi=pmix --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    --mpi=pmix --environment="${EDF_FILE}" bash -c "
source ${PREAMBLE_SCRIPT}
export VLLM_HOST_IP=${HEAD_ADDR}
ray start --head --port=${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --node-ip-address=${HEAD_ADDR}
echo \"[\${HOSTNAME}] Ray head started (${HEAD_ADDR}:${RAY_PORT})\"

sleep 2

export RAY_ADDRESS=${HEAD_ADDR}:${RAY_PORT}
echo \"[\${HOSTNAME}] Launching serve_vllm.py (VLLM_HOST_IP=\${VLLM_HOST_IP}, RAY_ADDRESS=\${RAY_ADDRESS})\"
cd ${SCRIPT_ROOT}
exec python serve_vllm.py --num-nodes ${NUM_NODES} --port ${PORT} ${VLLM_ARGS}
" &
HEAD_SRUN_PID=$!

# ── Optional evaluation ───────────────────────────────────────────────────────
if [ "${EVAL}" = "1" ]; then
    HEALTH_URL="http://${HEAD_ADDR}:${PORT}/health"
    echo ""
    echo "EVAL=1 — waiting for server to become healthy before running benchmarks..."
    echo "  Polling: ${HEALTH_URL}"
    HEALTHY=0
    for i in $(seq 1 180); do
        if curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
            echo "  Server is healthy after ${i}s."
            HEALTHY=1
            break
        fi
        sleep 20
    done

    if [ "${HEALTHY}" = "0" ]; then
        echo "WARNING: Server did not become healthy within 60 minutes. Skipping eval."
    else
        echo ""
        echo "=========================================="
        echo "Running lm-eval benchmarks"
        echo "Tasks      : ${EVAL_TASKS}"
        echo "Concurrent : ${EVAL_NUM_CONCURRENT}"
        echo "Config     : ${EVAL_CONFIG_FILE}"
        echo "=========================================="

        # Use the same model path the server is serving (--model-path overrides config resolution).
        EVAL_ARGS="--model-path ${MODEL_PATH}"
        EVAL_ARGS="${EVAL_ARGS} --base-url http://${HEAD_ADDR}:${PORT}/v1/completions"
        EVAL_ARGS="${EVAL_ARGS} --tasks ${EVAL_TASKS}"
        EVAL_ARGS="${EVAL_ARGS} --num-concurrent ${EVAL_NUM_CONCURRENT}"
        if [ -n "${RUN_ID}" ]; then
            EVAL_ARGS="${EVAL_ARGS} --run-id ${RUN_ID}"
        fi
        if [ -n "${EVAL_CONFIG_FILE}" ]; then
            EVAL_ARGS="${EVAL_ARGS} --config ${SCRIPT_ROOT}/${EVAL_CONFIG_FILE}"
        fi
        if [ -n "${EVAL_OUTPUT_DIR}" ]; then
            EVAL_ARGS="${EVAL_ARGS} --output-dir ${EVAL_OUTPUT_DIR}"
        fi
        if [ -n "${WANDB_RUN_ID:-}" ]; then
            EVAL_ARGS="${EVAL_ARGS} --wandb-run-id ${WANDB_RUN_ID}"
        fi
        if [ -n "${EVAL_WANDB_FLAG}" ]; then
            EVAL_ARGS="${EVAL_ARGS} ${EVAL_WANDB_FLAG}"
        fi

        unset SSL_CERT_FILE
        if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
            export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
        else
            export SSL_CERT_FILE=$(uv run python -c 'import certifi; print(certifi.where())')
        fi
        # shellcheck disable=SC2086
        uv run python "${SCRIPT_ROOT}/eval_model.py" ${EVAL_ARGS}
        echo "=========================================="
        echo "Evaluation completed at: $(date)"
        echo "=========================================="
    fi
else
    # Keep the job alive until the vLLM server exits (timeout or scancel)
    wait ${HEAD_SRUN_PID}
fi

echo "=========================================="
echo "End time: $(date)"
echo "vLLM server stopped."
echo "=========================================="
