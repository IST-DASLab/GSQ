#!/usr/bin/env bash
# ============================================================================
# GSQ — vLLM serving (single-node, bare metal)
# ============================================================================
# Launches a vLLM OpenAI-compatible server with tensor parallelism over all
# locally visible GPUs. Optionally runs lm-eval benchmarks once /health is up.
#
# Usage:
#   MODEL_PATH=/path/to/assembled bash scripts/serve_model.sh
#   RUN_ID=20260306-143025_a1b2c3 bash scripts/serve_model.sh
#   EVAL=1 MODEL_PATH=/path/to/assembled bash scripts/serve_model.sh
#
# Multi-node serving is intentionally not supported here — use a dedicated
# Ray-cluster setup if you need it.
# ============================================================================

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

MODEL_PATH="${MODEL_PATH:-}"
RUN_ID="${RUN_ID:-}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TP_SIZE="${TP_SIZE:-$(detect_num_gpus)}"
[[ "${TP_SIZE}" -lt 1 ]] && TP_SIZE=1

# --tokenizer-mode hf      : avoids garbled output on long-running serves (vLLM #35718)
# --mm-encoder-tp-mode data: required for Kimi-K2.5 (ViT dims not divisible by TP)
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---gpu-memory-utilization 0.85 --tokenizer-mode hf --mm-encoder-tp-mode data --max-num-seqs 4}"

EVAL="${EVAL:-0}"
EVAL_CONFIG_FILE="${EVAL_CONFIG_FILE:-configs/local/config.yaml}"
EVAL_TASKS="${EVAL_TASKS:-gsm8k,arc_challenge,arc_easy,winogrande,piqa}"
EVAL_NUM_CONCURRENT="${EVAL_NUM_CONCURRENT:-8}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"
EVAL_WANDB_FLAG="${EVAL_WANDB_FLAG:-}"

# Resolve MODEL_PATH from RUN_ID if needed.
if [[ -z "${MODEL_PATH}" ]]; then
    if [[ -z "${RUN_ID}" ]]; then
        echo "ERROR: set MODEL_PATH or RUN_ID before running." >&2
        exit 1
    fi
    CANDIDATE_DIR=$(find "${SCRATCH}/gsq/checkpoints" -type d -path "*/${RUN_ID}/assembled" 2>/dev/null | head -n1)
    if [[ -z "${CANDIDATE_DIR}" || ! -d "${CANDIDATE_DIR}" ]]; then
        echo "ERROR: no assembled model found for RUN_ID=${RUN_ID}" >&2
        exit 1
    fi
    MODEL_PATH="${CANDIDATE_DIR}"
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
fi
[[ -z "${EVAL_OUTPUT_DIR}" ]] && EVAL_OUTPUT_DIR="${MODEL_PATH}/evals"

# Resolve WANDB_RUN_ID (so eval can resume the same WandB run).
if [[ -z "${WANDB_RUN_ID:-}" && -n "${RUN_ID}" ]]; then
    PROGRESS_JSON=$(find "${SCRATCH}/gsq/checkpoints" -path "*/${RUN_ID}/progress.json" 2>/dev/null | head -n1)
    if [[ -f "${PROGRESS_JSON}" ]]; then
        WANDB_RUN_ID=$(python -c "import json; print(json.load(open('${PROGRESS_JSON}')).get('wandb_run_id',''))" 2>/dev/null || true)
        export WANDB_RUN_ID
    fi
fi

VLLM_ARGS=(
    "${MODEL_PATH}"
    --tensor-parallel-size "${TP_SIZE}"
    --trust-remote-code
    --host "${HOST}"
    --port "${PORT}"
    --max-model-len "${MAX_MODEL_LEN}"
)
# shellcheck disable=SC2206
EXTRA_ARRAY=(${EXTRA_VLLM_ARGS})
VLLM_ARGS+=("${EXTRA_ARRAY[@]}")

echo "=========================================="
echo "GSQ vLLM server (single node)"
echo "Model path : ${MODEL_PATH}"
echo "GPUs (TP)  : ${TP_SIZE}"
echo "URL        : http://${HOST}:${PORT}"
echo "  health   : http://${HOST}:${PORT}/health"
echo "  v1       : http://${HOST}:${PORT}/v1/completions"
echo "=========================================="

cd "${REPO_ROOT}"

# Sensible local cache locations to keep Triton/Inductor off the home filesystem.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH}/.triton_cache}"
export TRITON_HOME="${TRITON_HOME:-${SCRATCH}/.triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SCRATCH}/.inductor_cache}"
export TMPDIR="${TMPDIR:-${SCRATCH}/.tmp}"
mkdir -p "${TRITON_CACHE_DIR}" "${TRITON_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}"

# Launch the vLLM server in the background so we can optionally run eval.
python "${REPO_ROOT}/serve_vllm.py" --num-nodes 1 --port "${PORT}" "${VLLM_ARGS[@]}" &
SERVER_PID=$!

cleanup() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Stopping vLLM (pid ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "${EVAL}" = "1" ]]; then
    HEALTH_URL="http://127.0.0.1:${PORT}/health"
    echo "EVAL=1 — waiting for ${HEALTH_URL}..."
    HEALTHY=0
    for i in $(seq 1 180); do
        if curl -sf "${HEALTH_URL}" >/dev/null 2>&1; then
            echo "  Server healthy after ${i}*20s"
            HEALTHY=1
            break
        fi
        sleep 20
    done
    if [[ "${HEALTHY}" = "0" ]]; then
        echo "WARNING: server did not become healthy; skipping eval." >&2
    else
        EVAL_CONFIG_PATH="${EVAL_CONFIG_FILE}"
        [[ "${EVAL_CONFIG_PATH}" != /* ]] && EVAL_CONFIG_PATH="${REPO_ROOT}/${EVAL_CONFIG_PATH}"
        EVAL_ARGS=(
            --model-path "${MODEL_PATH}"
            --base-url "http://127.0.0.1:${PORT}/v1/completions"
            --tasks "${EVAL_TASKS}"
            --num-concurrent "${EVAL_NUM_CONCURRENT}"
            --config "${EVAL_CONFIG_PATH}"
        )
        [[ -n "${RUN_ID}" ]] && EVAL_ARGS+=(--run-id "${RUN_ID}")
        [[ -n "${EVAL_OUTPUT_DIR}" ]] && EVAL_ARGS+=(--output-dir "${EVAL_OUTPUT_DIR}")
        [[ -n "${WANDB_RUN_ID:-}" ]] && EVAL_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
        [[ -n "${EVAL_WANDB_FLAG}" ]] && EVAL_ARGS+=("${EVAL_WANDB_FLAG}")

        echo "Running lm-eval: tasks=${EVAL_TASKS}"
        python "${REPO_ROOT}/eval_model.py" "${EVAL_ARGS[@]}"
        echo "Eval finished. Server still up; Ctrl-C to stop or KEEP_SERVING=0 to exit."
    fi
fi

# Either EVAL=0 or eval finished — keep the server alive until it exits or we're killed.
KEEP_SERVING="${KEEP_SERVING:-1}"
if [[ "${KEEP_SERVING}" = "1" ]]; then
    wait "${SERVER_PID}"
fi
