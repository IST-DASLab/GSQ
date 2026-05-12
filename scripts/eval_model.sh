#!/usr/bin/env bash
# ============================================================================
# GSQ — lm-eval benchmarks against a running vLLM server
# ============================================================================
# Usage:
#   VLLM_URL=http://localhost:8000/v1/completions bash scripts/eval_model.sh
#   CONFIG_FILE=configs/kimi-k2.5/kimi_k2.5_2bit_gptq_gsq.yaml \
#       RUN_ID=20260306-143025_a1b2c3 \
#       VLLM_URL=http://localhost:8000/v1/completions \
#       TASKS=gsm8k,arc_challenge \
#       bash scripts/eval_model.sh
# ============================================================================

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

CONFIG_FILE="${CONFIG_FILE:-configs/local/config.yaml}"
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1/completions}"
RUN_ID="${RUN_ID:-}"
TASKS="${TASKS:-gsm8k,arc_challenge,arc_easy,winogrande,piqa}"
NUM_CONCURRENT="${NUM_CONCURRENT:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
WANDB_FLAG="${WANDB_FLAG:-}"

[[ "${CONFIG_FILE}" != /* ]] && CONFIG_FILE="${REPO_ROOT}/${CONFIG_FILE}"

ARGS=(
    --config "${CONFIG_FILE}"
    --base-url "${VLLM_URL}"
    --tasks "${TASKS}"
    --num-concurrent "${NUM_CONCURRENT}"
)
[[ -n "${RUN_ID}" ]] && ARGS+=(--run-id "${RUN_ID}")
[[ -n "${OUTPUT_DIR}" ]] && ARGS+=(--output-dir "${OUTPUT_DIR}")
[[ -n "${WANDB_FLAG}" ]] && ARGS+=("${WANDB_FLAG}")

echo "=========================================="
echo "GSQ eval"
echo "vLLM URL    : ${VLLM_URL}"
echo "Config      : ${CONFIG_FILE}"
echo "Run ID      : ${RUN_ID:-<latest>}"
echo "Tasks       : ${TASKS}"
echo "Concurrent  : ${NUM_CONCURRENT}"
echo "=========================================="

cd "${REPO_ROOT}"
exec python "${REPO_ROOT}/eval_model.py" "${ARGS[@]}" "$@"
