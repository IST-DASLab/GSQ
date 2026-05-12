#!/usr/bin/env bash
# ============================================================================
# GSQ — assemble per-layer shards into a HuggingFace checkpoint
# ============================================================================
# Usage:
#   bash scripts/save_model.sh
#   CONFIG_FILE=configs/kimi-k2.5/kimi_k2.5_2bit_gptq_gsq.yaml \
#       RUN_ID=20260306-143025_a1b2c3 OUT_DIR=./assembled \
#       bash scripts/save_model.sh
# ============================================================================

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

CONFIG_FILE="${CONFIG_FILE:-configs/local/config.yaml}"
RUN_ID="${RUN_ID:-}"
OUT_DIR="${OUT_DIR:-}"

[[ "${CONFIG_FILE}" != /* ]] && CONFIG_FILE="${REPO_ROOT}/${CONFIG_FILE}"

ARGS=(--config "${CONFIG_FILE}")
[[ -n "${RUN_ID}" ]] && ARGS+=(--run-id "${RUN_ID}")
[[ -n "${OUT_DIR}" ]] && ARGS+=(--out-dir "${OUT_DIR}")

echo "=========================================="
echo "GSQ save_model"
echo "Config : ${CONFIG_FILE}"
echo "Run ID : ${RUN_ID:-<latest>}"
echo "Out dir: ${OUT_DIR:-<default>}"
echo "=========================================="

cd "${REPO_ROOT}"
exec python "${REPO_ROOT}/save_model.py" "${ARGS[@]}" "$@"
