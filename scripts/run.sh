#!/usr/bin/env bash
# ============================================================================
# GSQ — production training run (bare metal)
# ============================================================================
# Single-node multi-GPU by default (via torchrun). For manual multi-node, export
# WORLD_SIZE / RANK / LOCAL_RANK / MASTER_ADDR / MASTER_PORT before invoking and
# the script will skip torchrun and run main.py directly.
#
# Usage:
#   bash scripts/run.sh
#   CONFIG_FILE=configs/kimi-k2.5/kimi_k2.5_2bit_gptq_gsq.yaml bash scripts/run.sh
#   NPROC=4 RESUME=latest bash scripts/run.sh
#   SMOKE_TEST=1 bash scripts/run.sh
#
# Multi-node (run on each node, with shared storage and matching configs):
#   WORLD_SIZE=8 RANK=0 LOCAL_RANK=0 MASTER_ADDR=node0 MASTER_PORT=29500 \
#       bash scripts/run.sh
# ============================================================================

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

CONFIG_FILE="${CONFIG_FILE:-configs/local/config.yaml}"
RESUME="${RESUME:-}"
SMOKE_TEST="${SMOKE_TEST:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"

[[ "${CONFIG_FILE}" != /* ]] && CONFIG_FILE="${REPO_ROOT}/${CONFIG_FILE}"

if [[ "${SMOKE_TEST}" = "1" ]]; then
    SMOKE_DIR="${GSQ_RUNTIME}/smoke/$$"
    mkdir -p "${SMOKE_DIR}"
    SMOKE_TEMPLATE="${REPO_ROOT}/configs/config_smoke.yaml"
    if [[ -f "${SMOKE_TEMPLATE}" ]]; then
        SMOKE_CONFIG="${SMOKE_DIR}/config_smoke.yaml"
        SLURM_JOB_ID="${SLURM_JOB_ID:-$$}" envsubst '${GSQ_RUNTIME} ${SLURM_JOB_ID}' \
            < "${SMOKE_TEMPLATE}" > "${SMOKE_CONFIG}"
        CONFIG_FILE="${SMOKE_CONFIG}"
    fi
fi

EXTRA_ARGS=()
[[ -n "${RESUME}" ]] && EXTRA_ARGS+=(--resume "${RESUME}")
[[ "${SMOKE_TEST}" = "1" ]] && EXTRA_ARGS+=(--max-layers 2)

NPROC="${NPROC:-$(detect_num_gpus)}"
[[ "${NPROC}" -lt 1 ]] && NPROC=1

echo "=========================================="
echo "GSQ run"
echo "Start time : $(date)"
echo "Host       : $(hostname)"
echo "Config     : ${CONFIG_FILE}"
echo "Resume     : ${RESUME:-<fresh>}"
echo "Smoke test : ${SMOKE_TEST}"
echo "=========================================="

cd "${REPO_ROOT}"

# Manual multi-node escape hatch: caller has set up rank/master env vars.
if [[ -n "${WORLD_SIZE:-}" && "${WORLD_SIZE}" != "1" && -n "${MASTER_ADDR:-}" ]]; then
    echo "Multi-node mode: WORLD_SIZE=${WORLD_SIZE} RANK=${RANK:-?} LOCAL_RANK=${LOCAL_RANK:-?}"
    echo "                 MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
    export MASTER_PORT
    exec python "${REPO_ROOT}/main.py" \
        --config "${CONFIG_FILE}" "${EXTRA_ARGS[@]}" "$@"
fi

echo "Single-node mode: torchrun --standalone --nproc-per-node=${NPROC}"
exec torchrun \
    --standalone \
    --nproc-per-node="${NPROC}" \
    "${REPO_ROOT}/main.py" \
    --config "${CONFIG_FILE}" \
    "${EXTRA_ARGS[@]}" "$@"
