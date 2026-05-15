#!/usr/bin/env bash
# ============================================================================
# GSQ — smoke end-to-end: run (--max-layers 2) -> save_model -> serve + lm-eval
# ============================================================================
#
# Prerequisites (typical gpu274 layout):
#   export HF_HOME=/nfs/scistore19/alistgrp/huggingface   # hub model cache
#   export HF_DATASETS_CACHE=.../.cache/huggingface/datasets # writable C4 cache
#
# Override any time:
#   NPROC=8 CONFIG_SLUGS="llama kimi" KEEP_SERVING=0 bash scripts/run_e2e_verify.sh
#
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

export HF_HOME="${HF_HOME:-/nfs/scistore19/alistgrp/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export SCRATCH="${SCRATCH:-${REPO_ROOT}/runtime}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}"
export NPROC="${NPROC:-8}"

KEEP_SERVING="${KEEP_SERVING:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-50}"

if [[ "${NPROC}" -lt 1 ]]; then NPROC=1; fi

CONFIG_SLUGS="${CONFIG_SLUGS:-llama qwen3 qwen35 kimi}"

declare -A _VERIFY_YAML
_VERIFY_YAML[llama]="configs/local/verify_llama32_1b.yaml"
_VERIFY_YAML[qwen3]="configs/local/verify_qwen3_30b.yaml"
_VERIFY_YAML[qwen35]="configs/local/verify_qwen35_35b.yaml"
_VERIFY_YAML[kimi]="configs/local/verify_kimi_k25.yaml"

declare -A _CKPT_PARENT
_CKPT_PARENT[llama]="verify-llama32-1b"
_CKPT_PARENT[qwen3]="verify-qwen3-30b"
_CKPT_PARENT[qwen35]="verify-qwen35-35b"
_CKPT_PARENT[kimi]="verify-kimi-k25"

_pick_latest_run_id() {
    local parent="${_CKPT_PARENT[$1]:?}"
    local ckpt="${REPO_ROOT}/runtime/gsq/checkpoints/${parent}"
    local latest=
    latest="$(find "${ckpt}" -mindepth 2 -maxdepth 2 -name progress.json -printf '%T@\t%p\n' 2>/dev/null \
        | sort -nr | head -n1 | cut -f2-)"
    if [[ -n "${latest}" ]]; then
        basename "$(dirname "${latest}")"
    fi
}

for slug in ${CONFIG_SLUGS}; do
    cfg="${_VERIFY_YAML[$slug]-}"
    if [[ -z "${cfg}" ]]; then
        echo "Unknown CONFIG_SLUGS entry: ${slug}" >&2
        exit 1
    fi
    [[ "${cfg}" != /* ]] && cfg="${REPO_ROOT}/${cfg}"
    echo "=========================================="
    echo "E2E verify: ${slug}  (${cfg})"
    echo "=========================================="
    cd "${REPO_ROOT}"
    CONFIG_FILE="${cfg}" bash scripts/run.sh --max-layers 2

    RUN_ID="$(_pick_latest_run_id "${slug}")"
    if [[ -z "${RUN_ID}" ]]; then
        echo "ERROR: no progress.json found under ${_CKPT_PARENT[$slug]}" >&2
        exit 1
    fi
    echo "Run ID : ${RUN_ID}"

    RUN_ID="${RUN_ID}" CONFIG_FILE="${cfg}" bash scripts/save_model.sh

    unset EXTRA_VLLM_ARGS
    if [[ "${slug}" != "kimi" ]]; then
        export EXTRA_VLLM_ARGS="--gpu-memory-utilization 0.85 --tokenizer-mode hf --max-num-seqs 4 --enforce-eager"
    fi

    CFG_REL="${cfg#"$REPO_ROOT/"}"
    export TP_SIZE="${TP_SIZE:-${NPROC}}"
    # Pass MODEL_PATH explicitly so serve_model.sh doesn't rely on ${SCRATCH}/gsq/checkpoints
    # (the user's SCRATCH may point outside the repo; assembled checkpoints live under
    # ${REPO_ROOT}/runtime/gsq/checkpoints by config).
    ASSEMBLED_PATH="${REPO_ROOT}/runtime/gsq/checkpoints/${_CKPT_PARENT[$slug]}/${RUN_ID}/assembled"
    if [[ ! -d "${ASSEMBLED_PATH}" ]]; then
        echo "ERROR: assembled model dir not found at ${ASSEMBLED_PATH}" >&2
        exit 1
    fi
    EVAL=1 \
        RUN_ID="${RUN_ID}" \
        MODEL_PATH="${ASSEMBLED_PATH}" \
        EVAL_CONFIG_FILE="${CFG_REL}" \
        EVAL_TASKS="${EVAL_TASKS:-arc_easy}" \
        EVAL_NUM_CONCURRENT="${EVAL_NUM_CONCURRENT:-4}" \
        EVAL_LIMIT="${EVAL_LIMIT}" \
        EVAL_WANDB_FLAG="${EVAL_WANDB_FLAG:---no-wandb}" \
        KEEP_SERVING="${KEEP_SERVING}" \
        bash scripts/serve_model.sh \
        || exit 1

    echo "=========================================="
    echo "DONE ${slug}"
    echo "=========================================="
done

echo "All requested slugs completed ok."
