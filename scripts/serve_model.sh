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
EVAL_LIMIT="${EVAL_LIMIT:-}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"
EVAL_WANDB_FLAG="${EVAL_WANDB_FLAG:-}"

# Resolve MODEL_PATH from RUN_ID if needed.
# Search both ${SCRATCH}/gsq/checkpoints (legacy / user-overridden SCRATCH) and
# ${REPO_ROOT}/runtime/gsq/checkpoints (config default). The user's SCRATCH may
# point outside the repo; assembled checkpoints live in the repo by config.
if [[ -z "${MODEL_PATH}" ]]; then
    if [[ -z "${RUN_ID}" ]]; then
        echo "ERROR: set MODEL_PATH or RUN_ID before running." >&2
        exit 1
    fi
    CANDIDATE_DIR=""
    for SEARCH_ROOT in "${SCRATCH}/gsq/checkpoints" "${REPO_ROOT}/runtime/gsq/checkpoints"; do
        [[ -d "${SEARCH_ROOT}" ]] || continue
        FOUND=$(find "${SEARCH_ROOT}" -type d -path "*/${RUN_ID}/assembled" -print -quit 2>/dev/null)
        if [[ -n "${FOUND}" && -d "${FOUND}" ]]; then
            CANDIDATE_DIR="${FOUND}"
            break
        fi
    done
    if [[ -z "${CANDIDATE_DIR}" ]]; then
        echo "ERROR: no assembled model found for RUN_ID=${RUN_ID}" >&2
        echo "       searched: ${SCRATCH}/gsq/checkpoints, ${REPO_ROOT}/runtime/gsq/checkpoints" >&2
        exit 1
    fi
    MODEL_PATH="${CANDIDATE_DIR}"
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
fi
[[ -z "${EVAL_OUTPUT_DIR}" ]] && EVAL_OUTPUT_DIR="${MODEL_PATH}/evals"

# vLLM's compressed-tensors WNA16 fused-MoE Marlin kernel requires
# (moe_intermediate_size / TP) % max(64, group_size) == 0 and
# group_size in {-1, 32, 64, 128} (marlin_utils.py:check_moe_marlin_supports_layer).
# If the check fails it silently falls back to a Triton WNA16 path that
# either crashes inside moe_sum during profile_run (sm_89) or races on the
# shared Triton cache on beegfs. See knowledge/05-vllm-tp-marlin-moe-shape-constraint.md
# Clamp TP_SIZE to the largest divisor of itself that satisfies the constraint.
TP_CLAMPED=$(MODEL_PATH="${MODEL_PATH}" TP_SIZE="${TP_SIZE}" python - <<'PY'
import json, os, sys
mp = os.environ["MODEL_PATH"]
tp = int(os.environ["TP_SIZE"])
try:
    with open(os.path.join(mp, "config.json")) as f:
        cfg = json.load(f)
except Exception:
    print(tp); sys.exit(0)
mis = cfg.get("moe_intermediate_size")
if not mis:
    print(tp); sys.exit(0)
gs = 128
qc = cfg.get("quantization_config") or {}
for g in (qc.get("config_groups") or {}).values():
    w = (g or {}).get("weights") or {}
    if isinstance(w.get("group_size"), int) and w["group_size"] > 0:
        gs = w["group_size"]; break
if gs not in (-1, 32, 64, 128):
    print(tp); sys.exit(0)
need = max(64, gs)
best = tp
while best > 1:
    if mis % best == 0 and (mis // best) % need == 0:
        break
    best -= 1
print(best)
PY
)
if [[ -n "${TP_CLAMPED}" && "${TP_CLAMPED}" != "${TP_SIZE}" ]]; then
    echo "[gsq-serve] WARNING: clamping TP_SIZE ${TP_SIZE} -> ${TP_CLAMPED} so that" >&2
    echo "            (moe_intermediate_size / TP) % max(64, group_size) == 0" >&2
    echo "            (vLLM Marlin WNA16 MoE shape constraint; see" >&2
    echo "            research_logs/knowledge/05-vllm-tp-marlin-moe-shape-constraint.md)" >&2
    echo "            Set TP_SIZE_FORCE=1 to skip this clamp." >&2
    if [[ "${TP_SIZE_FORCE:-0}" != "1" ]]; then
        TP_SIZE="${TP_CLAMPED}"
    fi
fi

# Resolve WANDB_RUN_ID (so eval can resume the same WandB run). Same dual-root,
# pipefail-safe lookup as MODEL_PATH above. ${SCRATCH} may legitimately point
# outside the repo (user-controlled), in which case the find on a non-existent
# path would exit non-zero and trip pipefail+set -e, silently killing the script.
if [[ -z "${WANDB_RUN_ID:-}" && -n "${RUN_ID}" ]]; then
    PROGRESS_JSON=""
    for SEARCH_ROOT in "${SCRATCH}/gsq/checkpoints" "${REPO_ROOT}/runtime/gsq/checkpoints"; do
        [[ -d "${SEARCH_ROOT}" ]] || continue
        FOUND=$(find "${SEARCH_ROOT}" -path "*/${RUN_ID}/progress.json" -print -quit 2>/dev/null)
        if [[ -n "${FOUND}" && -f "${FOUND}" ]]; then
            PROGRESS_JSON="${FOUND}"
            break
        fi
    done
    if [[ -n "${PROGRESS_JSON}" ]]; then
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

# Hopper / Ampere advisory check.
# vLLM's compressed-tensors WNA16 fused MoE has no Marlin kernel on Ada
# (sm_89, L40 / L40S / RTX 4090) and falls back to a Triton path that has
# crashed during profile_run on our setup. See
# research_logs/knowledge/04-hopper-ampere-required-for-serve.md
# Warn loudly but do NOT abort: dense / non-WNA16 cases may still work,
# and the kernel coverage is expected to improve in newer vLLM tags.
python - <<'PY' || true
import sys
try:
    import torch
except Exception as e:
    print(f"[gsq-serve] torch import failed during sm-cap probe: {e}", file=sys.stderr)
    sys.exit(0)
if not torch.cuda.is_available():
    sys.exit(0)
caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
names = {torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
bad = {c for c in caps if c[0] < 8 or c == (8, 9)}
if bad:
    print("=" * 70, file=sys.stderr)
    print("[gsq-serve] WARNING: GSQ has only been tested with vLLM on Hopper", file=sys.stderr)
    print("            (sm_90, H100). Detected GPUs with compute capabilities", file=sys.stderr)
    print(f"            {sorted(caps)} on devices {sorted(names)}.", file=sys.stderr)
    if (8, 9) in bad:
        print("            sm_89 (Ada, L40 / L40S / RTX 4090) is known-broken for", file=sys.stderr)
        print("            compressed-tensors WNA16 fused MoE in vLLM 0.20.x:", file=sys.stderr)
        print("            it falls back from Marlin to a Triton path that", file=sys.stderr)
        print("            crashes inside moe_sum during profile_run.", file=sys.stderr)
    print("            See research_logs/knowledge/04-hopper-ampere-required-for-serve.md", file=sys.stderr)
    print("            Proceeding anyway (warn-only).", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
PY

cd "${REPO_ROOT}"

# Sensible local cache locations to keep Triton/Inductor off the home filesystem.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH}/.triton_cache}"
export TRITON_HOME="${TRITON_HOME:-${SCRATCH}/.triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SCRATCH}/.inductor_cache}"
export TMPDIR="${TMPDIR:-${SCRATCH}/.tmp}"
mkdir -p "${TRITON_CACHE_DIR}" "${TRITON_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}"

# Persist vLLM stdout/stderr next to the model so failures are debuggable
# even when the terminal scrollback rolls (vLLM startup logs are huge).
SERVE_LOG="${SERVE_LOG:-${MODEL_PATH}/serve_vllm.log}"
mkdir -p "$(dirname "${SERVE_LOG}")"
echo "vLLM log   : ${SERVE_LOG}"

# Launch the vLLM server in the background so we can optionally run eval.
# Use stdbuf to keep output line-buffered into the log, and `tee` so the
# user still sees it on the terminal in real time.
( python "${REPO_ROOT}/serve_vllm.py" --num-nodes 1 --port "${PORT}" "${VLLM_ARGS[@]}" 2>&1 \
    | tee "${SERVE_LOG}" ) &
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
    if command -v curl >/dev/null 2>&1; then
        _HEALTH_CMD=(curl -sf "${HEALTH_URL}")
    elif command -v wget >/dev/null 2>&1; then
        _HEALTH_CMD=(wget -qO- "${HEALTH_URL}")
    else
        _HEALTH_CMD=(python -c "import sys, urllib.request as u; sys.exit(0 if u.urlopen('${HEALTH_URL}', timeout=5).status==200 else 1)")
    fi
    for i in $(seq 1 180); do
        if "${_HEALTH_CMD[@]}" >/dev/null 2>&1; then
            echo "  Server healthy after ${i}*20s"
            HEALTHY=1
            break
        fi
        # Bail out early if the server background process has already died
        # (otherwise we'd loop curl-poll for a full hour for no reason).
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: vLLM server pid ${SERVER_PID} exited before becoming healthy." >&2
            echo "       See ${SERVE_LOG} for details (last 40 lines):" >&2
            tail -n 40 "${SERVE_LOG}" >&2 || true
            exit 1
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
        [[ -n "${EVAL_LIMIT}" ]] && EVAL_ARGS+=(--limit "${EVAL_LIMIT}")

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
