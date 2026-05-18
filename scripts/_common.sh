# Shared bootstrap for GSQ bare-metal entry scripts.
# Source from any scripts/*.sh after `set -euo pipefail`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "${REPO_ROOT}/.env"
    set +a
fi

# GSQ artifact root only — do not use machine-wide SCRATCH here (often set on HPC to a quota path).
GSQ_RUNTIME="${GSQ_RUNTIME:-${REPO_ROOT}/runtime}"
export GSQ_RUNTIME
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"

# HuggingFace hub/datasets root. Override via `.env` or the shell when using a cluster cache.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
mkdir -p "${HF_HOME}/hub" "${HF_HOME}/datasets" "${HF_HOME}/xet" 2>/dev/null || true
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

if [[ -d "${VENV_PATH}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
else
    echo "WARNING: venv not found at ${VENV_PATH} — falling back to system python." >&2
    echo "         Run 'bash scripts/setup_env.sh' to create one." >&2
fi

ulimit -c 0

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# If a multi-GPU run hangs at the very end in destroy_process_group(), main.py bounds that
# call with GSQ_DIST_DESTROY_TIMEOUT_SEC (default 120 in code). Use 0 to wait indefinitely,
# or GSQ_SKIP_DIST_DESTROY=1 to skip destroy() entirely (exit may be noisier but returns).

mkdir -p "${REPO_ROOT}/logs"

detect_num_gpus() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l
    else
        echo 0
    fi
}
