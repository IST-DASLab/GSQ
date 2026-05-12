# Shared bootstrap for GSQ bare-metal entry scripts.
# Source from any scripts/*.sh after `set -euo pipefail`.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH:-${REPO_ROOT}/runtime}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"

if [[ -d "${VENV_PATH}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
else
    echo "WARNING: venv not found at ${VENV_PATH} — falling back to system python." >&2
    echo "         Run 'bash scripts/setup_env.sh' to create one." >&2
fi

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "${REPO_ROOT}/.env"
    set +a
fi

ulimit -c 0

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

mkdir -p "${REPO_ROOT}/logs"

detect_num_gpus() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l
    else
        echo 0
    fi
}
