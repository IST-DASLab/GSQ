#!/usr/bin/env bash
# ============================================================================
# GSQ — bare-metal environment setup (uv)
# ============================================================================
# Creates a project venv at ${VENV_PATH:-./.venv} via uv and installs GSQ in
# editable mode. Locked dependencies come from uv.lock; flash-attn is added on
# top (it cannot live in the lock because it requires --no-build-isolation).
#
# Usage:
#   bash scripts/setup_env.sh                       # default: full install
#   SKIP_FLASH_ATTN=1 bash scripts/setup_env.sh     # skip flash-attn build
#   PYTHON_VERSION=3.11 bash scripts/setup_env.sh   # pin Python version
#   VENV_PATH=/data/venvs/gsq bash scripts/setup_env.sh
#   TORCH_CUDA=cu128 bash scripts/setup_env.sh      # override torch wheel index
#                                                   #   (default: cu130 from pyproject.toml,
#                                                   #    matches vLLM 0.20+ which ships CUDA-13 ABI)
#
# Requires uv. Install with:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"
TORCH_CUDA="${TORCH_CUDA:-}"
if [[ -n "${TORCH_CUDA}" ]]; then
    export UV_INDEX_PYTORCH="https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed. Install with:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

echo "=========================================="
echo "GSQ — uv environment setup"
echo "Repo root  : ${REPO_ROOT}"
echo "Venv path  : ${VENV_PATH}"
echo "Python     : ${PYTHON_VERSION}"
echo "uv         : $(uv --version)"
echo "torch idx  : ${UV_INDEX_PYTORCH:-<from pyproject.toml: cu130>}"
echo "flash-attn : $([ "${SKIP_FLASH_ATTN}" = "1" ] && echo SKIP || echo INSTALL)"
echo "=========================================="

cd "${REPO_ROOT}"
export UV_PROJECT_ENVIRONMENT="${VENV_PATH}"

echo "[1/3] Syncing project dependencies (uv sync)..."
uv sync --python "${PYTHON_VERSION}"

if [[ "${SKIP_FLASH_ATTN}" != "1" ]]; then
    echo "[2/3] Installing flash-attn (no-build-isolation)..."
    export MAX_JOBS="${MAX_JOBS:-8}"
    uv pip install flash-attn --no-cache-dir --no-build-isolation || {
        echo "      WARNING: flash-attn install failed. Continuing without it." >&2
    }
else
    echo "[2/3] Skipping flash-attn (SKIP_FLASH_ATTN=1)"
fi

echo "[3/3] Verifying installed packages..."
echo "=========================================="
uv run --no-sync python - <<'PYEOF'
import sys
errors = []

def check(pkg, label=None):
    label = label or pkg
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        loc = getattr(mod, '__file__', 'unknown')
        print(f"  OK  {label} == {ver}  ({loc})")
    except ImportError as e:
        print(f"  FAIL {label}: {e}")
        errors.append(label)

check("torch")
check("transformers")
check("accelerate")
check("datasets")
check("safetensors")
check("compressed_tensors")
check("lion_pytorch")
check("wandb")
check("tqdm")
check("yaml", "pyyaml")
check("dotenv", "python-dotenv")

for optional in ("flash_attn", "ray", "vllm", "lm_eval"):
    try:
        mod = __import__(optional)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"  OK  {optional} == {ver}")
    except ImportError:
        print(f"  --  {optional} not installed (optional)")

try:
    import torch
    if torch.cuda.is_available():
        print(f"  OK  CUDA available — {torch.cuda.device_count()} GPU(s) — {torch.cuda.get_device_name(0)}")
    else:
        print("  WARN CUDA not available")
except ImportError as e:
    print(f"  FAIL torch: {e}")
    errors.append("torch")

if errors:
    print(f"\nFailed required packages: {errors}")
    sys.exit(1)
print("\nAll required packages verified.")
PYEOF

echo "=========================================="
echo "Setup complete."
echo ""
echo "Activate the venv with:"
echo "  source ${VENV_PATH}/bin/activate"
echo ""
echo "Or run any command via uv (no activation needed):"
echo "  uv run python main.py --config configs/local/config.yaml"
echo "=========================================="
