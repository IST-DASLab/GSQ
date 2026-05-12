#!/usr/bin/env bash
# ============================================================================
# GSQ — bare-metal environment setup
# ============================================================================
# Creates a host Python venv at ${VENV_PATH:-./.venv}, installs GSQ in editable
# mode together with PyTorch, vLLM, lm-eval, and (optionally) flash-attn.
#
# Usage:
#   bash scripts/setup_env.sh                # default: install everything
#   SKIP_FLASH_ATTN=1 bash scripts/setup_env.sh
#   VENV_PATH=/path/to/venv bash scripts/setup_env.sh
#   PYTHON=python3.11 bash scripts/setup_env.sh
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
PYTHON="${PYTHON:-python3}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

echo "=========================================="
echo "GSQ — host venv setup"
echo "Repo root  : ${REPO_ROOT}"
echo "Venv path  : ${VENV_PATH}"
echo "Python     : $(${PYTHON} --version)"
echo "Torch idx  : ${TORCH_INDEX_URL}"
echo "flash-attn : $([ "${SKIP_FLASH_ATTN}" = "1" ] && echo SKIP || echo INSTALL)"
echo "=========================================="

if [[ ! -d "${VENV_PATH}" ]]; then
    echo "[1/5] Creating venv at ${VENV_PATH}..."
    "${PYTHON}" -m venv "${VENV_PATH}"
else
    echo "[1/5] Reusing existing venv at ${VENV_PATH}"
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
echo "      Active python: $(which python) — $(python --version)"

echo "[2/5] Upgrading pip / setuptools / wheel..."
pip install --upgrade pip setuptools wheel --quiet

echo "[3/5] Installing PyTorch (${TORCH_INDEX_URL})..."
if ! python -c "import torch" 2>/dev/null; then
    pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
else
    echo "      torch already installed: $(python -c 'import torch; print(torch.__version__)')"
fi

echo "[4/5] Installing GSQ (editable) + vLLM + lm-eval..."
pip install -e "${REPO_ROOT}[torch]"

if [[ "${SKIP_FLASH_ATTN}" != "1" ]]; then
    echo "      Installing flash-attn (this can take a while; set SKIP_FLASH_ATTN=1 to skip)..."
    export MAX_JOBS="${MAX_JOBS:-8}"
    pip install flash-attn --no-cache-dir --no-build-isolation || {
        echo "      WARNING: flash-attn install failed. Continuing without it."
    }
fi

echo "[5/5] Verifying installed packages..."
echo "=========================================="
python - <<'PYEOF'
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
echo "Activate the venv with:"
echo "  source ${VENV_PATH}/bin/activate"
echo "=========================================="
