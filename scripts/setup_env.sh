#!/usr/bin/env bash
#
# One-time setup for GSQ in the Slurm/container environment.
#
# Two modes of operation:
#
#   1. --build-image [--image NAME]  : Pull the container, import as squashfs.
#                       Run FIRST on a compute node (no --environment). Optional
#                       --image: pytorch (default) | cuda
#
#      srun ... bash scripts/setup_env.sh --build-image
#      srun ... bash scripts/setup_env.sh --build-image --image cuda
#
#   2. (default)      : Create the Python venv inside the container and install
#                       GSQ. Use --environment=./scripts/gsq.toml (PyTorch) or
#                       --environment=./scripts/gsq-cuda.toml (CUDA base).
#                       For gsq-cuda.toml, use --mem=200G or more (flash-attn build is RAM-heavy).
#
#      srun ... --environment=./scripts/gsq.toml bash scripts/setup_env.sh
#      srun ... --mem=120G ... --environment=./scripts/gsq-cuda.toml bash scripts/setup_env.sh
#
# After both steps complete, the venv is ready and all sbatch scripts will
# activate it automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
SCRATCH="${SCRATCH:-"${PROJECT_ROOT}/runtime"}"

IMAGE_DIR="${IMAGE_DIR:-"${HOME}/.gsq/ce-images"}"

# Parse --build-image and optional --image (pytorch | cuda)
BUILD_IMAGE_TYPE="pytorch"
WAS_BUILD_IMAGE=0
if [ "${1:-}" = "--build-image" ]; then
    WAS_BUILD_IMAGE=1
    shift
    if [ "${1:-}" = "--image" ]; then
        shift
        BUILD_IMAGE_TYPE="${1:-pytorch}"
        shift
    fi
fi

case "${BUILD_IMAGE_TYPE}" in
    pytorch)
        SOURCE_IMAGE="nvcr.io#nvidia/pytorch:26.02-py3"
        IMAGE_NAME="ngc-pytorch-26.02.sqsh"
        EDF_FILE="gsq.toml"
        ;;
    cuda)
        SOURCE_IMAGE="docker.io#nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04"
        IMAGE_NAME="cuda-12.8.1-cudnn-devel-ubuntu24.04.sqsh"
        EDF_FILE="gsq-cuda.toml"
        ;;
    *)
        echo "Unknown --image: ${BUILD_IMAGE_TYPE}. Use pytorch or cuda."
        exit 1
        ;;
esac

IMAGE_PATH="${IMAGE_DIR}/${IMAGE_NAME}"

# ============================================================================
# Mode: --build-image
# ============================================================================
if [ "${WAS_BUILD_IMAGE}" = "1" ]; then
    echo "=========================================="
    echo "GSQ — Build Container Image (squashfs)"
    echo "=========================================="
    echo "Image type    : ${BUILD_IMAGE_TYPE}"
    echo "Source image  : ${SOURCE_IMAGE}"
    echo "Output        : ${IMAGE_PATH}"
    echo "=========================================="

    STORAGE_CONF="${XDG_CONFIG_HOME:-${HOME}/.config}/containers/storage.conf"
    if [ ! -f "${STORAGE_CONF}" ]; then
        echo "Creating podman storage config at ${STORAGE_CONF}..."
        mkdir -p "$(dirname "${STORAGE_CONF}")"
        cat > "${STORAGE_CONF}" <<'CONF'
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"
CONF
    fi

    # Set up Lustre striping on the image directory (CSCS recommendation)
    mkdir -p "${IMAGE_DIR}"
    lfs setstripe -E 4M -c 1 -E 64M -c 4 -E -1 -c -1 -S 4M "${IMAGE_DIR}" 2>/dev/null || true

    if [ "${BUILD_IMAGE_TYPE}" = "cuda" ]; then
        # CUDA base image has no Python; build from Dockerfile.cuda which adds Python.
        DOCKERFILE="${SCRIPT_DIR}/Dockerfile.cuda"
        if [ ! -f "${DOCKERFILE}" ]; then
            echo "ERROR: ${DOCKERFILE} not found. Cannot build CUDA image without it."
            exit 1
        fi
        echo ""
        echo ">>> Building image from ${DOCKERFILE} (adds Python to nvidia/cuda)..."
        podman build -f "${DOCKERFILE}" -t gsq-cuda:local "${PROJECT_ROOT}"
        echo ""
        echo ">>> Importing as squashfs: ${IMAGE_PATH}"
        enroot import -x mount -o "${IMAGE_PATH}" "podman://gsq-cuda:local"
    else
        echo ""
        echo ">>> Pulling image with podman..."
        podman pull "${SOURCE_IMAGE/\#/\/}"
        PODMAN_TAG="${SOURCE_IMAGE#*#}"
        echo ""
        echo ">>> Importing as squashfs: ${IMAGE_PATH}"
        enroot import -x mount -o "${IMAGE_PATH}" "podman://${PODMAN_TAG}"
    fi

    echo ""
    echo "=========================================="
    echo "Image built successfully!"
    echo "  ${IMAGE_PATH}"
    echo "  Size: $(du -h "${IMAGE_PATH}" | cut -f1)"
    echo ""
    echo "Next step — create the venv (inside the container):"
    echo "  srun --mpi=pmix --account=<ACCOUNT> --partition=normal --nodes=1 \\"
    echo "       --ntasks-per-node=1 --gpus-per-task=1 --time=01:00:00 \\"
    echo "       --environment=./scripts/${EDF_FILE} \\"
    echo "       bash scripts/setup_env.sh"
    echo "=========================================="
    exit 0
fi



# ============================================================================
# Mode: default — create venv and install dependencies
# ============================================================================
if python -c "import torch" 2>/dev/null; then
    VENV_NAME="venv-gsq"
    VENV_EXTRA="--system-site-packages"
    echo "Detected system PyTorch — venv: ${VENV_NAME} (--system-site-packages)."
else
    VENV_NAME="venv-gsq-cuda"
    VENV_EXTRA=""
    echo "No system PyTorch — venv: ${VENV_NAME} (all deps in venv)."
fi
VENV_PATH="${SCRATCH}/gsq/${VENV_NAME}"


echo "=========================================="
echo "GSQ — container environment setup"
echo "=========================================="
echo "Project root  : ${PROJECT_ROOT}"
echo "Venv path     : ${VENV_PATH}"
echo "Python        : $(python --version 2>/dev/null || echo 'not found')"
echo "=========================================="


if [ -d "${VENV_PATH}" ]; then
    echo "[1/5] Removing existing venv..."
    rm -rf "${VENV_PATH}"
fi
if [ -n "${VENV_EXTRA}" ]; then
    echo "[1/5] Creating venv (--system-site-packages)..."
    python -m venv ${VENV_EXTRA} "${VENV_PATH}"
else
    echo "[1/5] Creating venv..."
    python -m venv "${VENV_PATH}"
fi


echo "[2/5] Activating venv..."
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
echo "      Active Python: $(which python) — $(python --version)"

echo "[3/5] Upgrading pip, installing setuptools and wheel..."
pip install --upgrade pip setuptools wheel --quiet


echo "[4/5] Installing GSQ (editable) and all dependencies ..."
if [ -n "${VENV_EXTRA}" ]; then
    pip install -e "${PROJECT_ROOT}"
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install -e "${PROJECT_ROOT}[torch]"
fi
# Limit parallel nvcc jobs to avoid OOM during flash-attn build (each job can use 10–20G+)
export MAX_JOBS=${MAX_JOBS:-8}
# Flush caches so flash-attn builds against current container CUDA (avoid stale cu13 wheel/extension)
if [ -z "${VENV_EXTRA}" ]; then
    echo "Clearing pip and torch extension caches for a clean flash-attn build..."
    pip cache purge 2>/dev/null || true
    rm -rf "${HOME}/.cache/torch_extensions" 2>/dev/null || true
    rm -rf "${SCRATCH}/.cache/torch_extensions" 2>/dev/null || true
fi
pip install flash-attn --no-cache-dir --no-build-isolation


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
check("flash_attn")
check("ray")
check("vllm")

try:
    import torch
    if torch.cuda.is_available():
        print(f"  OK  CUDA available — {torch.cuda.device_count()} GPU(s) — {torch.cuda.get_device_name(0)}")
    else:
        print("  WARN CUDA not available (expected on compute node only)")
except ImportError as e:
    print(f"  FAIL torch: {e}")
    errors.append("torch")

if errors:
    print(f"\nFailed packages: {errors}")
    sys.exit(1)
else:
    print("\nAll required packages verified.")
PYEOF

echo "=========================================="
echo "Setup complete!"
echo ""
echo "Activate in your session with:"
echo "  source ${VENV_PATH}/bin/activate"
echo ""
echo "The sbatch scripts activate the venv automatically."
echo "=========================================="
