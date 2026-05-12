#!/usr/bin/env bash
# ============================================================================
# GSQ — pre-download HuggingFace models and datasets
# ============================================================================
# Downloads target models as flat local directories under ${MODELS_DIR} and
# pre-fetches calibration datasets into ${HF_DATASETS_CACHE}.
#
# Usage:
#   bash scripts/download.sh                           # full set
#   SKIP_LARGE_MODELS=1 bash scripts/download.sh       # debug-only assets
#   MODELS_DIR=/data/models bash scripts/download.sh
# ============================================================================

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

SKIP_LARGE_MODELS="${SKIP_LARGE_MODELS:-0}"
MODELS_DIR="${MODELS_DIR:-${SCRATCH}/gsq/models}"
export HF_HOME="${HF_HOME:-${SCRATCH}/.hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Gated models (Kimi-K2.5, LLaMA) require it." >&2
    echo "       Add HF_TOKEN=hf_... to .env in the repo root." >&2
    exit 1
fi

mkdir -p "${MODELS_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

echo "=========================================="
echo "GSQ download"
echo "Models dir   : ${MODELS_DIR}"
echo "HF_HOME      : ${HF_HOME}"
echo "Skip large   : ${SKIP_LARGE_MODELS}"
echo "=========================================="

download_model() {
    local repo_id="$1"
    local name
    name="$(basename "${repo_id}")"
    local dest="${MODELS_DIR}/${name}"
    echo ""
    echo ">>> Downloading model: ${repo_id} -> ${dest}"
    hf download "${repo_id}" --local-dir "${dest}"
    echo "    OK: ${name}"
}

echo ""
echo "=========================================="
echo "Downloading models"
echo "=========================================="

download_model 'meta-llama/Llama-3.2-1B'

if [[ "${SKIP_LARGE_MODELS}" != "1" ]]; then
    download_model 'moonshotai/Kimi-K2.5'
    download_model 'moonshotai/Kimi-K2-Thinking'
    download_model 'moonshotai/Kimi-K2-Instruct'
    download_model 'Qwen/Qwen3-235B-A22B'
    download_model 'Qwen/Qwen3-235B-A22B-Thinking-2507'
    download_model 'Qwen/Qwen3-235B-A22B-Instruct-2507'
    download_model 'Qwen/Qwen3-30B-A3B-Thinking-2507'
    download_model 'Qwen/Qwen3-30B-A3B-Instruct-2507'
    download_model 'Qwen/Qwen3.5-397B-A17B'
    download_model 'Qwen/Qwen3.5-122B-A10B'
    download_model 'Qwen/Qwen3.5-35B-A3B'
    download_model 'Qwen/Qwen3.5-35B-A3B-FP8'
fi

echo ""
echo "=========================================="
echo "Downloading datasets"
echo "=========================================="

python - <<'PYEOF'
from datasets import load_dataset

print('\n>>> Downloading dataset: allenai/c4 (en, train shard 0 + val shard 0)', flush=True)
load_dataset(
    'allenai/c4',
    data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
    split='train',
)
load_dataset(
    'allenai/c4',
    data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
    split='validation',
)
print('    OK: allenai/c4', flush=True)

print('\n>>> Downloading dataset: open-thoughts/OpenThoughts-114k', flush=True)
load_dataset('open-thoughts/OpenThoughts-114k', split='train')
print('    OK: open-thoughts/OpenThoughts-114k', flush=True)
PYEOF

echo ""
echo "=========================================="
echo "All downloads complete."
echo "Models dir : ${MODELS_DIR}"
echo "=========================================="
