#!/usr/bin/env bash
# ============================================================================
# GSQ — Model and dataset pre-download (Slurm)
# ============================================================================
# Purpose : Download all HuggingFace models and datasets needed by GSQ to
#           fast NVMe scratch before submitting a production run.
#
# Usage   : sbatch scripts/download.sbatch.sh
#
# Models are downloaded as flat local directories (no HF cache symlinks)
# using `huggingface-cli download --local-dir`. This gives us real files
# we can edit (e.g. to fix bugs in remote model code).
#
# What it downloads:
#   Models  : moonshotai/Kimi-K2.5 (production)
#              moonshotai/Kimi-K2-Thinking (production)
#              moonshotai/Kimi-K2-Instruct (production)
#              Qwen/Qwen3-235B-A22B (production, MoE, 128 experts)
#              Qwen/Qwen3-235B-A22B-Thinking-2507 (production)
#              Qwen/Qwen3-235B-A22B-Instruct-2507 (production)
#              Qwen/Qwen3-30B-A3B-Thinking-2507 (production)
#              Qwen/Qwen3-30B-A3B-Instruct-2507 (production)
#              Qwen/Qwen3.5-397B-A17B (production, MoE, 512 experts, hybrid attn)
#              Qwen/Qwen3.5-122B-A10B (production)
#              Qwen/Qwen3.5-35B-A3B (production)
#              Qwen/Qwen3.5-35B-A3B-FP8 (production)
#              meta-llama/Llama-3.2-1B (debug / smoke test)
#   Datasets: open-thoughts/OpenThoughts-114k (default train set)
#             allenai/c4 (debug / smoke test train set)
#             HuggingFaceFW/fineweb-edu (optional, sample-10BT split)
#
# Notes:
#   - Kimi-K2.5 is a ~260 GB model. Download can take 30–60 min.
#   - Runs on a CPU-only node (no GPU needed); adjust partition/time as needed.
#   - HF_TOKEN must be set in .env or the environment for gated models.
# ============================================================================

#SBATCH --job-name=gsq-download
#SBATCH --account=a-g200          # <-- replace with your project account
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/gsq_download_%j.out
#SBATCH --error=logs/slurm/gsq_download_%j.err

set -euo pipefail

# ============================================================================
# User configuration
# ============================================================================

# Set to "1" to skip large production models and only download debug assets.
SKIP_LARGE_MODELS="${SKIP_LARGE_MODELS:-0}"

# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EDF_FILE="${SCRIPT_ROOT}/scripts/gsq-cuda.toml"
SCRATCH="${SCRATCH:-"${SCRIPT_ROOT}/runtime"}"
if [[ "${EDF_FILE}" == *"gsq-cuda"* ]]; then
    VENV_PATH="${SCRATCH}/gsq/venv-gsq-cuda"
else
    VENV_PATH="${SCRATCH}/gsq/venv-gsq"
fi

MODELS_DIR="${SCRATCH}/gsq/models"
HF_HOME="${SCRATCH}/.hf"
HF_DATASETS_CACHE="${HF_HOME}/datasets"

# ── Load secrets (.env contains HF_TOKEN and WANDB_API_KEY) ─────────────────
if [ -f "${SCRIPT_ROOT}/.env" ]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' "${SCRIPT_ROOT}/.env" | xargs)
fi

# ── Validate required variables ───────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Gated models (Kimi-K2.5, LLaMA) require it."
    exit 1
fi

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p "${SCRIPT_ROOT}/logs/slurm" || true
mkdir -p "${MODELS_DIR}" || true
mkdir -p "${HF_HOME}" || true
mkdir -p "${HF_DATASETS_CACHE}" || true

echo "=========================================="
echo "GSQ Download Job"
echo "Start time   : $(date)"
echo "Host         : $(hostname)"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Models dir   : ${MODELS_DIR}"
echo "HF_HOME      : ${HF_HOME}"
echo "Large models : $([ "${SKIP_LARGE_MODELS}" = "1" ] && echo SKIP || echo YES)"
echo "=========================================="

srun --environment="${EDF_FILE}" --mpi=pmix bash -c "
set -euo pipefail

source ${VENV_PATH}/bin/activate

export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
export HF_TOKEN=${HF_TOKEN}
# export HF_HUB_ENABLE_HFS3=1

unset SSL_CERT_FILE
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
else
    export SSL_CERT_FILE=\$(python -c 'import certifi; print(certifi.where())')
fi

MODELS_DIR=${MODELS_DIR}
SKIP_LARGE_MODELS=${SKIP_LARGE_MODELS}

# ── Helper function ──────────────────────────────────────────────────────
download_model() {
    local repo_id=\"\$1\"
    local name=\"\$(basename \"\$repo_id\")\"
    local dest=\"\${MODELS_DIR}/\${name}\"

    echo \"\"
    echo \">>> Downloading model: \${repo_id} -> \${dest}\"

    hf download \"\${repo_id}\" \
        --local-dir \"\${dest}\" 

    echo \"    OK: \${name}\"
}

# ── Models ───────────────────────────────────────────────────────────────
echo ''
echo '=========================================='
echo 'Downloading models'
echo '=========================================='

download_model 'meta-llama/Llama-3.2-1B'

if [ \"\${SKIP_LARGE_MODELS}\" != \"1\" ]; then
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

# ── Datasets ─────────────────────────────────────────────────────────────
echo ''
echo '=========================================='
echo 'Downloading datasets'
echo '=========================================='

python - <<'PYEOF'
import os
from datasets import load_dataset

print('\\n>>> Downloading dataset: allenai/c4 (en, train shard 0 + val shard 0)', flush=True)
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

print('\\n>>> Downloading dataset: open-thoughts/OpenThoughts-114k', flush=True)
load_dataset('open-thoughts/OpenThoughts-114k', split='train')
print('    OK: open-thoughts/OpenThoughts-114k', flush=True)

print('\\n==========================================')
print('All downloads complete.')
print('==========================================')
PYEOF
"

echo "=========================================="
echo "End time: $(date)"
echo "Download job complete."
echo ""
echo "Models directory : ${MODELS_DIR}"
echo "To list models  : ls ${MODELS_DIR}/"
echo ""
echo "To verify : ls -lh ${MODELS_DIR}/"
echo "=========================================="
