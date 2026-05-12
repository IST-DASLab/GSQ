#!/usr/bin/env bash
# ============================================================================
# Smoke test: serve a small model (LLaMA-3.2-1B) with vLLM on a single GPU.
# Validates the CUDA/cuBLAS/PyTorch/vLLM stack independently of Kimi-K2.5.
#
# Usage: sbatch scripts/smoke_serve.sbatch.sh
# ============================================================================

#SBATCH --job-name=gsq-smoke
#SBATCH --account=a-g200
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --no-requeue
#SBATCH --output=logs/slurm/gsq_smoke_%j.out
#SBATCH --error=logs/slurm/gsq_smoke_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRATCH="${SCRATCH:-"${SCRIPT_ROOT}/runtime"}"
EDF_FILE="${SCRIPT_ROOT}/scripts/gsq-cuda.toml"
VENV_PATH="${SCRATCH}/gsq/venv-gsq-cuda"
MODEL="${SCRATCH}/gsq/models/Llama-3.2-1B"
PORT=8000

if [ -f "${SCRIPT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${SCRIPT_ROOT}/.env" | xargs)
fi

mkdir -p "${SCRIPT_ROOT}/logs/slurm"

echo "=========================================="
echo "GSQ vLLM Smoke Test"
echo "Start time : $(date)"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "Model      : ${MODEL}"
echo "=========================================="

srun --mpi=pmix --environment="${EDF_FILE}" bash -c '
set -euo pipefail
source '"${VENV_PATH}"'/bin/activate

echo "Python : $(which python)"
echo "vLLM   : $(python -c "import vllm; print(vllm.__version__)")"
echo "Torch  : $(python -c "import torch; print(torch.__version__)")"
echo "CUDA   : $(python -c "import torch; print(torch.version.cuda)")"
echo "Device : $(python -c "import torch; print(torch.cuda.get_device_name(0))")"
echo "---"

python -c "
import torch
a = torch.randn(32, 128, device=\"cuda\", dtype=torch.bfloat16)
b = torch.randn(128, 64, device=\"cuda\", dtype=torch.bfloat16)
c = a @ b
print(f\"cuBLAS bf16 matmul: {a.shape} x {b.shape} -> {c.shape} OK\")
"

echo "Starting vLLM serve on port '"${PORT}"'..."
vllm serve '"${MODEL}"' \
    --host 0.0.0.0 \
    --port '"${PORT}"' \
    --max-model-len 2048 \
    --enforce-eager \
    --trust-remote-code &
VLLM_PID=$!

HEALTHY=0
for i in $(seq 1 120); do
    if python -c "import urllib.request; urllib.request.urlopen(\"http://localhost:'"${PORT}"'/health\")" 2>/dev/null; then
        echo "Server is up after ${i}s"
        HEALTHY=1
        break
    fi
    sleep 2
done

if [ "${HEALTHY}" -eq 0 ]; then
    echo "FAIL: server did not become healthy in 240s"
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

python -c "
import json, urllib.request
req = urllib.request.Request(
    \"http://localhost:'"${PORT}"'/v1/completions\",
    data=json.dumps({
        \"model\": \"'"${MODEL}"'\",
        \"prompt\": \"The capital of France is\",
        \"max_tokens\": 20
    }).encode(),
    headers={\"Content-Type\": \"application/json\"}
)
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
print(\"Completions test:\", result[\"choices\"][0][\"text\"])
"

echo "=========================================="
echo "SMOKE TEST PASSED"
echo "=========================================="

kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true
'
