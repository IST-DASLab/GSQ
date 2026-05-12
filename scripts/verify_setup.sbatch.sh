#!/usr/bin/env bash
# ============================================================================
# GSQ — Environment setup verification (Slurm)
# ============================================================================
# Purpose : Run a minimal distributed PyTorch smoke test to confirm that the
#           container, venv, CUDA, NCCL, and Slingshot interconnect are all
#           working correctly — before attempting any real GSQ run.
#
# Usage   : sbatch scripts/verify_setup.sbatch.sh
#
# Runs on 2 nodes x 4 GPUs = 8 ranks (1 srun task/node, torchrun fans out). Each rank:
#   1. Prints its identity (rank, local_rank, node, GPU name, memory)
#   2. Allocates a tensor on GPU and performs an all-reduce via NCCL
#   3. Verifies the all-reduce result is numerically correct
#   4. Runs a small matmul and checks bf16 is supported
#   5. Prints NCCL / aws-ofi-nccl version strings
#
# Expected output: "ALL CHECKS PASSED on rank 0/7" (and similar per rank)
# ============================================================================

#SBATCH --job-name=gsq-verify
#SBATCH --account=a-g200          # <-- replace with your project account
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/gsq_verify_%j.out
#SBATCH --error=logs/slurm/gsq_verify_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EDF_FILE="${SCRIPT_ROOT}/scripts/gsq-cuda.toml"
SCRATCH="${SCRATCH:-"${SCRIPT_ROOT}/runtime"}"
if [[ "${EDF_FILE}" == *"gsq-cuda"* ]]; then
    VENV_PATH="${SCRATCH}/gsq/venv-gsq-cuda"
else
    VENV_PATH="${SCRATCH}/gsq/venv-gsq"
fi

mkdir -p "${SCRIPT_ROOT}/logs/slurm"

echo "=========================================="
echo "GSQ Setup Verification"
echo "Start time : $(date)"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Nodes      : ${SLURM_NNODES} x 4 GPUs (8 total ranks)"
echo "=========================================="

ulimit -c 0

VERIFY_SCRIPT="${SCRIPT_ROOT}/scripts/verify_smoke_test.py"

srun -ul --environment="${EDF_FILE}" --mpi=pmix --label bash -c "
set -euo pipefail

source ${VENV_PATH}/bin/activate

# Verbose NCCL logging for verification (confirms aws-ofi-nccl is loaded)
export NCCL_DEBUG=INFO

MASTER_ADDR=\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\" | head -n1)
GPUS_PER_NODE=\${SLURM_GPUS_ON_NODE:-4}

python -m torch.distributed.run \\
    --master-addr=\"\$MASTER_ADDR\" \\
    --master-port=29500 \\
    --node-rank=\"\$SLURM_PROCID\" \\
    --nnodes=\"\$SLURM_NNODES\" \\
    --nproc-per-node=\"\$GPUS_PER_NODE\" \\
    ${VERIFY_SCRIPT}
"

echo "=========================================="
echo "End time: $(date)"
echo "Verification complete. Check the output above for PASSED / FAILED."
echo "=========================================="
