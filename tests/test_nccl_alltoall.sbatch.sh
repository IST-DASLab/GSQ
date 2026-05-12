#!/usr/bin/env bash
#SBATCH --job-name=nccl-a2a-test
#SBATCH --account=a-g200
#SBATCH --partition=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH -C thp_never&nvidia_vboost_enabled
#SBATCH --output=logs/slurm/nccl_test_%j.out
#SBATCH --error=logs/slurm/nccl_test_%j.err

set -euo pipefail
ulimit -c 0

SCRIPT_ROOT="${HOME}/iopsstor/workspace/GSQ-Dev"
EDF_FILE="${SCRIPT_ROOT}/clariden/gsq.toml"
SCRATCH="/iopsstor/scratch/cscs/${USER}"
VENV_PATH="${SCRATCH}/gsq/venv-gsq"

if [ -f "${SCRIPT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${SCRIPT_ROOT}/.env" | xargs)
fi

mkdir -p "${SCRIPT_ROOT}/logs/slurm"
cd "${SCRIPT_ROOT}"

srun -ul --mpi=pmix --environment="${EDF_FILE}" --label bash -c "
set -euo pipefail
source ${VENV_PATH}/bin/activate

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4}
unset SSL_CERT_FILE

MASTER_ADDR=\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\" | head -n1)
export RANK=\${SLURM_PROCID}
export LOCAL_RANK=\${SLURM_LOCALID}
export WORLD_SIZE=\${SLURM_NTASKS}
export MASTER_ADDR=\${MASTER_ADDR}
export MASTER_PORT=29500

export NCCL_DEBUG=WARN

python ${SCRIPT_ROOT}/tests/test_nccl_alltoall.py
"
