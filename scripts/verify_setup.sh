#!/usr/bin/env bash
# ============================================================================
# GSQ — single-node setup verification
# ============================================================================
# Sanity-checks the host environment: imports torch, prints CUDA info, runs a
# tiny all-reduce across all visible GPUs (if there are >= 2). No Slurm.
# ============================================================================

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

NPROC="${NPROC:-$(detect_num_gpus)}"
[[ "${NPROC}" -lt 1 ]] && NPROC=1

echo "=========================================="
echo "GSQ verify_setup"
echo "Visible GPUs : ${NPROC}"
echo "=========================================="

cd "${REPO_ROOT}"

if [[ "${NPROC}" -ge 2 ]]; then
    exec torchrun --standalone --nproc-per-node="${NPROC}" \
        "${REPO_ROOT}/scripts/verify_smoke_test.py"
fi

python - <<'PYEOF'
import sys
print(f"Python: {sys.version}")
try:
    import torch
    print(f"torch:  {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name}  {props.total_memory/1e9:.1f} GB")
        x = torch.ones(1024, device='cuda')
        y = (x * 2).sum().item()
        print(f"GPU smoke matmul ok: {y}")
    else:
        print("WARN: CUDA not available")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
PYEOF

echo "Single-GPU verification complete."
