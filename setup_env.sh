#!/usr/bin/env bash
# StarVLA environment setup — portable across CUDA and GPU generations.
#
# Tested on:
#   RTX 4090 (sm_89, Ada)        + CUDA 12.8 + Py 3.10
#   RTX 5090 (sm_100, Blackwell) + CUDA 12.8 + Py 3.10
#   A100     (sm_80, Ampere)     + CUDA 12.4 + Py 3.10
#   H100     (sm_90, Hopper)     + CUDA 12.4/12.6 + Py 3.10
#
# What this script does
# ---------------------
#   1. Detects (or accepts overrides for) Python, CUDA, PyTorch versions.
#   2. Creates a conda env and installs a CUDA-matched PyTorch wheel.
#   3. Installs requirements.txt (minus pinned torchvision, which we manage).
#   4. Builds flash-attn from source against the installed torch + CUDA.
#   5. Installs the starVLA package in editable mode and prints a summary.
#
# Override knobs (export before running, all optional)
# ----------------------------------------------------
#   ENV_NAME        conda env name                       (default: starVLA)
#   PYTHON_VER      Python version                       (default: 3.10)
#   CUDA_HOME       path to a CUDA toolkit               (auto-detect)
#   CUDA_TAG        wheel suffix: cu121 / cu124 / cu128  (auto-derived from CUDA_HOME)
#   TORCH_VER       PyTorch version                      (auto-derived from CUDA_TAG)
#   FLASH_ATTN_VER  flash-attn version                   (default: 2.7.4.post1)
#   MAX_JOBS        parallel nvcc workers for flash-attn (default: 4)
#
# Notes on GPU architectures
# --------------------------
#   - flash-attn 2.x ships kernels for sm_80/86/89/90. On sm_100 (Blackwell)
#     it falls back to generic paths; upgrade to flash-attn 3.x once stable
#     for native Blackwell kernels.
#   - On H100/A100, CUDA 12.1+ is fine; 12.4 is the most-tested combo.
#   - On RTX 5090, you NEED CUDA 12.8 — earlier toolkits don't know sm_100.

set -euo pipefail

# ── Configuration (overridable) ────────────────────────────────────────────────
ENV_NAME="${ENV_NAME:-starVLA}"
PYTHON_VER="${PYTHON_VER:-3.10}"
FLASH_ATTN_VER="${FLASH_ATTN_VER:-2.7.4.post1}"
MAX_JOBS="${MAX_JOBS:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Resolve CUDA toolkit ───────────────────────────────────────────────────────
# Search order: explicit CUDA_HOME → /usr/local/cuda-<X.Y> dirs → /usr/local/cuda.
resolve_cuda_home() {
    if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
        return 0
    fi
    local cand
    for cand in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 \
                /usr/local/cuda-12.1 /usr/local/cuda; do
        if [ -x "${cand}/bin/nvcc" ]; then
            export CUDA_HOME="${cand}"
            return 0
        fi
    done
    return 1
}

if ! resolve_cuda_home; then
    echo "ERROR: no CUDA toolkit found. Set CUDA_HOME=/path/to/cuda or install one under /usr/local." >&2
    echo "       Verify with: which nvcc && nvcc --version" >&2
    exit 1
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# ── Derive CUDA_TAG (e.g. "cu128") from nvcc version, unless overridden ────────
if [ -z "${CUDA_TAG:-}" ]; then
    NVCC_VER="$(nvcc --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -n1)"
    case "${NVCC_VER}" in
        12.8|12.9) CUDA_TAG="cu128" ;;
        12.6|12.7) CUDA_TAG="cu126" ;;
        12.4|12.5) CUDA_TAG="cu124" ;;
        12.1|12.2|12.3) CUDA_TAG="cu121" ;;
        11.8) CUDA_TAG="cu118" ;;
        *)
            echo "WARNING: unrecognised CUDA ${NVCC_VER}; defaulting to cu121. Set CUDA_TAG to override." >&2
            CUDA_TAG="cu121"
            ;;
    esac
fi

# ── Pick a PyTorch version compatible with the chosen CUDA tag ─────────────────
# Mapping: each wheel index hosts a small range of torch versions. We pick the
# most recent torch that's broadly tested for each tag. Override TORCH_VER if
# you have a specific reason (e.g. compiling against an old extension).
if [ -z "${TORCH_VER:-}" ]; then
    case "${CUDA_TAG}" in
        cu128) TORCH_VER="2.7.0"; TORCHVISION_VER="0.22.0"; TORCHAUDIO_VER="2.7.0" ;;
        cu126) TORCH_VER="2.6.0"; TORCHVISION_VER="0.21.0"; TORCHAUDIO_VER="2.6.0" ;;
        cu124) TORCH_VER="2.5.1"; TORCHVISION_VER="0.20.1"; TORCHAUDIO_VER="2.5.1" ;;
        cu121) TORCH_VER="2.4.1"; TORCHVISION_VER="0.19.1"; TORCHAUDIO_VER="2.4.1" ;;
        cu118) TORCH_VER="2.4.1"; TORCHVISION_VER="0.19.1"; TORCHAUDIO_VER="2.4.1" ;;
        *) echo "ERROR: no torch mapping for CUDA_TAG=${CUDA_TAG}" >&2; exit 1 ;;
    esac
fi
TORCHVISION_VER="${TORCHVISION_VER:-0.22.0}"
TORCHAUDIO_VER="${TORCHAUDIO_VER:-${TORCH_VER}}"
PYTORCH_WHL_URL="https://download.pytorch.org/whl/${CUDA_TAG}"

echo "======================================================"
echo " StarVLA environment setup"
echo " ENV_NAME   : ${ENV_NAME}"
echo " Python     : ${PYTHON_VER}"
echo " CUDA_HOME  : ${CUDA_HOME}"
echo " CUDA_TAG   : ${CUDA_TAG}"
echo " PyTorch    : ${TORCH_VER}+${CUDA_TAG}"
echo " flash-attn : ${FLASH_ATTN_VER}"
echo "======================================================"
nvcc --version | tail -n 1
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -n 1
fi
echo ""

# ── Step 1: Conda env ──────────────────────────────────────────────────────────
echo "[1/5] Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VER})"
conda create -n "${ENV_NAME}" python="${PYTHON_VER}" -y

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "      Active Python: $(python --version)"

# ── Step 2: PyTorch ────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Installing PyTorch ${TORCH_VER}+${CUDA_TAG}, torchvision ${TORCHVISION_VER}"
pip install \
    "torch==${TORCH_VER}+${CUDA_TAG}" \
    "torchvision==${TORCHVISION_VER}+${CUDA_TAG}" \
    "torchaudio==${TORCHAUDIO_VER}" \
    --index-url "${PYTORCH_WHL_URL}"

python - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "CUDA not available after PyTorch install — check driver/toolkit."
cap = torch.cuda.get_device_capability(0)
print(f"      GPU: {torch.cuda.get_device_name(0)}  (sm_{cap[0]}{cap[1]})")
print(f"      PyTorch {torch.__version__}  CUDA runtime {torch.version.cuda}")
PYEOF

# ── Step 3: Project requirements ───────────────────────────────────────────────
# Exclude torchvision so we don't downgrade the version installed in step 2.
echo ""
echo "[3/5] Installing requirements.txt (torchvision pin excluded)"
grep -v "^torchvision" "${SCRIPT_DIR}/requirements.txt" | pip install -r /dev/stdin
pip install idna --quiet  # transitive dep of yarl

# ── Step 4: flash-attn (built from source against the installed torch) ─────────
echo ""
echo "[4/5] Building flash-attn ${FLASH_ATTN_VER} from source (15–30 min, MAX_JOBS=${MAX_JOBS})"
echo "      Each parallel worker uses ~2 GB RAM; raise MAX_JOBS if you have RAM,"
echo "      lower it if your build host OOMs."
MAX_JOBS="${MAX_JOBS}" pip install "flash-attn==${FLASH_ATTN_VER}" --no-build-isolation

# ── Step 5: starVLA package (editable) ─────────────────────────────────────────
echo ""
echo "[5/5] Installing starVLA package in editable mode"
pip install -e "${SCRIPT_DIR}"

# ── Verification ───────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Verification"
echo "======================================================"
python - <<'PYEOF'
import sys, torch, torchvision, flash_attn
print(f"  Python          : {sys.version.split()[0]}")
print(f"  PyTorch         : {torch.__version__}")
print(f"  CUDA runtime    : {torch.version.cuda}")
print(f"  torchvision     : {torchvision.__version__}")
print(f"  flash-attn      : {flash_attn.__version__}")
print(f"  CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    cap  = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    print(f"  Compute cap.    : sm_{cap[0]}{cap[1]}")
    print(f"  VRAM            : {vram:.1f} GB")
PYEOF

echo ""
echo "======================================================"
echo " Setup complete!"
echo " Activate with:  conda activate ${ENV_NAME}"
echo "======================================================"
echo ""
echo "Next step — download the base model:"
echo ""
echo "  conda activate ${ENV_NAME}"
echo "  pip install 'huggingface_hub[cli]'"
echo "  huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \\"
echo "      --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
echo ""
echo "Then run the smoke test:"
echo "  python starVLA/model/framework/QwenGR00T.py"
echo "  (WARNING messages during model loading are normal)"
