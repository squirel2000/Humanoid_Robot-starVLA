#!/usr/bin/env bash
# StarVLA environment setup script
#
# Tested configuration:
#   GPU    : NVIDIA RTX 4090 (Ada, sm_89) / NVIDIA RTX 5090 (Blackwell, sm_100)
#   CUDA   : 12.8  (nvcc V12.8.93)
#   Python : 3.10
#
# Why PyTorch 2.7.0 instead of the 2.6.0 noted in requirements.txt?
#   PyTorch 2.7.0 + cu128 provides future-proofing compatibility for integrating 
#   Isaac-GR00T models in the long run. It natively supports both RTX 4090 (sm_89) 
#   and RTX 5090 (sm_100). torchvision 0.22.0 is the matching version.
#
# Why flash-attn builds from source?
#   flash-attn 2.7.4.post1 prebuilt wheels only exist for specific torch/CUDA
#   combos. Building with --no-build-isolation uses the already-installed torch
#   and the system CUDA 12.8 toolkit, guaranteeing a compatible build.
#   Build time: ~15-30 min depending on CPU core count.
#   MAX_JOBS below caps parallel compilation to avoid OOM on the build host.
#
# Flash-attn + Blackwell note:
#   flash-attn 2.7.4.post1 ships kernels for sm_80/86/89/90. For sm_100 it
#   compiles via nvcc and may fall back to non-specialised paths at runtime.
#   If you observe reduced throughput, consider upgrading to flash-attn 3.x
#   which has explicit Blackwell kernel support.

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
ENV_NAME="starVLA"
PYTHON_VER="3.10"
TORCH_VER="2.7.0"
TORCHVISION_VER="0.22.0"
TORCHAUDIO_VER="2.7.0"
FLASH_ATTN_VER="2.7.4.post1"
CUDA_TAG="cu128"                                        # CUDA 12.8
PYTORCH_WHL_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
MAX_JOBS="${MAX_JOBS:-4}"                               # parallel nvcc workers for flash-attn
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Resolve CUDA 12.8 toolkit path ─────────────────────────────────────────────
# flash-attn's setup.py reads CUDA_HOME to locate nvcc and CUDA headers.
if   [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME="/usr/local/cuda-12.8"
elif [ -d "/usr/local/cuda" ] && nvcc --version 2>/dev/null | grep -q "12\.8"; then
    export CUDA_HOME="/usr/local/cuda"
else
    echo "ERROR: CUDA 12.8 toolkit not found under /usr/local/cuda-12.8 or /usr/local/cuda." >&2
    echo "       Verify with: nvcc --version" >&2
    exit 1
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "======================================================"
echo " StarVLA environment setup"
echo " CUDA_HOME : ${CUDA_HOME}"
echo " ENV_NAME  : ${ENV_NAME}"
echo " PyTorch   : ${TORCH_VER}+${CUDA_TAG}"
echo "======================================================"
nvcc --version
echo ""

# ── Step 1: Create conda environment ───────────────────────────────────────────
echo "[1/5] Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VER})"
conda create -n "${ENV_NAME}" python="${PYTHON_VER}" -y

# Activate — works in both interactive shells and non-interactive CI.
# shellcheck source=/dev/null
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "      Active Python: $(python --version)"

# ── Step 2: Install PyTorch 2.7.0 + CUDA 12.8 ──────────────────────────────────
echo ""
echo "[2/5] Installing PyTorch ${TORCH_VER}+${CUDA_TAG}, torchvision ${TORCHVISION_VER}"
pip install \
    "torch==${TORCH_VER}+${CUDA_TAG}" \
    "torchvision==${TORCHVISION_VER}+${CUDA_TAG}" \
    "torchaudio==${TORCHAUDIO_VER}" \
    --index-url "${PYTORCH_WHL_URL}"

# Quick sanity-check: confirm CUDA is visible before proceeding.
python - <<'PYEOF'
import torch, sys
assert torch.cuda.is_available(), "CUDA not available after PyTorch install — check driver/toolkit."
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"      GPU: {name}  (sm_{cap[0]}{cap[1]})")
print(f"      PyTorch {torch.__version__}  CUDA runtime {torch.version.cuda}")
PYEOF

# ── Step 3: Install project requirements ───────────────────────────────────────
# torchvision==0.21.0 in requirements.txt targets torch 2.6.0; skip that line
# and keep the 0.22.0 installed above.
echo ""
echo "[3/5] Installing requirements.txt (torchvision pin excluded)"
grep -v "^torchvision" "${SCRIPT_DIR}/requirements.txt" \
    | pip install -r /dev/stdin

# idna: transitive dep of yarl (used by huggingface_hub, websocket libs).
# Without it, pip warns of unmet dependencies. Installing explicitly.
pip install idna --quiet

# ── Step 4: Install flash-attn (build from source) ─────────────────────────────
echo ""
echo "[4/5] Building flash-attn ${FLASH_ATTN_VER} from source"
echo "      This typically takes 15–30 minutes. MAX_JOBS=${MAX_JOBS}"
echo "      Increase MAX_JOBS (export MAX_JOBS=8) to speed up on machines with"
echo "      many CPU cores, but watch RAM usage (each worker uses ~2 GB)."
MAX_JOBS="${MAX_JOBS}" pip install "flash-attn==${FLASH_ATTN_VER}" --no-build-isolation

# ── Step 5: Install starVLA package (editable) ─────────────────────────────────
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
    name = torch.cuda.get_device_name(0)
    cap  = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU             : {name}")
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
