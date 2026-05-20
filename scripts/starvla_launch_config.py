#!/usr/bin/env python3
"""Shared tyro configs for StarVLA smoke-test and training launchers.

The layout mirrors Isaac-GR00T's launch/config split:
    - scripts/1_smoke_test.py contains launcher logic.
    - scripts/3_train_starvla.py contains launcher logic.
    - this file contains typed dataclass configuration.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Manually-edited path constants. Edit these when moving the repo to a new
# machine / workspace layout. CLI flags (--dataset_root, --output_dir) override
# these at runtime.
# ---------------------------------------------------------------------------

DEFAULT_DATA_ROOT = Path("/home/asus/Gits/IsaacLab-GR00T/datasets")
DEFAULT_DATASET   = DEFAULT_DATA_ROOT / "OpenArm_O6_CanSorting_dataset_0408"
DEFAULT_OUT_DIR   = Path("/home/asus/Gits/IsaacLab-GR00T/artifacts/checkpoints/starvla")


# ---------------------------------------------------------------------------
# Repository-level constants
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[1]

# Default model / data paths
DEFAULT_BASE_VLM = REPO_ROOT / "playground/Pretrained_models/Qwen3-VL-4B-Instruct"
# IMPORTANT: This must be a Qwen3-VL build that is API-compatible with the
# StarVLA QwenGR00T framework (which expects Qwen3-VL Instruct + flash-attn).
DEFAULT_BASE_VLM_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_CONFIG   = REPO_ROOT / "examples/OpenArm_O6/train_files/starvla_train_openarm_o6.yaml"

# StarVLA entry-point scripts (avoids repeating long paths in every launcher)
DATALOADER_SCRIPT = REPO_ROOT / "starVLA/dataloader/lerobot_datasets.py"
FRAMEWORK_SCRIPT  = REPO_ROOT / "starVLA/model/framework/VLM4A/QwenGR00T.py"
TRAIN_SCRIPT      = REPO_ROOT / "starVLA/training/train_starvla.py"


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

def _vlm_is_complete(base_vlm: Path) -> bool:
    """Return True only if the base VLM directory has both metadata and shards."""
    if not base_vlm.is_dir():
        return False
    index = base_vlm / "model.safetensors.index.json"
    if not index.is_file():
        # Single-shard checkpoints land here. A lone model.safetensors counts.
        return (base_vlm / "model.safetensors").is_file()
    return bool(list(base_vlm.glob("model-*.safetensors")))


def download_base_vlm(
    base_vlm: Path,
    repo_id: str = DEFAULT_BASE_VLM_REPO_ID,
) -> None:
    """Download the Qwen3-VL VLM into ``base_vlm`` with live progress.

    Streams ``huggingface-cli download`` stdout to the terminal so the user can
    see per-shard progress instead of staring at a frozen prompt. The Qwen3-VL
    4B instruct shards weigh ~9 GB; on a typical home connection the download
    takes 5-30 minutes.
    """
    # Prefer the modern `hf` CLI; fall back to the older `huggingface-cli` if
    # someone is on an old huggingface_hub.
    cli = "hf" if shutil.which("hf") else "huggingface-cli"
    if shutil.which(cli) is None:
        raise RuntimeError(
            "Neither `hf` nor `huggingface-cli` found in PATH. Install with:\n"
            "  pip install -U 'huggingface_hub[cli]'"
        )

    base_vlm.mkdir(parents=True, exist_ok=True)
    print("─" * 72, flush=True)
    print(f"Downloading base VLM '{repo_id}'", flush=True)
    print(f"  destination : {base_vlm}", flush=True)
    print(f"  repo size   : ~9 GB (Qwen3-VL 4B Instruct)", flush=True)
    print(f"  expected    : 5–30 min depending on bandwidth.", flush=True)
    print("─" * 72, flush=True)

    cmd = [
        cli,
        "download",
        repo_id,
        "--local-dir",
        str(base_vlm),
    ]

    # Enable hf_transfer (Rust-based parallel chunked download) when the
    # package is installed. On slow links, this is 5-10x faster than the pure
    # Python downloader.
    download_env = os.environ.copy()
    try:
        import importlib
        importlib.import_module("hf_transfer")
        download_env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("[check_base_vlm] hf_transfer available → parallel download enabled.", flush=True)
    except ImportError:
        print(
            "[check_base_vlm] hf_transfer not installed; using single-stream download.\n"
            "  Install for ~10x speedup on slow links:  pip install hf_transfer",
            flush=True,
        )

    # No timeout: streaming subprocess.run inherits the parent stdout/stderr so
    # the hf-cli progress bars render live. Avoid capturing output (would buffer).
    proc = subprocess.run(cmd, env=download_env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"huggingface-cli exited with code {proc.returncode}. "
            f"Re-run after fixing the network / auth issue:\n"
            f"  {' '.join(cmd)}"
        )

    if not _vlm_is_complete(base_vlm):
        raise RuntimeError(
            f"Download finished but {base_vlm} still has no weight shards. "
            f"Inspect the directory and retry:\n"
            f"  ls {base_vlm}"
        )
    print(f"✓ VLM ready at {base_vlm}", flush=True)


def check_base_vlm(base_vlm: Path, *, auto_download: bool = True) -> None:
    """Validate that ``base_vlm`` contains a usable Qwen3-VL checkpoint.

    If the directory is missing or holds only the metadata files (model.safetensors.index.json
    without the matching shards), this will trigger a fresh download by default.
    Set ``auto_download=False`` to surface a hard error instead — useful for CI.
    """
    if _vlm_is_complete(base_vlm):
        return

    reason = (
        "directory does not exist" if not base_vlm.is_dir()
        else "weight shards (model-*.safetensors) are missing"
    )
    print(f"[check_base_vlm] base VLM at {base_vlm}: {reason}.", flush=True)

    if not auto_download:
        raise FileNotFoundError(
            f"base VLM not ready at {base_vlm} ({reason}). "
            f"Download it with:\n"
            f"  huggingface-cli download {DEFAULT_BASE_VLM_REPO_ID} --local-dir {base_vlm}"
        )

    download_base_vlm(base_vlm)


# ---------------------------------------------------------------------------
# Dataclass configs (one per launcher script)
# ---------------------------------------------------------------------------

@dataclass
class StarVLASmokeTestConfig:
    """Configuration for validating the StarVLA environment and OpenArm data path."""

    base_vlm: Path = DEFAULT_BASE_VLM
    """Local Qwen3-VL checkpoint directory."""

    config_yaml: Path = DEFAULT_CONFIG
    """StarVLA YAML config used by the dataloader and QwenGR00T smoke test."""

    skip_dataloader: bool = False
    """Skip the StarVLA LeRobot dataloader check."""

    skip_framework: bool = False
    """Skip the QwenGR00T forward/predict_action check."""


@dataclass
class CheckDatasetConfig:
    """Configuration for validating the OpenArm O6 dataset."""

    dataset_path: Path = DEFAULT_DATASET
    """Path to the LeRobot-style dataset root directory."""

    config_yaml: Path = DEFAULT_CONFIG
    """StarVLA YAML config used by the dataloader check."""

    skip_dataloader: bool = False
    """Skip the StarVLA dataloader check (fast metadata validation only)."""


@dataclass
class StarVLATrainConfig:
    """Configuration for launching StarVLA fine-tuning on OpenArm O6.

    Mirrors Isaac-GR00T's `FinetuneConfig`: a flat dataclass exposing only the
    arguments users typically tune. **Defaults below mirror the Path A recipe
    in `starvla_train_openarm_o6.yaml`** — VLM unfrozen, 200k steps,
    bs=4×grad_accum=8 sized for a 24 GB 4090. Anything not exposed here is
    read from `config_yaml`; edit that file for advanced tuning (optimizer
    betas, scheduler kwargs, etc.). For the common case, the only CLI flag
    you need is `--run_id`.
    """

    # --- Paths ---
    base_vlm: Path = DEFAULT_BASE_VLM
    """Local Qwen3-VL checkpoint directory."""

    config_yaml: Path = DEFAULT_CONFIG
    """StarVLA YAML config to pass to `train_starvla.py`."""

    dataset_root: Path = DEFAULT_DATA_ROOT
    """Root containing shared project datasets used by StarVLA data_mix entries."""

    output_dir: Path = DEFAULT_OUT_DIR
    """Parent directory for training outputs."""

    run_id: str = "openarm_o6_qwengroot_right_only"
    """Experiment/run name under `output_dir`."""

    # --- Compute ---
    num_processes: int = 1
    """Number of Accelerate processes, usually number of GPUs on one node."""

    num_machines: int = 1
    """Number of machines for multi-node Accelerate launch."""

    mixed_precision: str = "bf16"
    """Accelerate mixed precision mode: no, fp16, bf16, or fp8."""

    dynamo_backend: str = "no"
    """Accelerate dynamo backend. Keep `no` unless intentionally compiling."""

    use_deepspeed: bool | None = None
    """Use DeepSpeed. Defaults to False for one process and True for multi-GPU."""

    # --- Training hyperparameters ---
    max_train_steps: int = 200000
    """Total optimizer steps. Path A: 200k (was 100k while VLM was frozen)."""

    per_device_batch_size: int = 4
    """Per-GPU batch size. With VLM unfrozen, 4 fits a 24 GB 4090; raise to
    8-16 on 80 GB H100/A100 (and lower gradient_accumulation_steps in turn)."""

    gradient_accumulation_steps: int = 8
    """Forward passes accumulated per optimizer step. 4×8 = effective bs=32,
    a bit larger than the previous bs16 frozen run for stability."""

    num_warmup_steps: int = 1000
    """Linear warmup steps for the LR scheduler. Path A: longer warmup since
    the VLM is updating too."""

    weight_decay: float = 1e-5
    """AdamW weight decay (maps to `trainer.optimizer.weight_decay`)."""

    learning_rate_action_model: float = 5.0e-5
    """LR for the DiT action head. Path A: kept at 5e-5; the head still has
    the most to learn but doesn't need the YAML's old 1e-4."""

    learning_rate_qwen_vl: float = 5.0e-6
    """LR for the QwenVL interface. Active in Path A (VLM unfrozen).
    Kept small so the pretrained backbone drifts gently."""

    learning_rate_base: float = 5.0e-6
    """Default LR for any module not matched by the per-module overrides above."""

    gradient_clipping: float = 1.0
    """Global gradient-norm clip threshold."""

    freeze_modules: str = ""
    """Comma-separated module paths to freeze. Path A: empty = nothing frozen,
    so Qwen3-VL adapts to the can-sorting visual domain. Set to
    `qwen_vl_interface` to reproduce the old frozen-VLM recipe."""

    seed: int = 42
    """Top-level RNG seed shared across rank 0."""

    # --- Logging / checkpointing ---
    save_interval: int = 2000
    """Checkpoint interval in training steps."""

    max_checkpoints_to_keep: int = 5
    """Number of latest periodic checkpoints to keep. Set <=0 to disable pruning."""

    eval_interval: int = 2000
    """Action-eval interval in training steps."""

    logging_frequency: int = 100
    """Metric logging interval in training steps."""

    # --- W&B ---
    use_wandb: bool = False
    """If False, child runs in offline mode and nothing is uploaded."""

    wandb_project: str = "starVLA_OpenArm_O6"
    """W&B project name."""

    def resolved_use_deepspeed(self) -> bool:
        """Resolve `auto` DeepSpeed behavior in the same spirit as the shell script."""
        if self.use_deepspeed is not None:
            return self.use_deepspeed
        return self.num_processes > 1


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def child_environment(
    *,
    use_wandb: bool = False,
    use_deepspeed: bool | None = None,
    gradient_accumulation_steps: int | None = None,
) -> dict[str, str]:
    """Build an environment for child StarVLA processes.

    The always-on settings (TOKENIZERS_PARALLELISM, NO_ALBUMENTATIONS_UPDATE,
    PYTORCH_CUDA_ALLOC_CONF, torchvision deprecation filter, WANDB_SILENT) are
    hardcoded — they are noise suppressors, not user-tunable knobs.

    `gradient_accumulation_steps` is bridged via STARVLA_GRAD_ACCUM because
    `accelerate launch --gradient_accumulation_steps` is a no-op for
    non-DeepSpeed runs (it only configures the DeepSpeed plugin). The trainer
    reads STARVLA_GRAD_ACCUM at module import and passes it to `Accelerator(...)`.
    """
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    env["WANDB_SILENT"] = "true"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not use_wandb:
        env["WANDB_MODE"] = "offline"
    if use_deepspeed is not None:
        env["STARVLA_USE_DEEPSPEED"] = "true" if use_deepspeed else "false"
    if gradient_accumulation_steps is not None:
        env["STARVLA_GRAD_ACCUM"] = str(gradient_accumulation_steps)

    warning_filter = (
        "ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning"
    )
    env["PYTHONWARNINGS"] = (
        f"{env['PYTHONWARNINGS']},{warning_filter}" if env.get("PYTHONWARNINGS") else warning_filter
    )

    return env
