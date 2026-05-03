#!/usr/bin/env python3
"""Shared tyro configs for StarVLA smoke-test and training launchers.

The layout mirrors Isaac-GR00T's launch/config split:
    - scripts/1_smoke_test.py contains launcher logic.
    - scripts/3_train_starvla.py contains launcher logic.
    - this file contains typed dataclass configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Repository-level constants
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[1]

# Default model / data paths
DEFAULT_BASE_VLM = REPO_ROOT / "playground/Pretrained_models/Qwen3-VL-4B-Instruct"
DEFAULT_CONFIG   = REPO_ROOT / "examples/OpenArm_O6/train_files/starvla_train_openarm_o6.yaml"
DEFAULT_DATASET  = Path("/data/gr00t_datasets/OpenArm_O6_CanSorting_dataset_0408")
DEFAULT_OUT_DIR  = REPO_ROOT / "results/Checkpoints"

# StarVLA entry-point scripts (avoids repeating long paths in every launcher)
DATALOADER_SCRIPT = REPO_ROOT / "starVLA/dataloader/lerobot_datasets.py"
FRAMEWORK_SCRIPT  = REPO_ROOT / "starVLA/model/framework/VLM4A/QwenGR00T.py"
TRAIN_SCRIPT      = REPO_ROOT / "starVLA/training/train_starvla.py"


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

def check_base_vlm(base_vlm: Path) -> None:
    if not base_vlm.is_dir():
        raise FileNotFoundError(
            f"base VLM not found at {base_vlm}\n"
            f"Download it with:\n"
            f"  huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir {base_vlm}"
        )
    if (base_vlm / "model.safetensors.index.json").is_file() and not list(base_vlm.glob("model-*.safetensors")):
        raise FileNotFoundError(
            f"base VLM metadata exists, but safetensors weight shards are missing in {base_vlm}\n"
            f"Complete the download with:\n"
            f"  huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir {base_vlm}"
        )


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
    arguments users typically tune. Defaults are sized for a single 32GB GPU
    (action head only, Qwen frozen). Anything not exposed here is read from
    `config_yaml` — edit that file for advanced tuning (per-module learning
    rates, optimizer betas, scheduler kwargs, etc.).
    """

    # --- Paths ---
    base_vlm: Path = DEFAULT_BASE_VLM
    """Local Qwen3-VL checkpoint directory."""

    config_yaml: Path = DEFAULT_CONFIG
    """StarVLA YAML config to pass to `train_starvla.py`."""

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
    max_train_steps: int = 10000
    """Total optimizer steps."""

    per_device_batch_size: int = 4
    """Per-GPU batch size. Keep small on 32GB GPUs."""

    gradient_accumulation_steps: int = 1
    """Forward passes accumulated before each optimizer step (passed to Accelerate)."""

    num_warmup_steps: int = 100
    """Linear warmup steps for the LR scheduler."""

    weight_decay: float = 1e-8
    """AdamW weight decay (maps to `trainer.optimizer.weight_decay`)."""

    gradient_clipping: float = 1.0
    """Global gradient-norm clip threshold."""

    freeze_modules: str = "qwen_vl_interface"
    """Comma-separated module paths to freeze; empty string enables full fine-tuning."""

    seed: int = 42
    """Top-level RNG seed shared across rank 0."""

    # --- Logging / checkpointing ---
    save_interval: int = 2000
    """Checkpoint interval in training steps."""

    eval_interval: int = 1000
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
