#!/usr/bin/env python3
"""Launch StarVLA fine-tuning on the OpenArm O6 dataset.

The structure intentionally follows Isaac-GR00T's `launch_finetune.py`: tyro
parses a dataclass config, this launcher maps that config onto the project
runtime, then hands off to the actual training entry point.

# Example usage — ./starvla_train_openarm_o6.yaml:
conda activate starVLA
python scripts/3_train_starvla.py \
    --run_id openarm_o6_qwengroot_right_only_bs16_unfrozenVLM_200k \
    --max_train_steps 200000 \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --use_wandb
    
wandb login   # one-time
wandb sync ../artifacts/checkpoints/starvla/openarm_o6_qwengroot_right_only_bs16_50000/wandb/wandb/offline-run-20260504_195745-t46bnqcv

tensorboard --logdir ../artifacts/checkpoints/starvla/<run_id>/tb
# default URL: http://localhost:6006
"""

from __future__ import annotations

import subprocess
import sys

import tyro

from starvla_config import (
    REPO_ROOT,
    TRAIN_SCRIPT,
    StarVLATrainConfig,
    check_base_vlm,
    child_environment,
)


def build_accelerate_command(config: StarVLATrainConfig) -> list[str]:
    # Note: `accelerate launch --gradient_accumulation_steps` is intentionally
    # NOT passed here because that flag is a no-op for non-DeepSpeed runs in
    # accelerate 1.x. The value is bridged via the STARVLA_GRAD_ACCUM env var
    # in `child_environment`, which the trainer reads when constructing the
    # Accelerator (see train_starvla.py).
    return [
        "accelerate",
        "launch",
        "--num_processes",    str(config.num_processes),
        "--num_machines",     str(config.num_machines),
        "--mixed_precision",  config.mixed_precision,
        "--dynamo_backend",   config.dynamo_backend,
        str(TRAIN_SCRIPT),
        "--config_yaml",                              str(config.config_yaml.resolve()),
        "--seed",                                     str(config.seed),
        "--framework.qwenvl.base_vlm",                str(config.base_vlm.resolve()),
        "--datasets.vla_data.data_root_dir",          str(config.dataset_root.resolve()),
        "--datasets.vla_data.per_device_batch_size",  str(config.per_device_batch_size),
        "--trainer.freeze_modules",                   config.freeze_modules,
        "--trainer.max_train_steps",                  str(config.max_train_steps),
        "--trainer.num_warmup_steps",                 str(config.num_warmup_steps),
        "--trainer.gradient_clipping",                str(config.gradient_clipping),
        "--trainer.optimizer.weight_decay",           str(config.weight_decay),
        "--trainer.learning_rate.base",               str(config.learning_rate_base),
        "--trainer.learning_rate.action_model",       str(config.learning_rate_action_model),
        "--trainer.learning_rate.qwen_vl_interface",  str(config.learning_rate_qwen_vl),
        "--trainer.save_interval",                    str(config.save_interval),
        "--trainer.max_checkpoints_to_keep",          str(config.max_checkpoints_to_keep),
        "--trainer.eval_interval",                    str(config.eval_interval),
        "--trainer.logging_frequency",                str(config.logging_frequency),
        "--run_root_dir",                             str(config.output_dir.resolve()),
        "--run_id",                                   config.run_id,
        "--wandb_project",                            config.wandb_project,
    ]


def main(config: StarVLATrainConfig) -> None:
    base_vlm = config.base_vlm.resolve()
    run_dir = config.output_dir.resolve() / config.run_id
    use_deepspeed = config.resolved_use_deepspeed()

    print("== StarVLA OpenArm O6 training ==")
    print(f"repo         : {REPO_ROOT}")
    print(f"python       : {sys.executable}")
    print(f"config       : {config.config_yaml.resolve()}")
    print(f"dataset root : {config.dataset_root.resolve()}")
    print(f"base VLM     : {base_vlm}")
    print(f"run          : {run_dir}")
    print(f"processes    : {config.num_processes} x {config.num_machines} machine(s)")
    print(f"precision    : {config.mixed_precision} (deepspeed={use_deepspeed})")
    print(f"steps        : {config.max_train_steps} (warmup {config.num_warmup_steps})")
    print(f"batch/device : {config.per_device_batch_size} x grad_accum {config.gradient_accumulation_steps}")
    print(f"checkpoints  : every {config.save_interval} steps, {config.max_checkpoints_to_keep}")
    print(f"freeze       : {config.freeze_modules or '<none>'}")
    print(f"wandb        : {'online' if config.use_wandb else 'offline'} ({config.wandb_project})")
    print()

    check_base_vlm(base_vlm)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    env = child_environment(
        use_wandb=config.use_wandb,
        use_deepspeed=use_deepspeed,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    command = build_accelerate_command(config)

    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True, env=env)

    print()
    print(f"Training complete: {run_dir}")
    print(f"Comparison guide : {REPO_ROOT / 'docs/openarm_o6/GR00T_N15_COMPARISON.md'}")


if __name__ == "__main__":
    main(tyro.cli(StarVLATrainConfig, description=__doc__))
