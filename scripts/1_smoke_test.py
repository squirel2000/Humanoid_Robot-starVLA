#!/usr/bin/env python3
"""Smoke-test StarVLA on the OpenArm O6 config.

This follows the same launcher shape as Isaac-GR00T:
    config = tyro.cli(...)
    validate/register paths
    build the runtime commands
    run them
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import tyro

from starvla_launch_config import (
    DATALOADER_SCRIPT,
    FRAMEWORK_SCRIPT,
    REPO_ROOT,
    StarVLASmokeTestConfig,
    check_base_vlm,
    child_environment,
)


def _run(command: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True, env=env)


def main(config: StarVLASmokeTestConfig) -> None:
    config_yaml = config.config_yaml.resolve()
    base_vlm = config.base_vlm.resolve()

    print("== StarVLA smoke test ==")
    print(f"repo        : {REPO_ROOT}")
    print(f"python      : {sys.executable}")
    print(f"config      : {config_yaml}")
    print(f"base VLM    : {base_vlm}")
    print()

    check_base_vlm(base_vlm)
    env = child_environment()

    _run(
        [
            sys.executable,
            "-c",
            (
                "import sys, torch; "
                "print(f'python: {sys.executable}'); "
                "print(f'torch : {torch.__version__}'); "
                "print(f'cuda  : {torch.cuda.is_available()}'); "
                "print(f'gpu   : {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'gpu   : <none>')"
            ),
        ],
        env,
    )

    if not config.skip_dataloader:
        _run([sys.executable, str(DATALOADER_SCRIPT), "--config_yaml", str(config_yaml)], env)

    if not config.skip_framework:
        _run([sys.executable, str(FRAMEWORK_SCRIPT), "--config_yaml", str(config_yaml)], env)

    print("Smoke test PASS")


if __name__ == "__main__":
    main(tyro.cli(StarVLASmokeTestConfig, description=__doc__))
