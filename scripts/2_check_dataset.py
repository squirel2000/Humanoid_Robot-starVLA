#!/usr/bin/env python3
"""Validate the OpenArm O6 dataset and report the StarVLA mapping.

This checks the local LeRobot-style metadata and verifies that this repo's
OpenArm registry can build a StarVLA dataloader for the right-arm/right-hand
subset used by the can-sorting task.

The raw dataset is bimanual and 26-D:
    left arm 7 + right arm 7 + left hand 6 + right hand 6.

For this can-sorting run, only the right arm and right LinkerHand O6 are active,
so the StarVLA registry slices the raw vectors down to 13-D:
    right arm state/action[7:14] + right hand state/action[20:26].
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import tyro

from starvla_launch_config import (
    DATALOADER_SCRIPT,
    REPO_ROOT,
    CheckDatasetConfig,
    child_environment,
)


def load_json(path: Path) -> dict:
    """Read a JSON metadata file from the LeRobot dataset."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(path: Path, description: str) -> None:
    """Raise a clear error if an expected file/directory is absent."""
    if not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")


def count_files(root: Path, pattern: str) -> int:
    """Count files matching a glob pattern without materializing a big list."""
    return sum(1 for _ in root.glob(pattern))


def check_metadata(dataset_root: Path) -> tuple[dict, dict]:
    """Validate the pieces StarVLA needs from a LeRobot v2-style dataset."""
    meta = dataset_root / "meta"

    # StarVLA's dataloader expects the standard LeRobot directory layout:
    #   meta/*.json, data/chunk-*/episode_*.parquet, videos/chunk-*/*.mp4.
    require(meta, "LeRobot meta directory")
    for name in ("info.json", "modality.json", "stats.json", "tasks.jsonl", "episodes.jsonl"):
        require(meta / name, name)
    require(dataset_root / "data", "data directory")
    require(dataset_root / "videos", "videos directory")

    info = load_json(meta / "info.json")
    modality = load_json(meta / "modality.json")

    if info.get("codebase_version") != "v2.0":
        print(f"WARNING: expected LeRobot v2.0 metadata, saw {info.get('codebase_version')!r}")

    # These are the raw LeRobot feature names used by this dataset. The
    # friendlier names such as `state.right_arm` come from meta/modality.json.
    for key in ("observation.state", "action", "observation.images.camera"):
        if key not in info.get("features", {}):
            raise KeyError(f"info.json features missing {key!r}")

    # Check the exact right-side spans. If the dataset conversion changes joint
    # order, this catches it before training silently learns the wrong mapping.
    expected = {
        ("state", "right_arm"): (7, 14),
        ("state", "right_hand"): (20, 26),
        ("action", "right_arm"): (7, 14),
        ("action", "right_hand"): (20, 26),
    }
    for (section, name), span in expected.items():
        item = modality.get(section, {}).get(name)
        if item is None:
            raise KeyError(f"modality.json missing {section}.{name}")
        got = (item.get("start"), item.get("end"))
        if got != span:
            raise ValueError(f"{section}.{name} expected span {span}, got {got}")

    return info, modality


def print_summary(dataset_root: Path, info: dict, modality: dict) -> None:
    """Print the raw dataset shape and the reduced StarVLA training shape."""
    state_shape = info["features"]["observation.state"]["shape"][0]
    action_shape = info["features"]["action"]["shape"][0]
    video_info = info["features"]["observation.images.camera"]["info"]

    print("== Dataset summary ==")
    print(f"path           : {dataset_root}")
    print(f"robot_type     : {info.get('robot_type')}")
    print(f"episodes       : {info.get('total_episodes')}")
    print(f"frames         : {info.get('total_frames')}")
    print(f"fps            : {info.get('fps')}")
    print(f"state/action   : {state_shape} / {action_shape}")
    print(f"video          : {video_info['video.width']}x{video_info['video.height']} {video_info['video.codec']}")
    print(f"parquet files  : {count_files(dataset_root / 'data', 'chunk-*/*.parquet')}")
    print(f"video files    : {count_files(dataset_root / 'videos', 'chunk-*/observation.images.camera/*.mp4')}")
    print()

    print("== StarVLA training subset ==")
    for section in ("state", "action"):
        ra = modality[section]["right_arm"]
        rh = modality[section]["right_hand"]
        print(f"{section}.right_arm  -> observation/action[{ra['start']}:{ra['end']}]")
        print(f"{section}.right_hand -> observation/action[{rh['start']}:{rh['end']}]")
    print("model dims      : state_dim=13, action_dim=13")
    print("left side       : present in dataset, intentionally excluded by registry config")
    print()


def run_dataloader_check(config_yaml: Path) -> None:
    """Run StarVLA's own dataloader debug script against the OpenArm config."""
    print("== StarVLA dataloader check ==")
    cmd = ["python", str(DATALOADER_SCRIPT), "--config_yaml", str(config_yaml)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=child_environment(), cwd=REPO_ROOT)


def main(config: CheckDatasetConfig) -> int:
    dataset_root = config.dataset_path.resolve()
    config_yaml = config.config_yaml.resolve()

    info, modality = check_metadata(dataset_root)
    print_summary(dataset_root, info, modality)

    require(config_yaml, "StarVLA OpenArm config")
    if not config.skip_dataloader:
        run_dataloader_check(config_yaml)

    print("Dataset check PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(CheckDatasetConfig, description=__doc__)))
    except Exception as exc:
        print(f"Dataset check FAIL: {exc}", file=sys.stderr)
        raise
