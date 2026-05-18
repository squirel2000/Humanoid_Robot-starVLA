#!/usr/bin/env python3
"""Evaluate a fine-tuned StarVLA checkpoint with per-trajectory action MSE.

Mirrors Isaac-GR00T's `calc_mse_for_single_trajectory` so the resulting
numbers are directly comparable to `Isaac-GR00T/scripts/eval_policy.py`:

* unrolls each trajectory in `action_horizon`-step chunks,
* runs StarVLA's `predict_action` on each chunk,
* applies the inverse of StarVLA's q99 normalization (using
  `dataset_statistics.json` saved at training time) so predictions land in
  the same unnormalized joint-position space the GR00T eval uses,
* prints per-traj MSE + average MSE,
* optionally writes a JSON summary that the comparison wrapper reads.

Run example:

    conda activate starVLA
    python scripts/4_eval_starvla_traj.py \
        --checkpoint ../artifacts/checkpoints/starvla/openarm_o6_qwengroot_right_only_bs16_lr5e5_wd1e5/checkpoints/steps_100000_pytorch_model.pt \
        --dataset_path ../datasets/OpenArm_O6_CanSorting_dataset_0408 \
        --trajs 10 --steps 400 \
        --output_json results/eval/starvla_eval.json
"""

from __future__ import annotations

# Quiet the noisy torchvision deprecation warning that fires repeatedly during
# video decoding. We do this BEFORE importing torchvision (transitively pulled
# in by starVLA.dataloader). Once torchvision migrates to TorchCodec we can
# drop this filter; until then it just adds noise to the eval log.
import warnings  # noqa: E402
warnings.filterwarnings(
    "ignore",
    message=r"The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)
import os  # noqa: E402
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning",
)

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import tyro
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import dataset_path  # noqa: E402
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset  # noqa: E402
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag  # noqa: E402
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T  # noqa: E402

# Auto-load the OpenArm O6 registry (mirrors what train_starvla.py does on
# import: examples/*/train_files/data_registry/data_config.py is read at
# module import to populate ROBOT_TYPE_CONFIG_MAP).
from starVLA.dataloader.gr00t_lerobot.registry import (  # noqa: E402
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """CLI surface for the eval. Defaults match the comparison wrapper."""

    checkpoint: Path
    """Path to a StarVLA fine-tune output directory (must contain pytorch_model.pt)
    or directly to the .pt file. The parent run dir must hold dataset_statistics.json
    and config.full.yaml."""

    dataset_path: Path = dataset_path("OpenArm_O6_CanSorting_dataset_0408", PROJECT_ROOT)
    """Path to the LeRobot-style OpenArm O6 dataset."""

    config_yaml: Path | None = None
    """Override path to the YAML config the checkpoint was trained with.
    Defaults to <run_dir>/config.full.yaml or config.yaml if present, else the
    repo's examples/OpenArm_O6/.../starvla_train_openarm_o6.yaml."""

    robot_type: str = "openarm_o6_right_arm_hand"
    """Registry key for the OpenArm O6 right-arm/right-hand split."""

    trajs: int = 10
    """Number of trajectories to evaluate."""

    start_traj_id: int = 0
    """First trajectory id to evaluate (for held-out splits)."""

    steps: int = 400
    """Steps per trajectory."""

    action_horizon: int = 16
    """Chunk size used to unroll predictions. Must equal model.action_horizon."""

    denoising_steps: int | None = None
    """Override the model's diffusion denoising steps. None = keep the YAML default."""

    plot: bool = True
    """Save per-DOF prediction plots to <output_dir>/plots/."""

    output_json: Path | None = None
    """Optional path to dump a JSON summary the comparison wrapper consumes."""

    output_dir: Path = REPO_ROOT / "results" / "eval"
    """Where plots + intermediate artifacts go."""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _walk_up_for(start: Path, names: tuple[str, ...], max_levels: int = 4) -> Path | None:
    """Walk up from `start` looking for any of `names`. Returns first match.

    Lets the user point `--checkpoint` at either the run dir, the
    `final_model/` subdir, or a specific `checkpoints/steps_*_pytorch_model.pt`
    file — we discover `config.full.yaml` and `dataset_statistics.json` by
    walking up to the run root in any of those cases.
    """
    cur = start if start.is_dir() else start.parent
    for _ in range(max_levels + 1):
        for name in names:
            cand = cur / name
            if cand.is_file():
                return cand
        if cur.parent == cur:
            return None
        cur = cur.parent
    return None


def _resolve_config_yaml(checkpoint: Path, override: Path | None) -> Path:
    """Find the YAML the checkpoint was trained with."""
    if override is not None:
        return override.resolve()
    found = _walk_up_for(checkpoint, ("config.full.yaml", "config.yaml"))
    if found is not None:
        return found
    fallback = REPO_ROOT / "examples/OpenArm_O6/train_files/starvla_train_openarm_o6.yaml"
    if not fallback.is_file():
        raise FileNotFoundError(f"Could not locate StarVLA config YAML near {checkpoint}")
    return fallback


def _resolve_state_dict(checkpoint: Path) -> Path:
    """Locate the .pt file regardless of whether the user passed a dir or file."""
    if checkpoint.is_file():
        return checkpoint
    for name in ("pytorch_model.pt", "model.safetensors"):
        candidate = checkpoint / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No pytorch_model.pt or model.safetensors found under {checkpoint}"
    )


def _resolve_stats_path(checkpoint: Path) -> Path:
    """Locate dataset_statistics.json (saved by trainer next to checkpoints)."""
    found = _walk_up_for(checkpoint, ("dataset_statistics.json",))
    if found is None:
        raise FileNotFoundError(
            f"dataset_statistics.json not found anywhere up the tree from "
            f"{checkpoint}. This is needed to inverse-normalize StarVLA "
            f"predictions back to physical joint space for an apples-to-apples "
            f"MSE with Isaac-GR00T."
        )
    return found


def _load_q99_stats(stats_path: Path, embodiment_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (q01, q99) for the action vector under the given embodiment key."""
    with stats_path.open() as f:
        stats = json.load(f)
    if embodiment_key not in stats:
        # Fallback: take the first embodiment key — typical for single-robot runs.
        embodiment_key = next(iter(stats))
    action_stats = stats[embodiment_key]["action"]
    q01 = np.asarray(action_stats["q01"], dtype=np.float32)
    q99 = np.asarray(action_stats["q99"], dtype=np.float32)
    return q01, q99


def _q99_inverse(normalized: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Inverse of StarVLA's q99 transform: y = (x + 1) / 2 * (q99 - q01) + q01."""
    return (normalized + 1.0) / 2.0 * (q99 - q01) + q01


def _strip_known_prefixes(state: dict) -> dict:
    """Drop common Accelerate/DeepSpeed/DDP wrapper prefixes from key names."""
    prefixes = ("module.", "_orig_mod.")
    out = {}
    stripped = 0
    for k, v in state.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
                stripped += 1
                break
        out[new_k] = v
    if stripped:
        print(f"[load] stripped wrapper prefix from {stripped} state-dict keys")
    return out


def _scan_for_bad_values(state: dict) -> list[str]:
    """Return a list of parameter names that contain NaN or Inf values.

    A diverged training run can leave a checkpoint full of NaN weights — the
    model loads cleanly, but every forward pass produces NaN. Catching that
    here is much friendlier than letting the eval grind through 10 trajectories
    and report `MSE = nan`.
    """
    bad = []
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.is_floating_point() and not torch.isfinite(v).all():
            bad.append(k)
    return bad


def _load_model(config_yaml: Path, state_dict_path: Path, denoising_steps: int | None,
                device: torch.device) -> Qwen_GR00T:
    """Build the QwenGR00T framework, load weights, and switch to eval."""
    cfg = OmegaConf.load(config_yaml)
    if denoising_steps is not None:
        cfg.framework.action_model.num_inference_timesteps = int(denoising_steps)
    model = Qwen_GR00T(cfg)
    if state_dict_path.suffix == ".pt":
        state = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    else:
        # safetensors fallback
        from safetensors.torch import load_file
        state = load_file(state_dict_path)

    # Some training stacks (Accelerate, DeepSpeed-Zero, DDP) wrap the model and
    # save keys like 'module.qwen_vl_interface....'. Strip those so they line up
    # with the un-wrapped Qwen_GR00T module hierarchy.
    state = _strip_known_prefixes(state)

    bad = _scan_for_bad_values(state)
    if bad:
        print(
            f"[load] WARNING: {len(bad)} checkpoint tensors contain NaN/Inf — "
            f"this almost always means training diverged. First few: {bad[:5]}"
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[load] keys — checkpoint: {len(state)}, "
        f"loaded: {len(state) - len(unexpected)}, "
        f"missing: {len(missing)}, unexpected: {len(unexpected)}"
    )
    if missing:
        # If QwenVL is missing, predictions will be from a randomly-initialized
        # vision-language backbone — guaranteed garbage. Surface the categories.
        qwen_missing = sum(1 for k in missing if k.startswith("qwen_vl_interface"))
        action_missing = sum(1 for k in missing if k.startswith("action_model"))
        print(
            f"[load]   missing categories: qwen_vl_interface={qwen_missing}, "
            f"action_model={action_missing}, other={len(missing) - qwen_missing - action_missing}"
        )
        print(f"[load]   first missing keys: {missing[:5]}")
    if unexpected:
        print(f"[load]   first unexpected keys: {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model


def _build_dataset(robot_type: str, dataset_path: Path, include_state: bool) -> LeRobotSingleDataset:
    """Build the StarVLA dataset object (raw mode, no transforms applied).

    `include_state` must mirror the YAML's `datasets.vla_data.include_state`:
    when True, `_pack_sample` adds the q99-normalized proprioception under
    `sample["state"]`. Models trained with state expect it at inference too —
    dropping it silently makes the action head run without proprioception and
    produces near-untrained-quality predictions.
    """
    if robot_type not in ROBOT_TYPE_CONFIG_MAP:
        raise KeyError(
            f"robot_type '{robot_type}' not in ROBOT_TYPE_CONFIG_MAP. "
            f"Did you forget to import the OpenArm O6 registry?"
        )
    data_cfg = ROBOT_TYPE_CONFIG_MAP[robot_type]
    embodiment = ROBOT_TYPE_TO_EMBODIMENT_TAG.get(robot_type, EmbodimentTag.NEW_EMBODIMENT)
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=data_cfg.modality_config(),
        transforms=data_cfg.transform(),
        embodiment_tag=embodiment,
        video_backend="torchvision_av",
        delete_pause_frame=False,
        data_cfg={"video_backend": "torchvision_av", "include_state": include_state},
    )


# ---------------------------------------------------------------------------
# Per-trajectory evaluation
# ---------------------------------------------------------------------------

def _eval_one_trajectory(
    model: Qwen_GR00T,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    steps: int,
    action_horizon: int,
    q01: np.ndarray,
    q99: np.ndarray,
    state_keys: list[str],
    action_keys: list[str],
    plot_path: Path | None,
) -> float:
    """Return scalar MSE for one trajectory, optionally writing a plot."""
    gt_actions: list[np.ndarray] = []
    pred_actions: list[np.ndarray] = []
    states: list[np.ndarray] = []

    # Index of each action_key inside the concatenated 13-D vector.
    # state_keys / action_keys are ["state.right_arm","state.right_hand"] etc.
    # The dataset's get_step_data returns each key's data as [1, dim_for_key].
    # We rely on the same concat order the registry config uses.

    chunk = None
    # Walk the trajectory in execution-step strides equal to action_horizon
    # (rtc_delay=0, matching eval_policy.py defaults).
    for step in range(steps):
        raw = dataset.get_step_data(traj_id, step)

        gt_concat = np.concatenate(
            [np.atleast_1d(raw[k][0]) for k in action_keys], axis=0
        ).astype(np.float32)
        state_concat = np.concatenate(
            [np.atleast_1d(raw[k][0]) for k in state_keys], axis=0
        ).astype(np.float32)
        gt_actions.append(gt_concat)
        states.append(state_concat)

        if step % action_horizon == 0:
            sample = dataset[traj_id_to_global(dataset, traj_id, step)]
            with torch.no_grad():
                normalized = model.predict_action([sample])["normalized_actions"][0]
            normalized = np.asarray(normalized)

            # Diagnose whether the NaN comes from the model or from the
            # inverse-q99 stats. Print once per trajectory to keep the log
            # readable.
            if step == 0:
                n_nonfinite = int(np.sum(~np.isfinite(normalized)))
                if n_nonfinite > 0:
                    raise RuntimeError(
                        f"traj {traj_id}: model returned {n_nonfinite} non-finite "
                        f"values in normalized_actions (shape={normalized.shape}). "
                        f"This means the checkpoint itself is producing NaN/Inf — "
                        f"check `[load]` lines above for missing weights or "
                        f"corrupted tensors. If the checkpoint diverged during "
                        f"training, retry from an earlier save_interval."
                    )
                print(
                    f"  traj {traj_id} step 0 normalized range: "
                    f"min={normalized.min():.3f}, max={normalized.max():.3f}, "
                    f"mean={normalized.mean():.3f}"
                )

            unnormalized = _q99_inverse(normalized, q01, q99)
            if step == 0 and not np.all(np.isfinite(unnormalized)):
                # Normalized was finite but the inverse isn't → stats are bad.
                raise RuntimeError(
                    f"traj {traj_id}: q99 inverse produced non-finite values "
                    f"despite finite model output. Check that "
                    f"dataset_statistics.json next to your checkpoint has "
                    f"sane q01/q99 entries (no Inf, q99 != q01 except where "
                    f"intentionally constant)."
                )
            chunk = unnormalized

        # Emit one prediction per step from the current chunk.
        local = step % action_horizon
        pred_actions.append(chunk[local])

    gt = np.stack(gt_actions, axis=0)
    pred = np.stack(pred_actions, axis=0)
    states_arr = np.stack(states, axis=0)
    assert gt.shape == pred.shape, f"shape mismatch: {gt.shape} vs {pred.shape}"

    mse = float(np.mean((gt - pred) ** 2))
    print(f"traj {traj_id}: MSE = {mse:.6f}  shape={gt.shape}")

    if plot_path is not None:
        _plot_traj(traj_id, states_arr, gt, pred, action_horizon, plot_path)

    return mse


def traj_id_to_global(dataset: LeRobotSingleDataset, traj_id: int, step: int) -> int:
    """Translate (traj_id, step) into the dataset's global linear index.

    LeRobotSingleDataset is iterable as a flat sequence of frames; the lookup
    requires the trajectory-length prefix sum.
    """
    # `dataset.trajectory_lengths` is exposed on the GR00T fork. If it's not
    # yet populated (lazy init), force one access first.
    _ = dataset[0]  # warm up
    lengths = dataset.trajectory_lengths
    base = int(np.sum(lengths[:traj_id]))
    return base + step


def _plot_traj(traj_id: int, state: np.ndarray, gt: np.ndarray, pred: np.ndarray,
               action_horizon: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dim = gt.shape[1]
    fig, axes = plt.subplots(nrows=dim, ncols=1, figsize=(8, 2 * dim))
    fig.suptitle(f"StarVLA — traj {traj_id}", fontsize=14)
    for i, ax in enumerate(axes):
        ax.plot(state[:, i], label="state", alpha=0.6)
        ax.plot(gt[:, i], label="gt action")
        ax.plot(pred[:, i], label="pred action")
        for j in range(0, gt.shape[0], action_horizon):
            ax.plot(j, gt[j, i], "ro")
        ax.set_title(f"dim {i}")
        ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"  plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: EvalConfig) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yaml_path = _resolve_config_yaml(cfg.checkpoint, cfg.config_yaml)
    state_dict_path = _resolve_state_dict(cfg.checkpoint)
    stats_path = _resolve_stats_path(cfg.checkpoint)

    print(f"== StarVLA trajectory eval ==")
    print(f"checkpoint   : {state_dict_path}")
    print(f"config       : {yaml_path}")
    print(f"stats        : {stats_path}")
    print(f"dataset      : {cfg.dataset_path}")
    print(f"device       : {device}")
    print(f"trajs        : {cfg.trajs} starting at {cfg.start_traj_id}")
    print(f"steps/horiz. : {cfg.steps} / {cfg.action_horizon}")
    print()

    # Resolve state/action key order from the registry — must match the
    # 13-D vector layout the model was trained with.
    data_cfg = ROBOT_TYPE_CONFIG_MAP[cfg.robot_type]
    state_keys: List[str] = list(data_cfg.state_keys)
    action_keys: List[str] = list(data_cfg.action_keys)

    embodiment_key = ROBOT_TYPE_TO_EMBODIMENT_TAG.get(
        cfg.robot_type, EmbodimentTag.NEW_EMBODIMENT
    )
    embodiment_key_name = (
        embodiment_key.value if hasattr(embodiment_key, "value") else str(embodiment_key)
    )
    q01, q99 = _load_q99_stats(stats_path, embodiment_key_name)

    # Must mirror training: if the model was trained with proprioception
    # (datasets.vla_data.include_state: true), the dataset has to add a "state"
    # key to each sample or the model runs without state and produces garbage.
    yaml_cfg = OmegaConf.load(yaml_path)
    include_state = bool(yaml_cfg.datasets.vla_data.get("include_state", False))
    print(f"include_state  : {include_state} (from {yaml_path.name})")

    model = _load_model(yaml_path, state_dict_path, cfg.denoising_steps, device)
    dataset = _build_dataset(cfg.robot_type, cfg.dataset_path, include_state=include_state)

    plot_dir = cfg.output_dir / "plots" if cfg.plot else None

    mses: list[float] = []
    for offset in range(cfg.trajs):
        traj_id = cfg.start_traj_id + offset
        plot_path = (plot_dir / f"traj_{traj_id:03d}.png") if plot_dir else None
        mse = _eval_one_trajectory(
            model=model,
            dataset=dataset,
            traj_id=traj_id,
            steps=cfg.steps,
            action_horizon=cfg.action_horizon,
            q01=q01,
            q99=q99,
            state_keys=state_keys,
            action_keys=action_keys,
            plot_path=plot_path,
        )
        mses.append(mse)

    avg = float(np.mean(mses))
    print()
    print(f"Average MSE across {cfg.trajs} trajs: {avg:.6f}")

    if cfg.output_json is not None:
        cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": "starvla",
            "checkpoint": str(state_dict_path),
            "dataset": str(cfg.dataset_path),
            "trajectories": [
                {"id": cfg.start_traj_id + i, "mse": mses[i]} for i in range(cfg.trajs)
            ],
            "average_mse": avg,
            "trajs": cfg.trajs,
            "steps": cfg.steps,
            "action_horizon": cfg.action_horizon,
        }
        with cfg.output_json.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"summary  → {cfg.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(EvalConfig, description=__doc__)))
