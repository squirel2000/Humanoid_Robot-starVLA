#!/usr/bin/env python3
"""Compare a fine-tuned StarVLA checkpoint against an Isaac-GR00T N1.5
checkpoint on the OpenArm O6 can-sorting dataset.

Why two conda envs:
    StarVLA pins different transformers / accelerate / flash-attn versions
    than Isaac-GR00T, so they cannot coexist in one env. This script shells
    out to each side under its own conda env, parses the per-trajectory and
    average MSE numbers, and writes a side-by-side summary.

Usage:
    conda activate starVLA      # any env that has python is fine
    python /home/asus/Gits/IsaacLab-GR00T/starVLA/scripts/5_compare_starvla_vs_n15.py \
        --starvla-ckpt /home/asus/Gits/IsaacLab-GR00T/starVLA/results/Checkpoints/openarm_o6_qwengroot_right_only_bs16_lr5e5_wd1e5/final_model \
        --n15-ckpt /home/asus/Gits/IsaacLab-GR00T/Isaac-GR00T/outputs/openarm_linkerhando6_cansorting_N15_fft_100k_dataset_0408/checkpoint-100000 \
        --trajs 10 \
        --steps 150
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = Path(
    "/home/asus/Gits/IsaacLab-GR00T/IsaacLab/datasets/gr00t_collection/"
    "OpenArm_O6_CanSorting_dataset_0408"
)
DEFAULT_N15_CKPT = Path(
    "/home/asus/Gits/IsaacLab-GR00T/Isaac-GR00T/outputs/"
    "openarm_linkerhando6_cansorting_N15_fft_100k_dataset_0408/checkpoint-100000"
)
DEFAULT_N15_REPO = Path("/home/asus/Gits/IsaacLab-GR00T/Isaac-GR00T")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CompareConfig:
    """CLI surface, mirroring the other 1_/2_/3_/4_ scripts in this repo."""

    starvla_ckpt: Path
    """Path to a StarVLA fine-tune output directory (run dir or final_model)."""

    n15_ckpt: Path = DEFAULT_N15_CKPT
    """Path to the Isaac-GR00T N1.5 checkpoint directory."""

    dataset: Path = DEFAULT_DATASET
    """Path to the OpenArm O6 LeRobot dataset."""

    n15_repo: Path = DEFAULT_N15_REPO
    """Path to the Isaac-GR00T checkout (for scripts/eval_policy.py)."""

    trajs: int = 10
    """Number of trajectories to evaluate."""

    steps: int = 400
    """Steps per trajectory."""

    start_traj_id: int = 0
    """First trajectory id to evaluate (for held-out splits)."""

    action_horizon: int = 16
    """Chunk size used to unroll predictions."""

    denoising_steps: int | None = None
    """Override num_inference_timesteps on the StarVLA side only. The action
    head's saved default is 4 (Euler), which is fast but noisy; 16–20 gives
    cleaner trajectories and a fair head-to-head with N1.5's DDIM-style
    sampler. None = keep whatever the StarVLA YAML default says."""

    plot: bool = True
    """Save per-DOF plots from both evaluators."""

    starvla_env: str = "starVLA"
    """Conda env that has the StarVLA package installed."""

    gr00t_env: str = "env_gr00t"
    """Conda env that has Isaac-GR00T installed."""

    skip_starvla: bool = False
    """Reuse a previously-written starvla_eval.json under --output-dir."""

    skip_n15: bool = False
    """Reuse a previously-written n15_eval.log under --output-dir."""

    output_dir: Path | None = None
    """Write logs/JSON here. Defaults to results/comparison/<timestamp>."""


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def conda_bash_cmd(env: str, command: str) -> list[str]:
    """Build a bash invocation that activates a conda env first.

    `bash -lc <string>` runs <string> in a fresh login shell. We need a login
    shell because conda's activation hook is registered in ~/.bashrc /
    ~/.bash_profile (via `conda init`), and a non-login bash skips those.
    Then we explicitly source the conda hook on top of that for safety, since
    not every machine puts the hook in .bashrc.

    We avoid `conda run -n <env> ...` because it buffers child stdout, which
    breaks the live tee of training/eval progress.
    """
    conda_base = subprocess.check_output(
        ["conda", "info", "--base"], text=True
    ).strip()
    bootstrap = (
        f"source {shlex.quote(conda_base)}/etc/profile.d/conda.sh && "
        f"conda activate {shlex.quote(env)} && "
        # Mute torchvision's video deprecation warning at the child level so
        # the comparison log isn't drowned in identical UserWarnings.
        f"export PYTHONWARNINGS='ignore:The video decoding and encoding "
        f"capabilities of torchvision are deprecated:UserWarning' && "
        f"{command}"
    )
    return ["bash", "-lc", bootstrap]


def stream_to_file(cmd: list[str], log_path: Path, cwd: Path | None = None) -> int:
    """Run `cmd`, mirror stdout/stderr to console and to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"+ {' '.join(shlex.quote(p) for p in cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        rc = proc.wait()
    return rc


# ---------------------------------------------------------------------------
# Eval drivers
# ---------------------------------------------------------------------------

def run_n15_eval(cfg: CompareConfig, out_dir: Path) -> Path:
    """Run Isaac-GR00T's eval_policy.py against the N1.5 checkpoint."""
    log = out_dir / "n15_eval.log"
    if cfg.skip_n15 and log.is_file():
        print(f"[1/2] skipping N1.5 eval, reusing {log}")
        return log

    plot_flag = "--plot" if cfg.plot else ""
    inner = (
        f"python scripts/eval_policy.py {plot_flag} "
        f"--model-path {shlex.quote(str(cfg.n15_ckpt))} "
        f"--dataset-path {shlex.quote(str(cfg.dataset))} "
        f"--data-config openarm_linkerhand_o6 "
        f"--embodiment-tag new_embodiment "
        f"--modality-keys right_arm right_hand "
        f"--trajs {cfg.trajs} "
        f"--steps {cfg.steps} "
        f"--action-horizon {cfg.action_horizon}"
    ).strip()
    cmd = conda_bash_cmd(cfg.gr00t_env, inner)
    print(f"[1/2] Evaluating Isaac-GR00T N1.5 in env={cfg.gr00t_env} …", flush=True)
    rc = stream_to_file(cmd, log, cwd=cfg.n15_repo)
    if rc != 0:
        raise SystemExit(f"N1.5 eval failed with code {rc}")
    return log


def run_starvla_eval(cfg: CompareConfig, out_dir: Path) -> tuple[Path, Path]:
    """Run StarVLA's per-trajectory MSE eval."""
    log = out_dir / "starvla_eval.log"
    json_path = out_dir / "starvla_eval.json"
    if cfg.skip_starvla and json_path.is_file():
        print(f"[2/2] skipping StarVLA eval, reusing {json_path}")
        return log, json_path

    plot_flag = "--plot" if cfg.plot else ""
    ds_flag = (
        f"--denoising_steps {cfg.denoising_steps} "
        if cfg.denoising_steps is not None else ""
    )
    inner = (
        f"python scripts/4_eval_starvla_traj.py "
        f"--checkpoint {shlex.quote(str(cfg.starvla_ckpt))} "
        f"--dataset_path {shlex.quote(str(cfg.dataset))} "
        f"--trajs {cfg.trajs} "
        f"--start_traj_id {cfg.start_traj_id} "
        f"--steps {cfg.steps} "
        f"--action_horizon {cfg.action_horizon} "
        f"{ds_flag}"
        f"--output_json {shlex.quote(str(json_path))} "
        f"--output_dir {shlex.quote(str(out_dir))} "
        f"{plot_flag}"
    ).strip()
    cmd = conda_bash_cmd(cfg.starvla_env, inner)
    print(f"\n[2/2] Evaluating StarVLA in env={cfg.starvla_env} …", flush=True)
    rc = stream_to_file(cmd, log, cwd=REPO_ROOT)
    if rc != 0:
        raise SystemExit(f"StarVLA eval failed with code {rc}")
    return log, json_path


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

_AVG_RE = re.compile(r"Average MSE across .*?:\s*([0-9.eE+\-]+)")
_TRAJ_RE = re.compile(r"^MSE:\s*([0-9.eE+\-]+)\s*$")


def parse_n15_log(log_path: Path) -> tuple[list[float], float | None]:
    """Pull per-traj MSE and the final average from eval_policy.py output."""
    per_traj: list[float] = []
    avg: float | None = None
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        m = _TRAJ_RE.match(line)
        if m:
            per_traj.append(float(m.group(1)))
            continue
        m = _AVG_RE.search(line)
        if m:
            avg = float(m.group(1))
    return per_traj, avg


def load_starvla_json(json_path: Path) -> tuple[list[float], float]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    per_traj = [float(t["mse"]) for t in payload["trajectories"]]
    return per_traj, float(payload["average_mse"])


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

def render_summary(
    cfg: CompareConfig,
    out_dir: Path,
    starvla_per_traj: list[float],
    starvla_avg: float,
    n15_per_traj: list[float],
    n15_avg: float | None,
) -> Path:
    summary = out_dir / "summary.md"
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# StarVLA vs Isaac-GR00T N1.5 — {ts}",
        "",
        f"Dataset: `{cfg.dataset}`",
        f"Trajectories: {cfg.trajs} starting at {cfg.start_traj_id}, "
        f"steps/traj: {cfg.steps}, action_horizon: {cfg.action_horizon}",
        "",
        "## Average MSE (lower = better)",
        "",
        "| Model | Checkpoint | Avg MSE |",
        "| --- | --- | --- |",
        f"| StarVLA (Qwen3-VL) | `{cfg.starvla_ckpt}` | "
        f"{starvla_avg:.6f} |",
        f"| Isaac-GR00T N1.5 (Eagle 2) | `{cfg.n15_ckpt}` | "
        f"{n15_avg:.6f} |" if n15_avg is not None else
        f"| Isaac-GR00T N1.5 (Eagle 2) | `{cfg.n15_ckpt}` | "
        f"<not parsed — check log> |",
    ]

    if starvla_per_traj and n15_per_traj and len(starvla_per_traj) == len(n15_per_traj):
        lines += [
            "",
            "## Per-trajectory MSE",
            "",
            "| Traj | StarVLA | N1.5 | Δ (StarVLA − N1.5) |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for i, (sv, n15) in enumerate(zip(starvla_per_traj, n15_per_traj)):
            lines.append(
                f"| {cfg.start_traj_id + i} | "
                f"{sv:.6f} | {n15:.6f} | {sv - n15:+.6f} |"
            )

    lines += [
        "",
        f"Logs: `{out_dir / 'n15_eval.log'}`, `{out_dir / 'starvla_eval.log'}`",
        f"StarVLA structured output: `{out_dir / 'starvla_eval.json'}`",
    ]

    body = "\n".join(lines) + "\n"
    summary.write_text(body, encoding="utf-8")
    print()
    print(body)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: CompareConfig) -> int:
    if not cfg.starvla_ckpt.exists():
        sys.exit(f"ERROR: --starvla-ckpt does not exist: {cfg.starvla_ckpt}")
    if not cfg.skip_n15 and not cfg.n15_ckpt.is_dir():
        sys.exit(f"ERROR: --n15-ckpt dir does not exist: {cfg.n15_ckpt}")
    if not cfg.dataset.is_dir():
        sys.exit(f"ERROR: dataset dir does not exist: {cfg.dataset}")
    if not cfg.skip_n15 and not (cfg.n15_repo / "scripts/eval_policy.py").is_file():
        sys.exit(f"ERROR: eval_policy.py missing under --n15-repo: {cfg.n15_repo}")

    out_dir = (
        cfg.output_dir
        if cfg.output_dir is not None
        else REPO_ROOT / "results/comparison" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Banner
    sep = "=" * 72
    print(sep)
    print(" StarVLA vs Isaac-GR00T N1.5 — OpenArm O6 can sorting")
    print(sep)
    for label, val in [
        ("starvla ckpt", cfg.starvla_ckpt),
        ("n1.5 ckpt", cfg.n15_ckpt),
        ("dataset", cfg.dataset),
        ("trajs", f"{cfg.trajs} starting at {cfg.start_traj_id}"),
        ("steps", cfg.steps),
        ("action horiz.", cfg.action_horizon),
        ("output dir", out_dir),
    ]:
        print(f"  {label:13s}: {val}")
    print(sep)
    print()

    # Run both
    n15_log = run_n15_eval(cfg, out_dir)
    _, starvla_json = run_starvla_eval(cfg, out_dir)

    # Aggregate
    n15_per_traj, n15_avg = parse_n15_log(n15_log)
    starvla_per_traj, starvla_avg = load_starvla_json(starvla_json)

    summary = render_summary(
        cfg, out_dir,
        starvla_per_traj=starvla_per_traj, starvla_avg=starvla_avg,
        n15_per_traj=n15_per_traj, n15_avg=n15_avg,
    )
    print(f"Wrote: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(CompareConfig, description=__doc__)))
