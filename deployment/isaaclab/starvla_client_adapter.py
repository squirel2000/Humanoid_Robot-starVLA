"""StarVLA client adapter for IsaacLab.

Drop-in replacement for `IsaacLab/scripts/gr00t_script/utils/gr00t_client_adapter.py`
when you want to drive an IsaacLab simulation with a fine-tuned **StarVLA**
policy instead of an Isaac-GR00T N1.5 policy.

How this differs from `Gr00tClientAdapter`
------------------------------------------
* **Transport.** Isaac-GR00T's server uses ZeroMQ. StarVLA's server
  (`deployment/model_server/server_policy.py`) uses **WebSocket**
  (msgpack-numpy framed). This adapter speaks the WebSocket protocol via
  `WebsocketClientPolicy` from `deployment/model_server/tools/`.
* **Wire format.** GR00T sends the LeRobot-style observation dict
  (`video.camera`, `state.<key>`, `annotation.human.task_description`) and
  receives an action dict (`action.<key>`). StarVLA's
  `framework.predict_action` expects a flat list of `{image, lang, state}`
  examples and returns `{normalized_actions: (B, T, action_dim)}`. This
  adapter does the conversion in both directions.
* **Normalization.** StarVLA's framework returns *normalized* actions
  (q99 by default for OpenArm O6). The IsaacLab agent immediately writes
  the action to joint controllers, so we have to inverse-transform on the
  client side using `dataset_statistics.json` saved at training time.

Public API matches `Gr00tClientAdapter`
---------------------------------------
* `__init__(version="starvla", host=..., port=10093, stats_path=..., action_split=...)`
* `get_action(obs) -> dict` returning `{f"action.{k}": (T, dim_k)}`

Install
-------
1. Copy this file to
   `~/Gits/IsaacLab-GR00T/IsaacLab/scripts/gr00t_script/utils/starvla_client_adapter.py`
2. Make sure the `websockets` and `msgpack` Python packages are installed
   in the env that runs `gr00t_infer_agent.py` (usually `env_isaaclab`):

       pip install websockets "msgpack>=1.0" msgpack-numpy

3. Patch `gr00t_infer_agent.py` to choose between adapters via a CLI flag.
   See `STARVLA_INFERENCE_ISAACLAB.md` for the patch.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Locate StarVLA's WebSocket client.
# This module lives inside the StarVLA repo by default; if you are running
# this adapter from the IsaacLab `env_isaaclab` env, point STARVLA_REPO at
# the StarVLA checkout (or copy `deployment/` into your PYTHONPATH).
# ---------------------------------------------------------------------------

_DEFAULT_STARVLA_REPO = Path("/home/asus/Gits/humanoid_robot/starVLA")


def _import_websocket_client(repo_root: Path):
    """Import StarVLA's WebsocketClientPolicy without installing the package."""
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from deployment.model_server.tools.websocket_policy_client import (  # noqa: E402
        WebsocketClientPolicy,
    )
    return WebsocketClientPolicy


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

DEFAULT_ACTION_SPLIT: tuple[tuple[str, int], ...] = (
    ("right_arm", 7),
    ("right_hand", 6),
)


class StarVLAClientAdapter:
    """Speaks StarVLA's WebSocket protocol; quacks like Gr00tClientAdapter.

    Parameters
    ----------
    host, port:
        Where the StarVLA server (deployment/model_server/server_policy.py)
        is listening. Default port is 10093, matching server defaults.
    stats_path:
        Path to the `dataset_statistics.json` saved next to the StarVLA
        checkpoint at training time (e.g.
        `results/Checkpoints/<run_id>/dataset_statistics.json`). Required
        because the WebSocket server returns NORMALIZED actions and the
        IsaacLab agent expects unnormalized joint targets.
    embodiment_key:
        Top-level key inside `dataset_statistics.json` (defaults to the
        first key, which is `new_embodiment` for OpenArm O6 runs).
    action_split:
        Sequence of `(name, dim)` pairs that says how to split the flat
        action vector returned by StarVLA into per-modality keys the
        IsaacLab `JointMapper` consumes. The OpenArm O6 right-only default
        is `[("right_arm", 7), ("right_hand", 6)]`, matching the StarVLA
        registry and Isaac-GR00T's `OpenArmLinkerHandO6DataConfig`.
    starvla_repo:
        Path to the StarVLA checkout. Used only to import the WebSocket
        client module — we do NOT load the model here.
    """

    def __init__(
        self,
        version: str = "starvla",
        host: str = "127.0.0.1",
        port: int = 10093,
        stats_path: str | Path | None = None,
        embodiment_key: str | None = None,
        action_split: Sequence[tuple[str, int]] = DEFAULT_ACTION_SPLIT,
        starvla_repo: str | Path = _DEFAULT_STARVLA_REPO,
    ) -> None:
        self.version = version
        self._action_split = list(action_split)
        action_dim = sum(d for _, d in self._action_split)

        if stats_path is None:
            raise ValueError(
                "StarVLAClientAdapter needs --stats-path (the "
                "dataset_statistics.json beside your StarVLA checkpoint) "
                "to inverse-normalize actions before they reach the joint "
                "controllers."
            )

        WebsocketClientPolicy = _import_websocket_client(Path(starvla_repo))
        self._client = WebsocketClientPolicy(host=host, port=port)
        self._metadata = self._client.get_server_metadata()

        # --- Load q99 statistics for inverse transform ---
        with Path(stats_path).open("r", encoding="utf-8") as f:
            stats = json.load(f)
        if embodiment_key is None:
            embodiment_key = next(iter(stats))
        action_stats = stats[embodiment_key]["action"]
        self._q01 = np.asarray(action_stats["q01"], dtype=np.float32)
        self._q99 = np.asarray(action_stats["q99"], dtype=np.float32)
        if self._q01.shape[0] != action_dim or self._q99.shape[0] != action_dim:
            raise ValueError(
                f"action_split totals {action_dim} dims but stats have "
                f"{self._q01.shape[0]}. Did you point at the right "
                f"dataset_statistics.json?"
            )

        logging.info(
            "[StarVLAClientAdapter] connected to %s:%d, server metadata=%s, "
            "action_split=%s",
            host, port, self._metadata, self._action_split,
        )

    # ------------------------------------------------------------------
    # Compatibility surface used by the IsaacLab agent
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """StarVLA's WebSocket server has no explicit ping op exposed on
        WebsocketClientPolicy, but the connection is established eagerly in
        `_wait_for_server`, so reaching this point means the link is up."""
        return self._client._ws is not None  # type: ignore[attr-defined]

    def get_modality_config(self) -> dict[str, Any]:
        """The IsaacLab agent only logs this; we just report a static schema
        that matches the OpenArm O6 right-only setup."""
        return {
            "video": ["video.camera"],
            "state": [f"state.{name}" for name, _ in self._action_split],
            "action": [f"action.{name}" for name, _ in self._action_split],
            "language": ["annotation.human.task_description"],
        }

    def get_action(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Translate IsaacLab's GR00T-style obs → StarVLA examples → action."""
        sample = self._build_starvla_example(obs)

        # StarVLA framework signature is `predict_action(examples=[...])`.
        # The WebSocket router unpacks the payload as kwargs to predict_action.
        response = self._client.predict_action({"examples": [sample]})
        if not isinstance(response, dict) or response.get("ok") is False:
            err = response.get("error") if isinstance(response, dict) else response
            raise RuntimeError(f"StarVLA inference error: {err}")
        data = response["data"]
        normalized = np.asarray(data["normalized_actions"])  # (1, T, D)
        if normalized.ndim != 3:
            raise RuntimeError(
                f"Expected normalized_actions of shape (B, T, D); got {normalized.shape}"
            )
        unnorm = self._inverse_q99(normalized[0])           # (T, D)

        # Split (T, D) into the per-modality keys IsaacLab's joint_mapper expects.
        out: dict[str, np.ndarray] = {}
        cursor = 0
        for name, dim in self._action_split:
            out[f"action.{name}"] = unnorm[:, cursor : cursor + dim].astype(np.float32)
            cursor += dim
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_starvla_example(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Convert IsaacLab obs dict to StarVLA's `predict_action` example."""
        # 1) Image: GR00T-side passes (1, H, W, C) uint8. StarVLA's Qwen
        #    interfaces expect one list of camera views per example, so a
        #    single IsaacLab camera becomes [HxWxC].
        img = obs["video.camera"]
        if isinstance(img, np.ndarray) and img.ndim == 4:
            img = img[0]
        img = [img]

        # 2) Language: IsaacLab passes a list[str] under this key; StarVLA
        #    expects a single string per example.
        lang = obs.get("annotation.human.task_description")
        if isinstance(lang, (list, tuple)):
            lang = lang[0]

        # 3) State: concat per-key state vectors in the same order as
        #    action_split (right_arm + right_hand). The StarVLA framework
        #    expects shape (1, state_dim).
        state_parts = []
        for name, dim in self._action_split:
            key = f"state.{name}"
            if key not in obs:
                raise KeyError(
                    f"obs missing '{key}'. Did the JointMapper produce it? "
                    f"Existing keys: {sorted(obs.keys())}"
                )
            v = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            if v.shape[0] != dim:
                raise ValueError(
                    f"obs['{key}'] has shape {v.shape}, expected ({dim},)"
                )
            state_parts.append(v)
        state = np.concatenate(state_parts, axis=0)[None, :]  # (1, state_dim)

        return {"image": img, "lang": lang, "state": state}

    def _inverse_q99(self, normalized: np.ndarray) -> np.ndarray:
        """StarVLA's q99 inverse: y = (x + 1) / 2 * (q99 - q01) + q01."""
        return (normalized + 1.0) / 2.0 * (self._q99 - self._q01) + self._q01
