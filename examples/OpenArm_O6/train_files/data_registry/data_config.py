from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import VideoResize, VideoToNumpy, VideoToTensor


class OpenArmO6RightArmHandDataConfig:
    """StarVLA dataloader mapping for the OpenArm O6 can-sorting dataset.

    The raw dataset stores a 26-D bimanual vector:
        left arm [0:7], right arm [7:14], left hand [14:20], right hand [20:26].

    In the provided demonstrations, only the right 7-DOF arm and 6-DOA
    LinkerHand O6 are used. Therefore this config exposes only:
        state/action.right_arm + state/action.right_hand = 13 dimensions.
    """

    # These keys must match the friendly names in meta/modality.json. The
    # underlying raw columns are still observation.state, action, and
    # observation.images.camera.
    video_keys = ["video.camera"]
    state_keys = ["state.right_arm", "state.right_hand"]
    action_keys = ["action.right_arm", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]

    # One current observation frame and a 16-step future action chunk. This
    # action horizon must match framework.action_model.action_horizon in YAML.
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        # ModalityConfig tells LeRobotSingleDataset which named fields to fetch
        # and which temporal offsets to sample for each field.
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # Keep the single camera under its original `video.camera` key. The
        # current StarVLA _pack_sample path expects per-camera keys, so we do
        # not use ConcatTransform for video/state/action here.
        transforms = [
            # Video enters as numpy frames. StarVLA's video resize transform
            # expects tensor layout, then we convert back to numpy for PIL.
            VideoToTensor(apply_to=self.video_keys),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoToNumpy(apply_to=self.video_keys),

            # Normalize proprioception using robust q01/q99 statistics. This
            # is usually safer for robot joints than min/max when outliers exist.
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.right_arm": "q99",
                    "state.right_hand": "q99",
                },
            ),

            # Actions use the same right-side split and q99 normalization. The
            # dataset's metadata marks these as absolute joint positions.
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.right_arm": "q99",
                    "action.right_hand": "q99",
                },
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


ROBOT_TYPE_CONFIG_MAP = {
    # `robot_type` used by the mixture below. The central registry auto-loads
    # examples/*/train_files/data_registry/data_config.py at import time.
    "openarm_o6_right_arm_hand": OpenArmO6RightArmHandDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # NEW_EMBODIMENT is the safe default for custom robots not already known by
    # StarVLA/GR00T's built-in embodiment tag list.
    "openarm_o6_right_arm_hand": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    # The YAML references this name as datasets.vla_data.data_mix.
    # Tuple format: (dataset folder under data_root_dir, sampling weight, robot_type).
    "openarm_o6_cansort_right_only": [
        ("OpenArm_O6_CanSorting_dataset_0408", 1.0, "openarm_o6_right_arm_hand"),
    ],
}
