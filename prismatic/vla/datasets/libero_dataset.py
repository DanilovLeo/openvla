"""
libero_dataset.py

Map-style PyTorch Dataset that loads LIBERO HDF5 demonstrations, applies a
KeyframeExtractor to each demo, and returns samples in the format expected by
the OpenVLA PaddedCollatorForActionPrediction / fine-tuning loop.

Each item in the dataset is one (image, action, instruction) tuple at a
selected keyframe index.  Actions are normalized to [-1, 1] per-dimension
using the q01/q99 statistics computed across the entire dataset at init time.

Usage example
-------------
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets.libero_dataset import LiberoKeyframeDataset
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder

    dataset = LiberoKeyframeDataset(
        hdf5_paths=["path/to/task_demo.hdf5"],
        extractor=UniformExtractor(n_keyframes=10),
        traj_key="ee_pos",
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    # dataset.dataset_statistics — pass to save_dataset_statistics()
"""

import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# Allow importing from keyframe-selector/src when running from the openvla tree.
# Callers are expected to have the keyframe-selector src on sys.path already;
# if not, this optional path insert covers the default repo layout.
_kf_src = Path(__file__).parent.parent.parent.parent.parent / "keyframe-selector" / "src"
if _kf_src.exists() and str(_kf_src) not in sys.path:
    sys.path.insert(0, str(_kf_src))

from extractors.base import KeyframeExtractor  # noqa: E402
from utils.loader import list_demos, load_libero_demo  # noqa: E402

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer

IGNORE_INDEX = -100


class LiberoKeyframeDataset(Dataset):
    """Keyframe-selected LIBERO demonstrations as an OpenVLA-compatible Dataset.

    Pre-loads all keyframe images and actions into memory at construction time
    (safe for LIBERO scale: ~50 demos × ~15 keyframes × 128×128×3 ≈ 50 MB/task).

    Actions are normalized to [-1, 1] using per-dimension q01/q99 computed from
    the full dataset.  The resulting ``dataset_statistics`` attribute mirrors the
    format used by the RLDS pipeline and should be passed to
    ``save_dataset_statistics()`` so the checkpoint includes de-normalization info.

    Args:
        hdf5_paths:        List of LIBERO HDF5 file paths to include.
        extractor:         Keyframe extractor instance.
        traj_key:          Demo field passed to extractor.extract().
                           Defaults to "ee_pos".  Use "ee_vel" for
                           VelocityZeroExtractor, "gripper_state" for
                           GripperStateExtractor.
        action_tokenizer:  OpenVLA ActionTokenizer.
        base_tokenizer:    HuggingFace tokenizer.
        image_transform:   Callable that converts a PIL Image to a tensor.
        prompt_builder_fn: PromptBuilder class (e.g. PurePromptBuilder).
        predict_stop_token: Whether to include the stop token in the loss.
        dataset_name:      Name stored in dataset_statistics and returned per
                           sample for W&B logging.
    """

    def __init__(
        self,
        hdf5_paths: List[str],
        extractor: KeyframeExtractor,
        traj_key: str = "ee_pos",
        action_tokenizer: Optional[ActionTokenizer] = None,
        base_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        image_transform: Optional[Callable] = None,
        prompt_builder_fn: Optional[Type[PromptBuilder]] = None,
        predict_stop_token: bool = True,
        dataset_name: str = "libero_keyframe",
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.dataset_name = dataset_name

        # --- collect raw (image_uint8, action_raw, task_name) tuples ---
        raw_images: List[np.ndarray] = []   # (H, W, 3) uint8
        raw_actions: List[np.ndarray] = []  # (7,)
        task_names: List[str] = []

        for path in hdf5_paths:
            demo_keys = list_demos(path)
            for demo_idx in range(len(demo_keys)):
                demo = load_libero_demo(path, demo_idx=demo_idx)
                traj = demo[traj_key]
                kf_indices = extractor.extract(traj)
                task_name = demo["task_name"]
                images_all = demo["images"]   # (T, H, W, 3) uint8
                actions_all = demo["actions"] # (T, 7)

                for idx in kf_indices:
                    raw_images.append(images_all[idx])
                    raw_actions.append(actions_all[idx].astype(np.float32))
                    task_names.append(task_name)

        if not raw_images:
            raise ValueError("No keyframes found — check hdf5_paths and extractor.")

        self._task_names = task_names
        self._raw_images = raw_images

        # --- compute per-dimension q01/q99 and normalize actions to [-1, 1] ---
        actions_matrix = np.stack(raw_actions, axis=0)  # (N, 7)
        q01 = np.percentile(actions_matrix, 1,  axis=0).astype(np.float32)
        q99 = np.percentile(actions_matrix, 99, axis=0).astype(np.float32)

        # Avoid divide-by-zero for constant dimensions (e.g. gripper already ±1)
        denom = np.where(np.abs(q99 - q01) < 1e-8, 1.0, q99 - q01).astype(np.float32)
        self._normalized_actions: List[np.ndarray] = [
            np.clip(2.0 * (a - q01) / denom - 1.0, -1.0, 1.0)
            for a in raw_actions
        ]

        self.dataset_statistics = {
            dataset_name: {
                "action": {
                    "q01": q01,
                    "q99": q99,
                },
                "num_trajectories": _count_trajectories(hdf5_paths),
                "num_transitions": len(raw_images),
            }
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._raw_images)

    def __getitem__(self, idx: int) -> dict:
        image_np = self._raw_images[idx]
        action = self._normalized_actions[idx]
        task_name = self._task_names[idx]

        img = Image.fromarray(image_np)

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {task_name}?"},
            {"from": "gpt",   "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        labels = list(input_ids)

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # Only compute loss on the action tokens
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=self.dataset_name,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        n = len(self)
        n_tasks = len(set(self._task_names))
        return (
            f"LiberoKeyframeDataset: {n} keyframes across {n_tasks} task(s) "
            f"from {len(set(t for t in self._task_names))} unique instructions"
        )


def _count_trajectories(hdf5_paths: List[str]) -> int:
    total = 0
    for path in hdf5_paths:
        total += len(list_demos(path))
    return total


# ---------------------------------------------------------------------------
# Factory: build a LiberoKeyframeDataset from a string extractor spec
# ---------------------------------------------------------------------------

_TRAJ_KEY = {
    "uniform":        "ee_pos",
    "velocity_zero":  "ee_vel",
    "gripper_state":  "gripper_state",
    "awe":            "ee_pos",
}


def build_extractor(
    name: str,
    uniform_n: int = 10,
    velocity_percentile: float = 25.0,
    velocity_min_dist: int = 5,
    gripper_min_dist: int = 5,
    awe_eps: float = 0.01,
) -> Tuple[KeyframeExtractor, str]:
    """Return (extractor_instance, traj_key) for the given extractor name.

    name must be one of: "uniform", "velocity_zero", "gripper_state", "awe".
    """
    from extractors import (  # noqa: F401
        AWEExtractor,
        GripperStateExtractor,
        UniformExtractor,
        VelocityZeroExtractor,
    )

    if name == "uniform":
        ext = UniformExtractor(n_keyframes=uniform_n)
    elif name == "velocity_zero":
        ext = VelocityZeroExtractor(percentile=velocity_percentile, min_dist=velocity_min_dist)
    elif name == "gripper_state":
        ext = GripperStateExtractor(min_dist=gripper_min_dist)
    elif name == "awe":
        ext = AWEExtractor(error_threshold=awe_eps)
    else:
        raise ValueError(f"Unknown extractor '{name}'. Choose: uniform | velocity_zero | gripper_state | awe")

    return ext, _TRAJ_KEY[name]
