from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nedm.training.dataset import load_metadata
from nedm.training.model import HMMWVDynamicsModel


@dataclass(frozen=True)
class FrozenDynamics:
    model: HMMWVDynamicsModel
    metadata: dict[str, Any]
    config: dict[str, Any]
    checkpoint_path: Path
    context_steps: int
    dt_s: float


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }


def load_frozen_dynamics(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    processed_dataset_dir: str | Path | None = None,
) -> FrozenDynamics:
    """Load a trained HMMWV dynamics checkpoint for batched inference."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    if processed_dataset_dir is not None:
        config = dict(config)
        config["processed_dataset_dir"] = str(Path(processed_dataset_dir).expanduser().resolve())

    metadata = checkpoint.get("metadata")
    if metadata is None:
        metadata = load_metadata(Path(config["processed_dataset_dir"]).expanduser().resolve())

    model = HMMWVDynamicsModel(
        state_dim=len(metadata["state_fields"]),
        action_dim=len(metadata["action_fields"]),
        target_dim=len(metadata["state_fields"]),
        transformer_cfg=config["model"],
        normalization=metadata["normalization"],
    )
    model.load_state_dict(_strip_compile_prefix(checkpoint["model_state_dict"]))
    model.to(torch.device(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return FrozenDynamics(
        model=model,
        metadata=metadata,
        config=config,
        checkpoint_path=checkpoint_path,
        context_steps=int(config["model"]["block_size"]),
        dt_s=float(metadata["dt_s"]),
    )

