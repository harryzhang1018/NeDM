from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler

from nedm.training.dataset import WindowedHMMWVDataset, load_metadata, load_rollout_split
from nedm.training.model import HMMWVDynamicsModel


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a HMMWV sequence model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hmmwv_transformer_v1.json"),
        help="Training config JSON file.",
    )
    parser.add_argument(
        "--processed-dataset-dir",
        type=Path,
        default=None,
        help="Optional override for the processed dataset directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional override for run output directory.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Optional override for training epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Optional override for train steps per epoch.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional override for validation batches.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional override for batch size.")
    parser.add_argument("--device", type=str, default=None, help="Optional override for device.")
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=None,
        help="Optional cap on training windows for quick smoke tests.",
    )
    parser.add_argument(
        "--max-val-windows",
        type=int,
        default=None,
        help="Optional cap on validation windows for quick smoke tests.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume model/optimizer/global_step from a previous checkpoint.",
    )
    return parser.parse_args(argv)


def merge_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = json.loads(json.dumps(config))
    if args.processed_dataset_dir is not None:
        merged["processed_dataset_dir"] = str(args.processed_dataset_dir)
    if args.output_dir is not None:
        merged["output_dir"] = str(args.output_dir)
    if args.num_epochs is not None:
        merged["training"]["num_epochs"] = int(args.num_epochs)
    if args.steps_per_epoch is not None:
        merged["training"]["steps_per_epoch"] = int(args.steps_per_epoch)
    if args.max_val_batches is not None:
        merged["training"]["max_val_batches"] = int(args.max_val_batches)
    if args.batch_size is not None:
        merged["training"]["batch_size"] = int(args.batch_size)
    if args.device is not None:
        merged["training"]["device"] = args.device
    if args.max_train_windows is not None:
        merged["training"]["max_train_windows"] = int(args.max_train_windows)
    if args.max_val_windows is not None:
        merged["training"]["max_val_windows"] = int(args.max_val_windows)
    if args.resume_from_checkpoint is not None:
        merged["training"]["resume_from_checkpoint"] = str(args.resume_from_checkpoint)
    return merged


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def infinite_loader(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def mixed_infinite_loader(
    loaders: list[DataLoader], terrain_ids: list[int] | None = None
) -> Iterator[dict[str, torch.Tensor]]:
    iterators = [infinite_loader(loader) for loader in loaders]
    while True:
        batches = [next(iterator) for iterator in iterators]
        # Tag every sub-batch with its source terrain id (the label is free: each
        # train_mix source is one terrain). Done before the merge so the concatenated
        # batch carries a per-sample terrain id aligned with its rows.
        if terrain_ids is not None:
            for batch, terrain_id in zip(batches, terrain_ids, strict=True):
                batch["terrain_ids"] = torch.full(
                    (batch["states"].shape[0],), int(terrain_id), dtype=torch.long
                )
        if len(batches) == 1:
            yield batches[0]
            continue
        keys = batches[0].keys()
        merged = {
            key: torch.cat([batch[key] for batch in batches], dim=0)
            for key in keys
        }
        yield merged


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def metric_suffix(name: str) -> str:
    suffix = "".join(character if character.isalnum() else "_" for character in name.lower())
    return suffix.strip("_") or "dataset"


def allocate_batch_sizes(batch_size: int, fractions: list[float]) -> list[int]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not fractions:
        raise ValueError("at least one training dataset is required")
    if any(fraction < 0.0 for fraction in fractions):
        raise ValueError(f"train batch fractions must be non-negative, got {fractions}")
    if sum(fractions) <= 0.0:
        raise ValueError("at least one train batch fraction must be positive")

    normalized = [fraction / sum(fractions) for fraction in fractions]
    exact_counts = [batch_size * fraction for fraction in normalized]
    counts = [int(math.floor(count)) for count in exact_counts]
    remainder = batch_size - sum(counts)
    order = sorted(
        range(len(counts)),
        key=lambda index: exact_counts[index] - counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1

    positive_indices = [index for index, fraction in enumerate(fractions) if fraction > 0.0]
    if len(positive_indices) > batch_size:
        raise ValueError(
            f"batch_size={batch_size} is too small for {len(positive_indices)} positive train sources"
        )
    for index in positive_indices:
        if counts[index] > 0:
            continue
        donor = max(
            (candidate for candidate in positive_indices if counts[candidate] > 1),
            key=lambda candidate: counts[candidate],
            default=None,
        )
        if donor is None:
            raise ValueError(f"could not allocate positive batch count for train source {index}")
        counts[donor] -= 1
        counts[index] = 1
    return counts


def validate_compatible_metadata(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    candidate_root: Path,
) -> None:
    for key in ("state_fields", "action_fields", "rollout_fields"):
        if list(candidate.get(key, [])) != list(reference.get(key, [])):
            raise ValueError(f"{candidate_root} has incompatible {key}")
    if abs(float(candidate["dt_s"]) - float(reference["dt_s"])) > 1e-12:
        raise ValueError(
            f"{candidate_root} has dt_s={candidate['dt_s']}, expected {reference['dt_s']}"
        )


def build_optimizer(model: HMMWVDynamicsModel, optimizer_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() >= 2 and not name.endswith("bias"):
            decay_params.append(parameter)
        else:
            no_decay_params.append(parameter)
    groups = [
        {"params": decay_params, "weight_decay": float(optimizer_cfg["weight_decay"])},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=float(optimizer_cfg["lr"]),
        betas=tuple(float(x) for x in optimizer_cfg["betas"]),
    )


class HMMWVTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.processed_root = Path(config["processed_dataset_dir"]).resolve()
        self.output_dir = Path(config["output_dir"]).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metadata = load_metadata(self.processed_root)
        self.state_index = {
            field_name: field_index
            for field_index, field_name in enumerate(self.metadata["state_fields"])
        }

        # Terrain conditioning: concatenate a one-hot terrain code to each token so a
        # shared backbone can specialize per terrain (attacks the flat tax). The label
        # is free — each dataset spec's terrain is given by its "terrain" key, or its
        # "name" when that already matches a terrain (e.g. the "flat"/"crm" mix sources).
        terrain_cfg = config.get("terrain_conditioning", {})
        self.terrain_enabled = bool(terrain_cfg.get("enabled", False))
        self.terrains = [str(name) for name in terrain_cfg.get("terrains", [])] if self.terrain_enabled else []
        self.terrain_to_id = {name: index for index, name in enumerate(self.terrains)}
        self.num_terrains = len(self.terrains)
        if self.terrain_enabled:
            if self.num_terrains < 2:
                raise ValueError("terrain_conditioning.terrains needs at least 2 entries when enabled")
            self.metadata["terrain_conditioning"] = {
                "terrains": self.terrains,
                "num_terrains": self.num_terrains,
            }
            print(f"terrain_conditioning: one-hot over {self.terrains}")

        # Optional: override the model's input/output normalization with an
        # equal-domain-combined (flat+CRM) normalization, so CRM inputs are not
        # left off-center by flat-only stats. Pooled equal-weight per domain:
        # combined_var = mean_d(std_d^2 + (mean_d - combined_mean)^2). This is
        # baked into self.metadata so it is also saved in the checkpoint.
        model_norm_cfg = config.get("model_normalization")
        if model_norm_cfg and model_norm_cfg.get("mode") == "equal_domain_combined":
            self.metadata["normalization"] = self._equal_domain_normalization(model_norm_cfg["datasets"])
            print(f"model_normalization: equal_domain_combined over {model_norm_cfg['datasets']}")

        training_cfg = config["training"]
        self.sequence_length = int(config["model"]["block_size"])
        self.device = resolve_device(training_cfg.get("device", "auto"))
        self.seed = int(training_cfg["seed"])
        load_dataset_into_memory = bool(training_cfg.get("load_dataset_into_memory", False))
        seed_everything(self.seed)

        batch_size = int(training_cfg["batch_size"])
        self.num_epochs = int(training_cfg["num_epochs"])
        self.steps_per_epoch = int(training_cfg["steps_per_epoch"])
        self.max_val_batches = int(training_cfg["max_val_batches"])
        num_workers = int(training_cfg.get("num_workers", 0))
        pin_memory = bool(training_cfg.get("pin_memory", self.device.type == "cuda"))

        train_mix_cfg = config.get("train_mix", {})
        train_specs = list(train_mix_cfg.get("datasets", []))
        if not train_specs:
            train_specs = [
                {
                    "name": config.get("validation_dataset_name", "primary"),
                    "processed_dataset_dir": str(self.processed_root),
                    "batch_fraction": 1.0,
                }
            ]
        batch_fractions = [float(spec.get("batch_fraction", spec.get("fraction", 1.0))) for spec in train_specs]
        batch_counts = allocate_batch_sizes(batch_size, batch_fractions)

        self.train_loaders: list[DataLoader] = []
        self.train_terrain_ids: list[int] = []
        self.train_source_batch_sizes: dict[str, int] = {}
        self.train_dataset: WindowedHMMWVDataset | None = None
        for source_index, (spec, source_batch_size) in enumerate(zip(train_specs, batch_counts, strict=True)):
            if source_batch_size <= 0:
                continue
            source_name = str(spec.get("name", f"source_{source_index}"))
            source_root = Path(spec.get("processed_dataset_dir", self.processed_root)).resolve()
            source_metadata = load_metadata(source_root)
            validate_compatible_metadata(self.metadata, source_metadata, source_root)
            max_train_windows = spec.get("max_train_windows")
            if max_train_windows is None and source_root == self.processed_root:
                max_train_windows = training_cfg.get("max_train_windows")
            source_dataset = WindowedHMMWVDataset(
                source_root,
                split="train",
                sequence_length=self.sequence_length,
                max_windows=max_train_windows,
                seed=self.seed + source_index,
                load_into_memory=bool(spec.get("load_dataset_into_memory", load_dataset_into_memory)),
            )
            if self.train_dataset is None:
                self.train_dataset = source_dataset
            train_sampler = RandomSampler(
                source_dataset,
                replacement=True,
                num_samples=self.steps_per_epoch * source_batch_size,
            )
            self.train_loaders.append(
                DataLoader(
                    source_dataset,
                    batch_size=source_batch_size,
                    sampler=train_sampler,
                    drop_last=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )
            self.train_source_batch_sizes[source_name] = source_batch_size
            if self.terrain_enabled:
                self.train_terrain_ids.append(self._resolve_terrain_id(spec))
        if not self.train_loaders or self.train_dataset is None:
            raise ValueError("no training loaders were configured")

        self.val_dataset = WindowedHMMWVDataset(
            self.processed_root,
            split="val",
            sequence_length=self.sequence_length,
            max_windows=training_cfg.get("max_val_windows"),
            seed=self.seed + 1,
            load_into_memory=load_dataset_into_memory,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        validation_dataset_raw = str(config.get("validation_dataset_name", "primary"))
        self.validation_dataset_name = metric_suffix(validation_dataset_raw)
        # The primary val set is the model's processed_root (flat). Default to the
        # first terrain when its name does not literally match the vocabulary.
        self.primary_val_terrain_id = self._resolve_terrain_id(validation_dataset_raw, default=0)
        self.extra_val_loaders: list[dict[str, Any]] = []
        for source_index, spec in enumerate(config.get("validation_datasets", [])):
            source_name = metric_suffix(str(spec.get("name", f"extra_{source_index}")))
            source_root = Path(spec["processed_dataset_dir"]).resolve()
            source_metadata = load_metadata(source_root)
            validate_compatible_metadata(self.metadata, source_metadata, source_root)
            source_dataset = WindowedHMMWVDataset(
                source_root,
                split=str(spec.get("split", "val")),
                sequence_length=self.sequence_length,
                max_windows=spec.get("max_val_windows"),
                seed=int(spec.get("seed", self.seed + 100 + source_index)),
                load_into_memory=bool(spec.get("load_dataset_into_memory", load_dataset_into_memory)),
            )
            source_loader = DataLoader(
                source_dataset,
                batch_size=int(spec.get("batch_size", batch_size)),
                shuffle=False,
                drop_last=False,
                num_workers=int(spec.get("num_workers", num_workers)),
                pin_memory=bool(spec.get("pin_memory", pin_memory)),
            )
            self.extra_val_loaders.append(
                {
                    "name": source_name,
                    "loader": source_loader,
                    "max_batches": int(spec.get("max_val_batches", self.max_val_batches)),
                    "terrain_id": self._resolve_terrain_id(spec, default=0),
                }
            )

        normalization = self.metadata["normalization"]
        self.model = HMMWVDynamicsModel(
            state_dim=len(self.metadata["state_fields"]),
            action_dim=len(self.metadata["action_fields"]),
            target_dim=len(self.metadata["state_fields"]),
            transformer_cfg=config["model"],
            normalization=normalization,
            num_terrains=self.num_terrains if self.terrain_enabled else 0,
        ).to(self.device)

        if bool(training_cfg.get("compile", False)) and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.optimizer = build_optimizer(self.model, config["optimizer"])
        self.grad_clip_norm = float(config["optimizer"].get("grad_clip_norm", 1.0))
        self.rollout_eval = config.get("rollout_eval", {})
        self.validation_loss_weights = {
            metric_suffix(str(name)): float(weight)
            for name, weight in config.get("validation_loss_weights", {}).items()
        }

        # Loss configuration: per-channel weights (to stop a single channel such
        # as CRM normal-force from dominating the flat-normalized MSE) plus an
        # optional robust (Huber) loss. Defaults reproduce plain MSE.
        loss_cfg = config.get("loss", {})
        self.loss_type = str(loss_cfg.get("type", "mse")).lower()
        self.huber_delta = float(loss_cfg.get("huber_delta", 1.0))
        self.channel_weights = self._build_channel_weights(loss_cfg)
        if self.channel_weights is not None:
            named = {
                field: round(float(weight), 5)
                for field, weight in zip(self.metadata["state_fields"], self.channel_weights.tolist(), strict=True)
            }
            print(f"loss: type={self.loss_type} huber_delta={self.huber_delta} channel_weights={named}")

        self.checkpoint_metric = str(training_cfg.get("checkpoint_metric", "val_loss"))
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.start_epoch = 0
        self.dt_s = float(self.metadata["dt_s"])
        if training_cfg.get("resume_from_checkpoint"):
            self.load_checkpoint(Path(training_cfg["resume_from_checkpoint"]))

    def _resolve_terrain_id(self, spec: dict[str, Any] | str, default: int | None = None) -> int | None:
        """Terrain id for a dataset spec; None when conditioning is disabled.

        Prefers an explicit ``terrain`` key, falls back to ``name`` when it matches a
        terrain in the vocabulary, then to ``default`` (used for legacy single-domain
        eval). Raises if conditioning is on and nothing resolves.
        """
        if not self.terrain_enabled:
            return None
        if isinstance(spec, str):
            key = spec
        else:
            key = spec.get("terrain") or spec.get("name")
        if key in self.terrain_to_id:
            return self.terrain_to_id[key]
        if default is not None:
            return default
        raise ValueError(
            f"terrain_conditioning enabled but could not resolve terrain for {spec!r}; "
            f"add a \"terrain\" in {self.terrains}"
        )

    def scheduled_lr(self) -> float:
        optimizer_cfg = self.config["optimizer"]
        warmup_steps = int(optimizer_cfg.get("warmup_steps", 0))
        min_lr = float(optimizer_cfg.get("min_lr", optimizer_cfg["lr"]))
        max_lr = float(optimizer_cfg["lr"])
        total_steps = self.num_epochs * self.steps_per_epoch
        if warmup_steps > 0 and self.global_step < warmup_steps:
            return max_lr * float(self.global_step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return min_lr
        progress = (self.global_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + cosine * (max_lr - min_lr)

    def _equal_domain_normalization(self, dataset_dirs: list[str]) -> dict[str, list[float]]:
        metas = []
        for dataset_dir in dataset_dirs:
            source_meta = load_metadata(Path(dataset_dir).resolve())
            validate_compatible_metadata(self.metadata, source_meta, Path(dataset_dir))
            metas.append(source_meta["normalization"])
        out: dict[str, list[float]] = {}
        for mean_key, std_key in (("state_mean", "state_std"), ("action_mean", "action_std"), ("target_mean", "target_std")):
            means = np.stack([np.asarray(m[mean_key], dtype=np.float64) for m in metas], axis=0)
            stds = np.stack([np.asarray(m[std_key], dtype=np.float64) for m in metas], axis=0)
            combined_mean = means.mean(axis=0)
            combined_var = (stds ** 2 + (means - combined_mean) ** 2).mean(axis=0)
            out[mean_key] = combined_mean.tolist()
            out[std_key] = np.sqrt(np.maximum(combined_var, 1e-12)).tolist()
        return out

    def _build_channel_weights(self, loss_cfg: dict[str, Any]) -> torch.Tensor | None:
        """Per-channel loss weights, mean-normalized to 1.

        The training/validation loss is computed in the model's (flat) normalized
        target space. A channel whose delta scale differs wildly across domains
        (e.g. CRM normal-force, ~30x the flat per-step std) otherwise dominates the
        MSE by ~900x. ``equal_domain_combined_std`` reweights each channel so the
        residual is effectively normalized by the equal-domain-combined std
        (scale_i^2 = mean_d std_{d,i}^2) instead of the flat-only std:
        w_i = flat_std_i^2 / scale_i^2.
        """
        fields = self.metadata["state_fields"]
        explicit = loss_cfg.get("channel_weights")
        if explicit is not None:
            weights = np.asarray([float(explicit.get(field, 1.0)) for field in fields], dtype=np.float64)
        elif loss_cfg.get("channel_weight_mode") == "equal_domain_combined_std":
            dataset_dirs = loss_cfg["channel_weight_datasets"]
            stds = []
            for dataset_dir in dataset_dirs:
                source_meta = load_metadata(Path(dataset_dir).resolve())
                validate_compatible_metadata(self.metadata, source_meta, Path(dataset_dir))
                stds.append(np.asarray(source_meta["normalization"]["target_std"], dtype=np.float64))
            scale_sq = (np.stack(stds, axis=0) ** 2).mean(axis=0)
            flat_std = np.asarray(self.metadata["normalization"]["target_std"], dtype=np.float64)
            weights = (flat_std ** 2) / np.maximum(scale_sq, 1e-12)
        else:
            return None
        # Optional multiplicative emphasis on specific channels (e.g. upweight vx),
        # applied on top of the scale rebalancing, before mean-normalization.
        for field, factor in loss_cfg.get("channel_weight_overrides", {}).items():
            if field in fields:
                weights[fields.index(field)] *= float(factor)
        weights = weights * (weights.size / max(weights.sum(), 1e-12))  # mean-normalize to 1
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _compute_loss(self, prediction_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
        residual = prediction_norm - target_norm
        if self.channel_weights is not None:
            residual = residual * torch.sqrt(self.channel_weights)
        if self.loss_type == "huber":
            return torch.nn.functional.huber_loss(
                residual, torch.zeros_like(residual), delta=self.huber_delta, reduction="mean"
            )
        return residual.pow(2).mean()

    def training_step(self, batch: dict[str, torch.Tensor]) -> float:
        batch = move_batch(batch, self.device)
        lr = self.scheduled_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        prediction_norm = self.model(batch["states"], batch["actions"], terrain=batch.get("terrain_ids"))
        target_norm = self.model.normalize_target(batch["targets"])
        loss = self._compute_loss(prediction_norm, target_norm)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.global_step += 1
        return float(loss.item())

    @torch.no_grad()
    def _evaluate_window_loader(
        self, loader: DataLoader, max_batches: int, terrain_id: int | None = None
    ) -> dict[str, Any]:
        total_loss = 0.0
        total_batches = 0
        total_tokens = 0
        state_sq_error = torch.zeros(len(self.metadata["state_fields"]), dtype=torch.float64)

        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = move_batch(batch, self.device)
            prediction_norm = self.model(batch["states"], batch["actions"], terrain=terrain_id)
            target_norm = self.model.normalize_target(batch["targets"])
            loss = self._compute_loss(prediction_norm, target_norm)
            total_loss += float(loss.item())
            total_batches += 1

            prediction = self.model.denormalize_target(prediction_norm)
            diff = prediction - batch["targets"]
            state_sq_error += diff.pow(2).sum(dim=(0, 1)).cpu().double()
            total_tokens += diff.shape[0] * diff.shape[1]

        rmse = torch.sqrt(state_sq_error / max(total_tokens, 1))
        return {
            "loss": total_loss / max(total_batches, 1),
            "rmse": {
                field: float(value)
                for field, value in zip(self.metadata["state_fields"], rmse.tolist(), strict=True)
            },
        }

    @torch.no_grad()
    def evaluate_windows(self) -> dict[str, Any]:
        self.model.eval()
        primary_metrics = self._evaluate_window_loader(
            self.val_loader, self.max_val_batches, self.primary_val_terrain_id
        )
        metrics = {
            "val_loss": primary_metrics["loss"],
            "val_rmse": primary_metrics["rmse"],
            f"val_{self.validation_dataset_name}_loss": primary_metrics["loss"],
            f"val_{self.validation_dataset_name}_rmse": primary_metrics["rmse"],
        }
        for source in self.extra_val_loaders:
            source_metrics = self._evaluate_window_loader(
                source["loader"], source["max_batches"], source.get("terrain_id")
            )
            metrics[f"val_{source['name']}_loss"] = source_metrics["loss"]
            metrics[f"val_{source['name']}_rmse"] = source_metrics["rmse"]

        if self.validation_loss_weights:
            weighted_loss = 0.0
            total_weight = 0.0
            for name, weight in self.validation_loss_weights.items():
                metric_name = f"val_{name}_loss"
                if metric_name not in metrics:
                    raise KeyError(
                        f"validation loss weight references {name!r}, but {metric_name} was not computed"
                    )
                weighted_loss += weight * float(metrics[metric_name])
                total_weight += weight
            metrics["val_mixed_loss"] = weighted_loss / max(total_weight, 1e-12)
        return metrics

    @torch.no_grad()
    def evaluate_rollouts(self) -> dict[str, Any]:
        rollout_cfg = self.rollout_eval
        if not rollout_cfg:
            return {}

        # Domains to roll out. Legacy configs (no "datasets") roll out the primary
        # processed_root only; multi-domain configs list flat + CRM (+ bumpy) so we
        # can select on a combined, distance-normalized open-loop error.
        domains = rollout_cfg.get("datasets")
        if not domains:
            domains = [
                {
                    "name": self.validation_dataset_name,
                    "processed_dataset_dir": str(self.processed_root),
                    "weight": 1.0,
                }
            ]
        horizons_s = [float(value) for value in rollout_cfg.get("horizons_s", [1.0, 2.0, 5.0])]
        horizons_steps = [max(1, int(round(horizon / self.dt_s))) for horizon in horizons_s]
        num_episodes = int(rollout_cfg.get("num_episodes", 24))
        selection_horizon_s = float(rollout_cfg.get("selection_horizon_s", horizons_s[-1]))

        metrics: dict[str, Any] = {}
        selection_terms: list[float] = []
        selection_weight = 0.0
        self.model.eval()
        for domain in domains:
            name = metric_suffix(str(domain.get("name", "rollout")))
            domain_root = Path(domain["processed_dataset_dir"]).resolve()
            weight = float(domain.get("weight", 1.0))
            terrain_id = self._resolve_terrain_id(domain, default=self.primary_val_terrain_id)
            split_data = load_rollout_split(domain_root, str(domain.get("split", "val")))
            selected_episodes = self._select_rollout_episodes(split_data["episodes"], num_episodes)

            for horizon_s, horizon_steps in zip(horizons_s, horizons_steps, strict=True):
                pos_sq_error = 0.0
                yaw_sq_error = 0.0
                count = 0
                gt_distance = 0.0
                episode_count = 0
                for episode in selected_episodes:
                    result = self._rollout_episode(episode, horizon_steps, terrain_id)
                    if result is None:
                        continue
                    predicted_states, predicted_pose, _, gt_pose = result
                    pos_sq_error += ((predicted_pose[:, :2] - gt_pose[:, :2]).pow(2).sum(dim=-1)).sum().item()
                    yaw_sq_error += wrap_angle(predicted_pose[:, 2] - gt_pose[:, 2]).pow(2).sum().item()
                    count += predicted_states.shape[0]
                    gt_xy = gt_pose[:, :2]
                    if gt_xy.shape[0] >= 2:
                        gt_distance += (gt_xy[1:] - gt_xy[:-1]).pow(2).sum(dim=-1).sqrt().sum().item()
                    episode_count += 1

                if count == 0:
                    continue
                xy_rmse = math.sqrt(pos_sq_error / count)
                mean_distance = gt_distance / max(episode_count, 1)
                # distance-normalized error: the honest cross-domain comparison,
                # since CRM episodes are short/slow relative to flat.
                errdist = xy_rmse / mean_distance if mean_distance > 1e-6 else float("nan")
                metrics[f"rollout_{name}_{horizon_s:.1f}s"] = {
                    "xy_rmse_m": xy_rmse,
                    "yaw_rmse_rad": math.sqrt(yaw_sq_error / count),
                    "errdist": errdist,
                    "mean_dist_m": mean_distance,
                    "episodes": episode_count,
                }
                if abs(horizon_s - selection_horizon_s) < 1e-9 and math.isfinite(errdist):
                    selection_terms.append(weight * errdist)
                    selection_weight += weight

        if selection_weight > 0.0:
            metrics["rollout_sel"] = sum(selection_terms) / selection_weight
        return metrics

    def _select_rollout_episodes(self, episodes: list[dict[str, Any]], max_episodes: int) -> list[dict[str, Any]]:
        by_family: dict[str, list[dict[str, Any]]] = {}
        for episode in episodes:
            by_family.setdefault(episode["scenario_family"], []).append(episode)

        families = sorted(by_family)
        selected: list[dict[str, Any]] = []
        family_index = 0
        while len(selected) < max_episodes and families:
            family = families[family_index % len(families)]
            family_episodes = by_family[family]
            if family_episodes:
                selected.append(family_episodes.pop(0))
            if not family_episodes:
                families.remove(family)
                family_index -= 1
            family_index += 1
        return selected

    def _rollout_episode(
        self,
        episode: dict[str, Any],
        horizon_steps: int,
        terrain_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        states = torch.from_numpy(episode["states"]).to(self.device)
        actions = torch.from_numpy(episode["actions"]).to(self.device)
        rollout = torch.from_numpy(episode["rollout"]).to(self.device)
        if states.shape[0] <= self.sequence_length + 1:
            return None

        steps = min(horizon_steps, states.shape[0] - self.sequence_length)
        if steps <= 0:
            return None

        history_states = states[: self.sequence_length].clone()
        history_actions = actions[: self.sequence_length].clone()
        current_pose = rollout[self.sequence_length - 1].clone()
        predicted_states: list[torch.Tensor] = []
        predicted_pose: list[torch.Tensor] = []

        for step_index in range(steps):
            state_window = history_states[-self.sequence_length :].unsqueeze(0)
            action_window = history_actions[-self.sequence_length :].unsqueeze(0)
            delta = self.model.predict_delta(state_window, action_window, terrain=terrain_id)[:, -1, :].squeeze(0)
            next_state = history_states[-1] + delta
            current_pose = self._integrate_pose(current_pose, next_state)
            predicted_states.append(next_state)
            predicted_pose.append(current_pose.clone())

            if self.sequence_length + step_index < actions.shape[0]:
                history_actions = torch.cat(
                    [history_actions, actions[self.sequence_length + step_index].unsqueeze(0)],
                    dim=0,
                )
            history_states = torch.cat([history_states, next_state.unsqueeze(0)], dim=0)

        gt_states = states[self.sequence_length : self.sequence_length + steps]
        gt_pose = rollout[self.sequence_length : self.sequence_length + steps]
        return (
            torch.stack(predicted_states, dim=0),
            torch.stack(predicted_pose, dim=0),
            gt_states,
            gt_pose,
        )

    def _integrate_pose(self, pose: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        x_pos, y_pos, yaw = pose
        yaw_rate = next_state[self.state_index["yaw_rate_radps"]]
        vx_body = next_state[self.state_index["vel_body_x_mps"]]
        vy_body = next_state[self.state_index["vel_body_y_mps"]]
        yaw_next = yaw + self.dt_s * yaw_rate
        cos_yaw = torch.cos(yaw_next)
        sin_yaw = torch.sin(yaw_next)
        vx_world = cos_yaw * vx_body - sin_yaw * vy_body
        vy_world = sin_yaw * vx_body + cos_yaw * vy_body
        x_next = x_pos + self.dt_s * vx_world
        y_next = y_pos + self.dt_s * vy_world
        return torch.stack([x_next, y_next, yaw_next])

    def save_checkpoint(self, name: str, epoch: int, metrics: dict[str, Any]) -> Path:
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "config": self.config,
                "metadata": self.metadata,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("global_step", 0))

        if self.metrics_path.exists():
            records = [
                json.loads(line)
                for line in self.metrics_path.read_text().splitlines()
                if line.strip()
            ]
            if records:
                self.best_val_loss = min(
                    float(record.get(self.checkpoint_metric, record["val_loss"]))
                    for record in records
                )
        elif checkpoint.get("metrics"):
            metrics = checkpoint["metrics"]
            if self.checkpoint_metric in metrics or "val_loss" in metrics:
                self.best_val_loss = float(metrics.get(self.checkpoint_metric, metrics["val_loss"]))
        print(
            f"resumed from {checkpoint_path} at epoch {self.start_epoch}, "
            f"global_step {self.global_step}, best {self.checkpoint_metric}={self.best_val_loss}"
        )

    def log_metrics(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a") as fp:
            fp.write(json.dumps(record) + "\n")

    def train(self) -> Path:
        train_iterator = mixed_infinite_loader(
            self.train_loaders, self.train_terrain_ids if self.terrain_enabled else None
        )
        last_checkpoint = self.checkpoint_dir / "last.pt"
        if self.start_epoch >= self.num_epochs:
            print(f"training already reached epoch {self.start_epoch}; target is {self.num_epochs}")
            return last_checkpoint

        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            self.model.train()
            epoch_losses: list[float] = []
            for _ in range(self.steps_per_epoch):
                batch = next(train_iterator)
                loss = self.training_step(batch)
                epoch_losses.append(loss)

            window_metrics = self.evaluate_windows()
            rollout_metrics = self.evaluate_rollouts()
            record = {
                "epoch": epoch,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "train_loss": float(sum(epoch_losses) / max(len(epoch_losses), 1)),
                **window_metrics,
                **rollout_metrics,
            }
            self.log_metrics(record)
            self.save_checkpoint("last", epoch, record)
            if self.checkpoint_metric not in record:
                raise KeyError(f"checkpoint metric {self.checkpoint_metric!r} was not logged")
            checkpoint_value = float(record[self.checkpoint_metric])
            if checkpoint_value < self.best_val_loss:
                self.best_val_loss = checkpoint_value
                self.save_checkpoint("best_val", epoch, record)
            print(json.dumps(record, indent=2))
        return last_checkpoint


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = merge_cli_overrides(load_json(args.config.resolve()), args)
    trainer = HMMWVTrainer(config)
    final_checkpoint = trainer.train()
    print(f"training completed; last checkpoint: {final_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
