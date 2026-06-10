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


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


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

        training_cfg = config["training"]
        self.sequence_length = int(config["model"]["block_size"])
        self.device = resolve_device(training_cfg.get("device", "auto"))
        self.seed = int(training_cfg["seed"])
        load_dataset_into_memory = bool(training_cfg.get("load_dataset_into_memory", False))
        seed_everything(self.seed)

        self.train_dataset = WindowedHMMWVDataset(
            self.processed_root,
            split="train",
            sequence_length=self.sequence_length,
            max_windows=training_cfg.get("max_train_windows"),
            seed=self.seed,
            load_into_memory=load_dataset_into_memory,
        )
        self.val_dataset = WindowedHMMWVDataset(
            self.processed_root,
            split="val",
            sequence_length=self.sequence_length,
            max_windows=training_cfg.get("max_val_windows"),
            seed=self.seed + 1,
            load_into_memory=load_dataset_into_memory,
        )

        batch_size = int(training_cfg["batch_size"])
        self.num_epochs = int(training_cfg["num_epochs"])
        self.steps_per_epoch = int(training_cfg["steps_per_epoch"])
        num_workers = int(training_cfg.get("num_workers", 0))
        pin_memory = bool(training_cfg.get("pin_memory", self.device.type == "cuda"))
        train_sampler = RandomSampler(
            self.train_dataset,
            replacement=True,
            num_samples=self.steps_per_epoch * batch_size,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        normalization = self.metadata["normalization"]
        self.model = HMMWVDynamicsModel(
            state_dim=len(self.metadata["state_fields"]),
            action_dim=len(self.metadata["action_fields"]),
            target_dim=len(self.metadata["state_fields"]),
            transformer_cfg=config["model"],
            normalization=normalization,
        ).to(self.device)

        if bool(training_cfg.get("compile", False)) and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.optimizer = build_optimizer(self.model, config["optimizer"])
        self.max_val_batches = int(training_cfg["max_val_batches"])
        self.grad_clip_norm = float(config["optimizer"].get("grad_clip_norm", 1.0))
        self.rollout_eval = config.get("rollout_eval", {})
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.dt_s = float(self.metadata["dt_s"])

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

    def training_step(self, batch: dict[str, torch.Tensor]) -> float:
        batch = move_batch(batch, self.device)
        lr = self.scheduled_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        prediction_norm = self.model(batch["states"], batch["actions"])
        target_norm = self.model.normalize_target(batch["targets"])
        loss = torch.nn.functional.mse_loss(prediction_norm, target_norm)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.global_step += 1
        return float(loss.item())

    @torch.no_grad()
    def evaluate_windows(self) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        total_tokens = 0
        state_sq_error = torch.zeros(len(self.metadata["state_fields"]), dtype=torch.float64)

        for batch_index, batch in enumerate(self.val_loader):
            if batch_index >= self.max_val_batches:
                break
            batch = move_batch(batch, self.device)
            prediction_norm = self.model(batch["states"], batch["actions"])
            target_norm = self.model.normalize_target(batch["targets"])
            loss = torch.nn.functional.mse_loss(prediction_norm, target_norm)
            total_loss += float(loss.item())
            total_batches += 1

            prediction = self.model.denormalize_target(prediction_norm)
            diff = prediction - batch["targets"]
            state_sq_error += diff.pow(2).sum(dim=(0, 1)).cpu().double()
            total_tokens += diff.shape[0] * diff.shape[1]

        rmse = torch.sqrt(state_sq_error / max(total_tokens, 1))
        metrics = {
            "val_loss": total_loss / max(total_batches, 1),
            "val_rmse": {
                field: float(value)
                for field, value in zip(self.metadata["state_fields"], rmse.tolist(), strict=True)
            },
        }
        return metrics

    @torch.no_grad()
    def evaluate_rollouts(self) -> dict[str, Any]:
        rollout_cfg = self.rollout_eval
        if not rollout_cfg:
            return {}

        split_data = load_rollout_split(self.processed_root, "val")
        selected_episodes = self._select_rollout_episodes(
            split_data["episodes"],
            max_episodes=int(rollout_cfg.get("num_episodes", 24)),
        )
        horizons_s = [float(value) for value in rollout_cfg.get("horizons_s", [1.0, 2.0, 5.0])]
        horizons_steps = [max(1, int(round(horizon / self.dt_s))) for horizon in horizons_s]

        metrics: dict[str, Any] = {}
        self.model.eval()
        for horizon_s, horizon_steps in zip(horizons_s, horizons_steps, strict=True):
            state_sq_error = torch.zeros(len(self.metadata["state_fields"]), dtype=torch.float64)
            pos_sq_error = 0.0
            yaw_sq_error = 0.0
            count = 0

            for episode in selected_episodes:
                result = self._rollout_episode(episode, horizon_steps)
                if result is None:
                    continue
                predicted_states, predicted_pose, gt_states, gt_pose = result
                state_sq_error += (predicted_states - gt_states).pow(2).sum(dim=0).cpu().double()
                pos_sq_error += ((predicted_pose[:, :2] - gt_pose[:, :2]).pow(2).sum(dim=-1)).sum().item()
                yaw_sq_error += wrap_angle(predicted_pose[:, 2] - gt_pose[:, 2]).pow(2).sum().item()
                count += predicted_states.shape[0]

            if count == 0:
                continue
            state_rmse = torch.sqrt(state_sq_error / count)
            metrics[f"rollout_{horizon_s:.1f}s"] = {
                "state_rmse": {
                    field: float(value)
                    for field, value in zip(self.metadata["state_fields"], state_rmse.tolist(), strict=True)
                },
                "xy_rmse_m": float(math.sqrt(pos_sq_error / count)),
                "yaw_rmse_rad": float(math.sqrt(yaw_sq_error / count)),
                "episodes": len(selected_episodes),
            }
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
            delta = self.model.predict_delta(state_window, action_window)[:, -1, :].squeeze(0)
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

    def log_metrics(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a") as fp:
            fp.write(json.dumps(record) + "\n")

    def train(self) -> Path:
        train_iterator = infinite_loader(self.train_loader)
        last_checkpoint = self.checkpoint_dir / "last.pt"
        for epoch in range(1, self.num_epochs + 1):
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
            if record["val_loss"] < self.best_val_loss:
                self.best_val_loss = record["val_loss"]
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
