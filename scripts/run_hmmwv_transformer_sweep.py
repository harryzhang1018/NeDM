from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a sweep of HMMWV transformer recipes.")
    parser.add_argument(
        "--recipes",
        type=Path,
        default=Path("configs/hmmwv_transformer_sweep_v04_v18.json"),
        help="Sweep recipe JSON.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device override for training and plotting.")
    parser.add_argument("--force-train", action="store_true", help="Retrain models even when last.pt exists.")
    parser.add_argument("--force-eval", action="store_true", help="Rerun 20-episode overlay eval when summary exists.")
    parser.add_argument("--max-models", type=int, default=None, help="Optional cap for smoke testing.")
    parser.add_argument("--start-at", type=str, default=None, help="Optional version to start at, e.g. v10.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_status(
    run_root: Path,
    state: str,
    stage: str,
    message: str,
    current_model: str | None,
    completed: int,
    total: int,
) -> None:
    write_json(
        run_root / "status.json",
        {
            "updated_at_utc": now_utc(),
            "state": state,
            "stage": stage,
            "message": message,
            "current_model": current_model,
            "completed": completed,
            "total": total,
        },
    )


def build_config(sweep: dict[str, Any], recipe: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    model_cfg = deep_merge(sweep["model_defaults"], recipe.get("model"))
    optimizer_cfg = deep_merge(sweep["optimizer_defaults"], recipe.get("optimizer"))
    training_cfg = deep_merge(sweep["training_defaults"], recipe.get("training"))
    return {
        "processed_dataset_dir": sweep["processed_dataset_dir"],
        "output_dir": str(output_dir),
        "model": model_cfg,
        "optimizer": optimizer_cfg,
        "training": training_cfg,
        "rollout_eval": {},
        "sweep_recipe": {
            "version": recipe["version"],
            "slug": recipe["slug"],
            "notes": recipe.get("notes", ""),
        },
    }


def run_command(command: list[str], log_label: str) -> None:
    print(f"[{now_utc()}] {log_label}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def read_metrics(metrics_path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No metrics found in {metrics_path}")
    best = min(records, key=lambda record: float(record["val_loss"]))
    last = records[-1]
    return {
        "epochs": len(records),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "best_train_loss": float(best["train_loss"]),
        "last_epoch": int(last["epoch"]),
        "last_val_loss": float(last["val_loss"]),
        "last_train_loss": float(last["train_loss"]),
    }


def training_is_complete(output_dir: Path, expected_epochs: int) -> bool:
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "checkpoints" / "best_val.pt"
    if not metrics_path.exists() or not checkpoint_path.exists():
        return False
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not records:
        return False
    return int(records[-1].get("epoch", 0)) >= expected_epochs


def archive_interrupted_run(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = output_dir.with_name(f"{output_dir.name}_interrupted_{timestamp}")
    suffix = 1
    while archive_path.exists():
        archive_path = output_dir.with_name(f"{output_dir.name}_interrupted_{timestamp}_{suffix}")
        suffix += 1
    shutil.move(str(output_dir), str(archive_path))
    return archive_path


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def aggregate_eval(summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    episodes = summary["episodes"]
    xy = [float(episode["metrics"]["xy_rmse_m"]) for episode in episodes]
    yaw = [float(episode["metrics"]["yaw_rmse_rad"]) for episode in episodes]
    steps = [int(episode["rollout_steps"]) for episode in episodes]
    families = Counter(episode["scenario_family"] for episode in episodes)
    return {
        "summary_path": str(summary_path),
        "num_episodes": len(episodes),
        "xy_rmse_mean_m": float(statistics.mean(xy)),
        "xy_rmse_median_m": float(statistics.median(xy)),
        "xy_rmse_p75_m": percentile(xy, 0.75),
        "xy_rmse_p90_m": percentile(xy, 0.90),
        "xy_rmse_min_m": float(min(xy)),
        "xy_rmse_max_m": float(max(xy)),
        "yaw_rmse_mean_rad": float(statistics.mean(yaw)),
        "yaw_rmse_median_rad": float(statistics.median(yaw)),
        "rollout_steps_mean": float(statistics.mean(steps)),
        "families": dict(sorted(families.items())),
    }


def load_results(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    latest_by_version: dict[str, dict[str, Any]] = {}
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        latest_by_version[record["version"]] = record
    return list(latest_by_version.values())


def write_leaderboard(run_root: Path, results_path: Path) -> None:
    records = load_results(results_path)
    completed = [
        record
        for record in records
        if record.get("status") == "complete" and record.get("eval", {}).get("num_episodes", 0) > 0
    ]
    completed.sort(
        key=lambda record: (
            record["eval"]["xy_rmse_median_m"],
            record["eval"]["xy_rmse_mean_m"],
            record["training"]["best_val_loss"],
        )
    )

    leaderboard = {
        "updated_at_utc": now_utc(),
        "rank_metric": "eval.xy_rmse_median_m, then eval.xy_rmse_mean_m",
        "num_complete": len(completed),
        "models": completed,
    }
    write_json(run_root / "leaderboard.json", leaderboard)

    lines = [
        "# HMMWV Transformer Sweep Leaderboard",
        "",
        "Rank metric: median XY RMSE over the fixed 20 validation rollouts, then mean XY RMSE.",
        "",
        "| Rank | Version | Recipe | Best Val Loss | Median XY RMSE m | Mean XY RMSE m | P75 XY RMSE m | Max XY RMSE m | Median Yaw RMSE rad | Epochs |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, record in enumerate(completed, start=1):
        lines.append(
            "| {rank} | {version} | {slug} | {val:.6f} | {xy_med:.3f} | {xy_mean:.3f} | {xy_p75:.3f} | {xy_max:.3f} | {yaw_med:.4f} | {epochs} |".format(
                rank=rank,
                version=record["version"],
                slug=record["slug"],
                val=record["training"]["best_val_loss"],
                xy_med=record["eval"]["xy_rmse_median_m"],
                xy_mean=record["eval"]["xy_rmse_mean_m"],
                xy_p75=record["eval"].get("xy_rmse_p75_m", float("nan")),
                xy_max=record["eval"]["xy_rmse_max_m"],
                yaw_med=record["eval"]["yaw_rmse_median_rad"],
                epochs=record["training"]["epochs"],
            )
        )
    (run_root / "leaderboard.md").write_text("\n".join(lines) + "\n")

    robust_completed = sorted(
        completed,
        key=lambda record: (
            record["eval"]["xy_rmse_mean_m"],
            record["eval"].get("xy_rmse_p75_m", record["eval"]["xy_rmse_mean_m"]),
            record["eval"]["xy_rmse_median_m"],
            record["training"]["best_val_loss"],
        ),
    )
    robust_lines = [
        "# HMMWV Transformer Sweep Robust Leaderboard",
        "",
        "Rank metric: mean XY RMSE, then P75 XY RMSE, then median XY RMSE.",
        "",
        "| Rank | Version | Recipe | Best Val Loss | Mean XY RMSE m | P75 XY RMSE m | Median XY RMSE m | Max XY RMSE m | Epochs |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, record in enumerate(robust_completed, start=1):
        robust_lines.append(
            "| {rank} | {version} | {slug} | {val:.6f} | {xy_mean:.3f} | {xy_p75:.3f} | {xy_med:.3f} | {xy_max:.3f} | {epochs} |".format(
                rank=rank,
                version=record["version"],
                slug=record["slug"],
                val=record["training"]["best_val_loss"],
                xy_mean=record["eval"]["xy_rmse_mean_m"],
                xy_p75=record["eval"].get("xy_rmse_p75_m", float("nan")),
                xy_med=record["eval"]["xy_rmse_median_m"],
                xy_max=record["eval"]["xy_rmse_max_m"],
                epochs=record["training"]["epochs"],
            )
        )
    (run_root / "leaderboard_by_mean.md").write_text("\n".join(robust_lines) + "\n")


def append_result(results_path: Path, record: dict[str, Any]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def selected_recipes(recipes: list[dict[str, Any]], start_at: str | None, max_models: int | None) -> list[dict[str, Any]]:
    if start_at is not None:
        start_index = next(
            (index for index, recipe in enumerate(recipes) if recipe["version"] == start_at),
            None,
        )
        if start_index is None:
            raise ValueError(f"start version {start_at!r} not found")
        recipes = recipes[start_index:]
    if max_models is not None:
        recipes = recipes[:max_models]
    return recipes


def main() -> int:
    args = parse_args()
    sweep = load_json(args.recipes.resolve())
    run_root = Path(sweep.get("run_root", "artifacts/training_runs/hmmwv_sweep_v04_v18")).resolve()
    config_dir = run_root / "configs"
    results_path = run_root / "results.jsonl"
    recipes = selected_recipes(sweep["recipes"], args.start_at, args.max_models)
    total = len(recipes)

    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "recipes.resolved.json", sweep)
    write_status(run_root, "running", "start", "sweep started", None, 0, total)

    completed = 0
    for recipe in recipes:
        version = recipe["version"]
        slug = recipe["slug"]
        model_name = f"{version}_{slug}"
        output_dir = Path(f"artifacts/training_runs/hmmwv_transformer_{model_name}").resolve()
        config_path = config_dir / f"{model_name}.json"
        eval_cfg = sweep["eval"]
        eval_dir = output_dir / "plots" / (
            f"random_{eval_cfg['split']}_overlays_n{eval_cfg['num_random']}_seed_{eval_cfg['seed']}"
        )
        summary_path = eval_dir / "summary.json"
        checkpoint_path = output_dir / "checkpoints" / "best_val.pt"

        config = build_config(sweep, recipe, output_dir)
        config["training"]["device"] = args.device
        write_json(config_path, config)
        expected_epochs = int(config["training"]["num_epochs"])

        base_record = {
            "version": version,
            "slug": slug,
            "notes": recipe.get("notes", ""),
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "checkpoint_path": str(checkpoint_path),
            "started_at_utc": now_utc(),
            "model": config["model"],
            "optimizer": config["optimizer"],
            "training_config": config["training"],
        }

        try:
            complete_training = training_is_complete(output_dir, expected_epochs)
            if output_dir.exists() and not complete_training and not args.force_train:
                archive_path = archive_interrupted_run(output_dir)
                print(
                    f"[{now_utc()}] train {model_name}: archived interrupted run to {archive_path}",
                    flush=True,
                )
                complete_training = False

            if not complete_training or args.force_train:
                write_status(run_root, "running", "train", f"training {model_name}", model_name, completed, total)
                run_command(
                    [
                        sys.executable,
                        "scripts/train_hmmwv_dynamics.py",
                        "--config",
                        str(config_path),
                        "--device",
                        args.device,
                    ],
                    f"train {model_name}",
                )
            else:
                print(f"[{now_utc()}] train {model_name}: skipping existing checkpoint", flush=True)

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint_path}")

            if not summary_path.exists() or args.force_eval:
                write_status(run_root, "running", "eval", f"evaluating {model_name}", model_name, completed, total)
                eval_command = [
                    sys.executable,
                    "scripts/plot_hmmwv_rollout_overlay.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--split",
                    str(eval_cfg["split"]),
                    "--num-random",
                    str(eval_cfg["num_random"]),
                    "--seed",
                    str(eval_cfg["seed"]),
                    "--device",
                    args.device,
                    "--output-dir",
                    str(eval_dir),
                ]
                try:
                    run_command(eval_command, f"eval {model_name}")
                except subprocess.CalledProcessError:
                    if args.device == "cpu":
                        raise
                    cpu_eval_command = list(eval_command)
                    cpu_eval_command[cpu_eval_command.index("--device") + 1] = "cpu"
                    run_command(cpu_eval_command, f"eval {model_name} cpu_fallback")
            else:
                print(f"[{now_utc()}] eval {model_name}: skipping existing summary", flush=True)

            training_metrics = read_metrics(output_dir / "metrics.jsonl")
            eval_metrics = aggregate_eval(summary_path)
            record = {
                **base_record,
                "status": "complete",
                "finished_at_utc": now_utc(),
                "training": training_metrics,
                "eval": eval_metrics,
            }
            completed += 1
        except Exception as exc:  # noqa: BLE001 - keep the long sweep moving across failed recipes.
            record = {
                **base_record,
                "status": "failed",
                "finished_at_utc": now_utc(),
                "error": repr(exc),
            }
            print(f"[{now_utc()}] {model_name} failed: {exc!r}", flush=True)

        append_result(results_path, record)
        write_leaderboard(run_root, results_path)
        write_status(run_root, "running", "between_models", f"finished {model_name}", model_name, completed, total)

    write_leaderboard(run_root, results_path)
    final_records = load_results(results_path)
    complete_records = [record for record in final_records if record.get("status") == "complete"]
    best = None
    if complete_records:
        best = sorted(
            complete_records,
            key=lambda record: (
                record["eval"]["xy_rmse_median_m"],
                record["eval"]["xy_rmse_mean_m"],
                record["training"]["best_val_loss"],
            ),
        )[0]
    write_status(
        run_root,
        "complete",
        "complete",
        f"best={best['version']}_{best['slug']}" if best is not None else "no complete models",
        best["version"] if best is not None else None,
        completed,
        total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
