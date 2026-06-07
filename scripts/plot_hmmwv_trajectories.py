from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_episode_positions(csv_path: Path) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_values.append(float(row["pos_x_m"]))
            y_values.append(float(row["pos_y_m"]))
    return x_values, y_values


def load_dataset_index(dataset_root: Path) -> dict:
    with (dataset_root / "dataset_index.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_trajectories(dataset_root: Path, output_path: Path) -> None:
    dataset_index = load_dataset_index(dataset_root)
    episodes = dataset_index["episodes"]

    fig, ax = plt.subplots(figsize=(12, 10))
    family_names = sorted({episode.get("scenario_family", episode["scenario_name"]) for episode in episodes})
    family_colors = {
        family_name: plt.cm.tab10(index % 10)
        for index, family_name in enumerate(family_names)
    }
    large_dataset = len(episodes) > 20
    legend_seen: set[str] = set()

    for episode in episodes:
        csv_path = dataset_root / episode["csv_path"]
        x_values, y_values = read_episode_positions(csv_path)
        family_name = episode.get("scenario_family", episode["scenario_name"])
        color = family_colors[family_name]
        label = family_name if family_name not in legend_seen else None
        legend_seen.add(family_name)

        if large_dataset:
            ax.plot(x_values, y_values, linewidth=0.35, alpha=0.08, color=color)
        else:
            ax.plot(x_values, y_values, linewidth=2.0, color=color, label=label)
            ax.scatter([x_values[0]], [y_values[0]], s=30, marker="o", color=color)
            ax.scatter([x_values[-1]], [y_values[-1]], s=40, marker="x", color=color)

    if large_dataset:
        for family_name, color in family_colors.items():
            ax.plot([], [], color=color, linewidth=2.0, label=family_name)

    ax.set_title(f"Vehicle trajectories: {dataset_index['dataset_name']}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot XY vehicle trajectories for all episodes in a dataset.")
    parser.add_argument(
        "--dataset-root",
        default="artifacts/datasets/hmmwv_overfit_v1",
        help="Path to a collected dataset root.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Defaults to <dataset-root>/plots/vehicle_trajectories.png",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = dataset_root / "plots" / "vehicle_trajectories.png"

    plot_trajectories(dataset_root, output_path)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
