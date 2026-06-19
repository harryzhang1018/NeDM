from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-trajectory cleaning diagnostics for the HMMWV bumpy 15D processed cache."
    )
    parser.add_argument(
        "--bumpy-dir",
        type=Path,
        default=Path("artifacts/training_datasets/hmmwv_bumpy_10g_normal_force_omega_seq_v1"),
    )
    parser.add_argument(
        "--flat-dir",
        type=Path,
        default=Path("artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis/hmmwv_bumpy_trajectory_cleaning"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def format_float(value: float) -> str:
    if pd.isna(value):
        return ""
    abs_value = abs(float(value))
    if abs_value == 0.0:
        return "0"
    if abs_value >= 100000.0 or abs_value < 0.001:
        return f"{value:.3e}"
    if abs_value >= 1000.0:
        return f"{value:.1f}"
    if abs_value >= 100.0:
        return f"{value:.2f}"
    return f"{value:.4f}"


def bool_count(df: pd.DataFrame, column: str) -> int:
    return int(df[column].astype(bool).sum())


def summarize_filter(df: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    retained = int(mask.sum())
    removed = int((~mask).sum())
    transitions_retained = int(df.loc[mask, "length"].sum())
    transitions_removed = int(df.loc[~mask, "length"].sum())
    return {
        "retained_trajectories": retained,
        "removed_trajectories": removed,
        "retained_transition_rows": transitions_retained,
        "removed_transition_rows": transitions_removed,
        "retained_trajectory_frac": retained / max(len(df), 1),
        "retained_transition_frac": transitions_retained / max(int(df["length"].sum()), 1),
    }


def markdown_table(df: pd.DataFrame, columns: list[str], headers: list[str] | None = None) -> list[str]:
    if headers is None:
        headers = columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in columns[1:]]) + " |",
    ]
    for _, row in df[columns].iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, str):
                values.append(f"`{value}`")
            elif isinstance(value, (bool, np.bool_)):
                values.append(str(bool(value)))
            elif isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                values.append(format_float(float(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def analyze_split(
    root: Path,
    split: str,
    fields: list[str],
    flat_state_mean: np.ndarray,
    flat_state_std: np.ndarray,
    flat_target_mean: np.ndarray,
    flat_target_std: np.ndarray,
) -> list[dict[str, Any]]:
    split_meta = load_json(root / f"{split}_episodes.json")
    starts = np.load(root / f"{split}_episode_starts.npy")
    lengths = np.load(root / f"{split}_episode_lengths.npy")
    states = np.load(root / f"{split}_states.npy", mmap_mode="r")
    targets = np.load(root / f"{split}_targets.npy", mmap_mode="r")

    fz_indices = [index for index, field in enumerate(fields) if "_force_wheel_fz_n" in field]
    body_indices = list(range(7))
    omega_indices = [index for index, field in enumerate(fields) if "spindle_omega" in field]
    pitch_index = fields.index("pitch_rad")
    body_y_ang_vel_index = fields.index("ang_vel_body_y_radps")

    rows: list[dict[str, Any]] = []
    for episode_index, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist(), strict=True)):
        end = int(start) + int(length)
        episode_states = np.asarray(states[int(start) : end], dtype=np.float64)
        episode_targets = np.asarray(targets[int(start) : end], dtype=np.float64)
        fz_state = episode_states[:, fz_indices]
        fz_target = episode_targets[:, fz_indices]
        state_z = np.abs((episode_states - flat_state_mean) / flat_state_std)
        target_z = np.abs((episode_targets - flat_target_mean) / flat_target_std)

        fz_state_negative = fz_state < 0.0
        fz_state_gt_15k = fz_state > 15_000.0
        fz_state_gt_20k = fz_state > 20_000.0
        fz_state_gt_50k = fz_state > 50_000.0
        fz_state_gt_100k = fz_state > 100_000.0
        fz_target_abs = np.abs(fz_target)
        fz_target_abs_gt_1k = fz_target_abs > 1_000.0
        fz_target_abs_gt_5k = fz_target_abs > 5_000.0
        fz_target_abs_gt_20k = fz_target_abs > 20_000.0
        fz_target_abs_gt_100k = fz_target_abs > 100_000.0

        count_den = max(int(length) * len(fz_indices), 1)
        row: dict[str, Any] = {
            "split": split,
            "episode_index": episode_index,
            "episode_id": split_meta["episode_ids"][episode_index],
            "scenario_family": split_meta["scenario_families"][episode_index],
            "source_dataset": split_meta["source_datasets"][episode_index],
            "source_csv_path": split_meta["source_csv_paths"][episode_index],
            "start": int(start),
            "length": int(length),
            "fz_state_min": float(np.min(fz_state)),
            "fz_state_max": float(np.max(fz_state)),
            "fz_target_abs_max": float(np.max(fz_target_abs)),
            "fz_state_negative_count": int(fz_state_negative.sum()),
            "fz_state_gt_15k_count": int(fz_state_gt_15k.sum()),
            "fz_state_gt_20k_count": int(fz_state_gt_20k.sum()),
            "fz_state_gt_50k_count": int(fz_state_gt_50k.sum()),
            "fz_state_gt_100k_count": int(fz_state_gt_100k.sum()),
            "fz_target_abs_gt_1k_count": int(fz_target_abs_gt_1k.sum()),
            "fz_target_abs_gt_5k_count": int(fz_target_abs_gt_5k.sum()),
            "fz_target_abs_gt_20k_count": int(fz_target_abs_gt_20k.sum()),
            "fz_target_abs_gt_100k_count": int(fz_target_abs_gt_100k.sum()),
            "fz_state_negative_frac": float(fz_state_negative.sum() / count_den),
            "fz_state_gt_15k_frac": float(fz_state_gt_15k.sum() / count_den),
            "fz_state_gt_20k_frac": float(fz_state_gt_20k.sum() / count_den),
            "fz_state_gt_50k_frac": float(fz_state_gt_50k.sum() / count_den),
            "fz_state_gt_100k_frac": float(fz_state_gt_100k.sum() / count_den),
            "fz_target_abs_gt_1k_frac": float(fz_target_abs_gt_1k.sum() / count_den),
            "fz_target_abs_gt_5k_frac": float(fz_target_abs_gt_5k.sum() / count_den),
            "fz_target_abs_gt_20k_frac": float(fz_target_abs_gt_20k.sum() / count_den),
            "fz_target_abs_gt_100k_frac": float(fz_target_abs_gt_100k.sum() / count_den),
            "state_abs_z_gt_5_frac_all": float((state_z > 5.0).mean()),
            "state_abs_z_gt_5_frac_body7": float((state_z[:, body_indices] > 5.0).mean()),
            "state_abs_z_gt_5_frac_fz": float((state_z[:, fz_indices] > 5.0).mean()),
            "state_abs_z_gt_5_frac_omega": float((state_z[:, omega_indices] > 5.0).mean()),
            "target_abs_z_gt_5_frac_all": float((target_z > 5.0).mean()),
            "target_abs_z_gt_5_frac_body7": float((target_z[:, body_indices] > 5.0).mean()),
            "target_abs_z_gt_5_frac_fz": float((target_z[:, fz_indices] > 5.0).mean()),
            "target_abs_z_gt_5_frac_omega": float((target_z[:, omega_indices] > 5.0).mean()),
            "state_abs_z_max_all": float(state_z.max()),
            "target_abs_z_max_all": float(target_z.max()),
            "pitch_state_abs_z_gt_5_frac": float((state_z[:, pitch_index] > 5.0).mean()),
            "pitch_target_abs_z_gt_5_frac": float((target_z[:, pitch_index] > 5.0).mean()),
            "body_y_ang_vel_target_abs_z_gt_5_frac": float(
                (target_z[:, body_y_ang_vel_index] > 5.0).mean()
            ),
            "any_negative_fz_state": bool(fz_state_negative.any()),
            "any_fz_state_gt_100k": bool(fz_state_gt_100k.any()),
            "any_fz_target_abs_gt_100k": bool(fz_target_abs_gt_100k.any()),
            "any_fz_state_gt_50k": bool(fz_state_gt_50k.any()),
            "any_fz_target_abs_gt_20k": bool(fz_target_abs_gt_20k.any()),
            "any_fz_state_gt_20k": bool(fz_state_gt_20k.any()),
            "any_fz_target_abs_gt_5k": bool(fz_target_abs_gt_5k.any()),
        }
        row["severe_force_pathology"] = bool(
            row["any_negative_fz_state"]
            or row["any_fz_state_gt_100k"]
            or row["any_fz_target_abs_gt_100k"]
        )
        row["hard_force_outlier"] = bool(
            row["any_negative_fz_state"]
            or row["any_fz_state_gt_50k"]
            or row["any_fz_target_abs_gt_20k"]
        )
        row["flat_force_outlier_strict"] = bool(
            row["any_negative_fz_state"]
            or row["any_fz_state_gt_20k"]
            or row["any_fz_target_abs_gt_5k"]
        )
        row["flat_like_score"] = float(
            1000.0 * float(row["severe_force_pathology"])
            + 100.0 * row["state_abs_z_gt_5_frac_all"]
            + 100.0 * row["target_abs_z_gt_5_frac_body7"]
            + 50.0 * row["target_abs_z_gt_5_frac_fz"]
            + 10.0 * row["fz_state_gt_20k_frac"]
            + 10.0 * row["fz_target_abs_gt_5k_frac"]
        )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bumpy_meta = load_json(args.bumpy_dir / "metadata.json")
    flat_meta = load_json(args.flat_dir / "metadata.json")
    fields = list(bumpy_meta["state_fields"])
    if fields != list(flat_meta["state_fields"]):
        raise ValueError("Flat and bumpy state fields do not match.")
    flat_norm = flat_meta["normalization"]
    flat_state_mean = np.asarray(flat_norm["state_mean"], dtype=np.float64)
    flat_state_std = np.asarray(flat_norm["state_std"], dtype=np.float64)
    flat_target_mean = np.asarray(flat_norm["target_mean"], dtype=np.float64)
    flat_target_std = np.asarray(flat_norm["target_std"], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for split in ("train", "val"):
        rows.extend(
            analyze_split(
                args.bumpy_dir,
                split,
                fields,
                flat_state_mean,
                flat_state_std,
                flat_target_mean,
                flat_target_std,
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "trajectory_metrics.csv", index=False)

    filters: dict[str, pd.Series] = {
        "A_severe_force_clean": ~df["severe_force_pathology"],
        "B_hard_force_clean": ~df["hard_force_outlier"],
        "C_strict_flat_force_clean": ~df["flat_force_outlier_strict"],
        "D_body_state_target_flat_like": (
            (~df["severe_force_pathology"])
            & (df["state_abs_z_gt_5_frac_all"] <= 0.01)
            & (df["target_abs_z_gt_5_frac_body7"] <= 0.01)
        ),
        "E_full_target_flat_like": (
            (~df["severe_force_pathology"])
            & (df["state_abs_z_gt_5_frac_all"] <= 0.01)
            & (df["target_abs_z_gt_5_frac_all"] <= 0.01)
        ),
        "F_force_and_body_flat_like": (
            (~df["flat_force_outlier_strict"])
            & (df["state_abs_z_gt_5_frac_all"] <= 0.01)
            & (df["target_abs_z_gt_5_frac_body7"] <= 0.01)
        ),
    }
    filter_summary = pd.DataFrame(
        [{"filter": name, **summarize_filter(df, mask)} for name, mask in filters.items()]
    )
    filter_summary.to_csv(args.output_dir / "filter_summary.csv", index=False)

    closest = df.sort_values("flat_like_score").copy()
    closest.head(200).to_csv(args.output_dir / "closest_flat_like_trajectories_top200.csv", index=False)
    removed_severe = df[df["severe_force_pathology"]].sort_values(
        ["fz_state_negative_count", "fz_state_gt_100k_count", "fz_target_abs_gt_100k_count"],
        ascending=False,
    )
    removed_severe.to_csv(args.output_dir / "severe_force_pathology_trajectories.csv", index=False)

    summary = {
        "bumpy_dir": str(args.bumpy_dir),
        "flat_dir": str(args.flat_dir),
        "trajectory_count": int(len(df)),
        "transition_rows": int(df["length"].sum()),
        "counts": {
            "any_negative_fz_state": bool_count(df, "any_negative_fz_state"),
            "any_fz_state_gt_100k": bool_count(df, "any_fz_state_gt_100k"),
            "any_fz_target_abs_gt_100k": bool_count(df, "any_fz_target_abs_gt_100k"),
            "severe_force_pathology": bool_count(df, "severe_force_pathology"),
            "any_fz_state_gt_50k": bool_count(df, "any_fz_state_gt_50k"),
            "any_fz_target_abs_gt_20k": bool_count(df, "any_fz_target_abs_gt_20k"),
            "hard_force_outlier": bool_count(df, "hard_force_outlier"),
            "any_fz_state_gt_20k": bool_count(df, "any_fz_state_gt_20k"),
            "any_fz_target_abs_gt_5k": bool_count(df, "any_fz_target_abs_gt_5k"),
            "flat_force_outlier_strict": bool_count(df, "flat_force_outlier_strict"),
        },
        "filters": {
            row["filter"]: {
                key: (float(value) if isinstance(value, np.floating) else int(value) if isinstance(value, np.integer) else value)
                for key, value in row.items()
                if key != "filter"
            }
            for row in filter_summary.to_dict("records")
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    by_scenario = (
        df.groupby("scenario_family")
        .agg(
            trajectories=("episode_id", "count"),
            negative_fz=("any_negative_fz_state", "sum"),
            fz_state_gt_100k=("any_fz_state_gt_100k", "sum"),
            fz_target_abs_gt_100k=("any_fz_target_abs_gt_100k", "sum"),
            severe_force_pathology=("severe_force_pathology", "sum"),
            hard_force_outlier=("hard_force_outlier", "sum"),
            strict_flat_force_outlier=("flat_force_outlier_strict", "sum"),
            median_flat_like_score=("flat_like_score", "median"),
        )
        .reset_index()
        .sort_values("severe_force_pathology", ascending=False)
    )
    by_scenario.to_csv(args.output_dir / "scenario_summary.csv", index=False)

    top_severe = df.sort_values(
        ["severe_force_pathology", "fz_state_max", "fz_target_abs_max"],
        ascending=False,
    ).head(20)
    closest_preview = closest.head(20)

    lines: list[str] = [
        "# HMMWV Bumpy Trajectory Cleaning Analysis",
        "",
        "This report counts bumpy-terrain trajectories with negative tire normal forces, very large Fz state spikes, and large Fz delta-target spikes. It also defines several candidate filters for selecting a bumpy subset closer to the flat-terrain distribution.",
        "",
        "## Inputs",
        "",
        f"- Bumpy processed cache: `{args.bumpy_dir}`",
        f"- Flat reference cache: `{args.flat_dir}`",
        f"- Trajectories: `{len(df)}`",
        f"- Transition rows: `{int(df['length'].sum())}`",
        "",
        "## Key Counts",
        "",
        f"- Trajectories with any negative Fz state: `{summary['counts']['any_negative_fz_state']}` / `{len(df)}`",
        f"- Trajectories with any Fz state > 100k N: `{summary['counts']['any_fz_state_gt_100k']}` / `{len(df)}`",
        f"- Trajectories with any |delta Fz| target > 100k N: `{summary['counts']['any_fz_target_abs_gt_100k']}` / `{len(df)}`",
        f"- Severe force pathology, defined as any of the three above: `{summary['counts']['severe_force_pathology']}` / `{len(df)}`",
        f"- Trajectories with any Fz state > 50k N: `{summary['counts']['any_fz_state_gt_50k']}` / `{len(df)}`",
        f"- Trajectories with any |delta Fz| target > 20k N: `{summary['counts']['any_fz_target_abs_gt_20k']}` / `{len(df)}`",
        f"- Hard force outlier, defined as negative Fz or Fz state > 50k N or |delta Fz| > 20k N: `{summary['counts']['hard_force_outlier']}` / `{len(df)}`",
        f"- Trajectories with any Fz state > 20k N: `{summary['counts']['any_fz_state_gt_20k']}` / `{len(df)}`",
        f"- Trajectories with any |delta Fz| target > 5k N: `{summary['counts']['any_fz_target_abs_gt_5k']}` / `{len(df)}`",
        f"- Strict flat-force outlier, defined as negative Fz or Fz state > 20k N or |delta Fz| > 5k N: `{summary['counts']['flat_force_outlier_strict']}` / `{len(df)}`",
        "",
        "## Candidate Filters",
        "",
        "These are candidate selection rules. `retained` means the trajectory remains in the cleaned bumpy subset; `removed` means filtered out.",
        "",
    ]
    lines.extend(
        markdown_table(
            filter_summary,
            [
                "filter",
                "retained_trajectories",
                "removed_trajectories",
                "retained_trajectory_frac",
                "retained_transition_frac",
            ],
            ["filter", "retained traj", "removed traj", "retained traj frac", "retained row frac"],
        )
    )
    lines.extend(
        [
            "",
            "Filter definitions:",
            "",
            "- `A_severe_force_clean`: remove any trajectory with negative Fz state, Fz state > 100k N, or |delta Fz| > 100k N.",
            "- `B_hard_force_clean`: remove any trajectory with negative Fz state, Fz state > 50k N, or |delta Fz| > 20k N.",
            "- `C_strict_flat_force_clean`: remove any trajectory with negative Fz state, Fz state > 20k N, or |delta Fz| > 5k N.",
            "- `D_body_state_target_flat_like`: keep non-severe trajectories with <=1% of all state values outside |flat z|>5 and <=1% of body-7 target values outside |flat z|>5. This ignores Fz target mismatch except for severe spikes.",
            "- `E_full_target_flat_like`: same as D, but also requires all 15 target channels, including Fz, to have <=1% outside |flat z|>5.",
            "- `F_force_and_body_flat_like`: combines strict flat-force clean with the body-state/body-target flat-like criteria.",
            "",
            "## Scenario Summary",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            by_scenario,
            [
                "scenario_family",
                "trajectories",
                "negative_fz",
                "fz_state_gt_100k",
                "fz_target_abs_gt_100k",
                "severe_force_pathology",
                "hard_force_outlier",
                "strict_flat_force_outlier",
                "median_flat_like_score",
            ],
            [
                "scenario",
                "traj",
                "neg Fz",
                "state >100k",
                "|dFz| >100k",
                "severe",
                "hard",
                "strict",
                "median score",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Top Severe/Spiky Trajectories",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            top_severe,
            [
                "split",
                "episode_id",
                "scenario_family",
                "fz_state_min",
                "fz_state_max",
                "fz_target_abs_max",
                "fz_state_negative_count",
                "fz_state_gt_100k_count",
                "fz_target_abs_gt_100k_count",
            ],
            [
                "split",
                "episode",
                "scenario",
                "min Fz",
                "max Fz",
                "max |dFz|",
                "neg count",
                "state >100k count",
                "|dFz| >100k count",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Closest Flat-Like Trajectory Preview",
            "",
            "`flat_like_score` is a ranking heuristic: lower is closer to flat. It penalizes severe force pathology, all-state |flat z|>5 rate, body target |flat z|>5 rate, Fz target |flat z|>5 rate, Fz state >20k rate, and |delta Fz| >5k rate.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            closest_preview,
            [
                "split",
                "episode_id",
                "scenario_family",
                "flat_like_score",
                "state_abs_z_gt_5_frac_all",
                "target_abs_z_gt_5_frac_body7",
                "target_abs_z_gt_5_frac_fz",
                "fz_state_max",
                "fz_target_abs_max",
            ],
            [
                "split",
                "episode",
                "scenario",
                "score",
                "state z>5 frac",
                "body target z>5 frac",
                "Fz target z>5 frac",
                "max Fz",
                "max |dFz|",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `trajectory_metrics.csv`: per-trajectory metrics and flags.",
            "- `filter_summary.csv`: retained/removed counts for the candidate filters.",
            "- `scenario_summary.csv`: counts by scenario family.",
            "- `severe_force_pathology_trajectories.csv`: trajectories removed by the severe force pathology filter.",
            "- `closest_flat_like_trajectories_top200.csv`: lowest-score trajectories under the flat-like heuristic.",
            "- `summary.json`: machine-readable summary.",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines))
    print(args.output_dir)


if __name__ == "__main__":
    main()
