from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare flat-vs-bumpy HMMWV processed-cache channel distributions "
            "for the normal-force/omega dynamics state."
        )
    )
    parser.add_argument(
        "--flat-dir",
        type=Path,
        default=Path("artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1"),
    )
    parser.add_argument(
        "--bumpy-dir",
        type=Path,
        default=Path("artifacts/training_datasets/hmmwv_bumpy_10g_normal_force_omega_seq_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis/hmmwv_flat_vs_bumpy_15d_distribution"),
    )
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument(
        "--sample-per-split",
        type=int,
        default=1_000_000,
        help="Maximum rows to sample from each split and dataset.",
    )
    parser.add_argument(
        "--arrays",
        nargs="+",
        default=["states", "targets"],
        choices=["states", "targets"],
        help="Processed arrays to compare. targets are delta-state labels.",
    )
    parser.add_argument("--hist-bins", type=int, default=180)
    return parser.parse_args()


def load_metadata(root: Path) -> dict:
    return json.loads((root / "metadata.json").read_text())


def validate_compatible(flat: DatasetSpec, bumpy: DatasetSpec) -> list[str]:
    flat_meta = load_metadata(flat.root)
    bumpy_meta = load_metadata(bumpy.root)
    flat_fields = list(flat_meta["state_fields"])
    bumpy_fields = list(bumpy_meta["state_fields"])
    if flat_fields != bumpy_fields:
        raise ValueError(f"State fields differ:\nflat={flat_fields}\nbumpy={bumpy_fields}")
    return flat_fields


def sample_rows(
    root: Path,
    array_kind: str,
    sample_per_split: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for split_index, split in enumerate(("train", "val")):
        path = root / f"{split}_{array_kind}.npy"
        arr = np.load(path, mmap_mode="r")
        n_rows = int(arr.shape[0])
        take = min(sample_per_split, n_rows)
        if take == n_rows:
            indices = np.arange(n_rows)
        else:
            split_rng = np.random.default_rng(seed + 1009 * (split_index + 1))
            indices = split_rng.choice(n_rows, size=take, replace=False)
            indices.sort()
        # Materialize this split as float64 for stable downstream statistics.
        samples.append(np.asarray(arr[indices], dtype=np.float64))
        counts[f"{split}_rows_total"] = n_rows
        counts[f"{split}_rows_sampled"] = int(take)
    rng.shuffle(samples)
    return np.concatenate(samples, axis=0), counts


def quantile_dict(values: np.ndarray) -> dict[str, float]:
    quantiles = [0.0, 0.001, 0.005, 0.01, 0.05, 0.50, 0.95, 0.99, 0.995, 0.999, 1.0]
    names = ["min", "p00_1", "p00_5", "p01", "p05", "p50", "p95", "p99", "p99_5", "p99_9", "max"]
    computed = np.quantile(values, quantiles)
    return {name: float(value) for name, value in zip(names, computed, strict=True)}


def population_stability_index(flat_values: np.ndarray, bumpy_values: np.ndarray, bins: int = 20) -> float:
    edges = np.quantile(flat_values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    flat_counts, _ = np.histogram(flat_values, bins=edges)
    bumpy_counts, _ = np.histogram(bumpy_values, bins=edges)
    eps = 1e-6
    flat_p = np.maximum(flat_counts / max(flat_counts.sum(), 1), eps)
    bumpy_p = np.maximum(bumpy_counts / max(bumpy_counts.sum(), 1), eps)
    return float(np.sum((bumpy_p - flat_p) * np.log(bumpy_p / flat_p)))


def distribution_rows(
    fields: Iterable[str],
    flat_sample: np.ndarray,
    bumpy_sample: np.ndarray,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for i, field in enumerate(fields):
        flat_values = flat_sample[:, i]
        bumpy_values = bumpy_sample[:, i]
        flat_stats = quantile_dict(flat_values)
        bumpy_stats = quantile_dict(bumpy_values)
        flat_mean = float(np.mean(flat_values))
        flat_std = float(np.std(flat_values))
        bumpy_mean = float(np.mean(bumpy_values))
        bumpy_std = float(np.std(bumpy_values))
        flat_p005, flat_p995 = flat_stats["p00_5"], flat_stats["p99_5"]
        flat_p01, flat_p99 = flat_stats["p01"], flat_stats["p99"]
        flat_p05, flat_p95 = flat_stats["p05"], flat_stats["p95"]
        bumpy_outside_005_995 = float(
            np.mean((bumpy_values < flat_p005) | (bumpy_values > flat_p995))
        )
        bumpy_outside_01_99 = float(np.mean((bumpy_values < flat_p01) | (bumpy_values > flat_p99)))
        bumpy_outside_05_95 = float(np.mean((bumpy_values < flat_p05) | (bumpy_values > flat_p95)))
        denom = flat_std if flat_std > 1e-12 else 1.0
        row: dict[str, float | str] = {
            "channel": field,
            "flat_mean": flat_mean,
            "bumpy_mean": bumpy_mean,
            "mean_shift_in_flat_sigma": (bumpy_mean - flat_mean) / denom,
            "flat_std": flat_std,
            "bumpy_std": bumpy_std,
            "std_ratio_bumpy_over_flat": bumpy_std / denom,
            "bumpy_outside_flat_p00_5_p99_5_frac": bumpy_outside_005_995,
            "bumpy_outside_flat_p01_p99_frac": bumpy_outside_01_99,
            "bumpy_outside_flat_p05_p95_frac": bumpy_outside_05_95,
            "psi_flat_quantile_bins": population_stability_index(flat_values, bumpy_values),
            "flat_nonfinite_frac": float(np.mean(~np.isfinite(flat_values))),
            "bumpy_nonfinite_frac": float(np.mean(~np.isfinite(bumpy_values))),
            "flat_exact_zero_frac": float(np.mean(flat_values == 0.0)),
            "bumpy_exact_zero_frac": float(np.mean(bumpy_values == 0.0)),
        }
        for key, value in flat_stats.items():
            row[f"flat_{key}"] = value
        for key, value in bumpy_stats.items():
            row[f"bumpy_{key}"] = value
        rows.append(row)
    return rows


def plot_histograms(
    fields: list[str],
    flat_sample: np.ndarray,
    bumpy_sample: np.ndarray,
    title: str,
    output: Path,
    bins: int,
) -> None:
    n_cols = 3
    n_rows = int(np.ceil(len(fields) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.4 * n_rows), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for i, field in enumerate(fields):
        ax = axes_flat[i]
        flat_values = flat_sample[:, i]
        bumpy_values = bumpy_sample[:, i]
        lo = float(min(np.quantile(flat_values, 0.001), np.quantile(bumpy_values, 0.001)))
        hi = float(max(np.quantile(flat_values, 0.999), np.quantile(bumpy_values, 0.999)))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = float(min(flat_values.min(), bumpy_values.min()))
            hi = float(max(flat_values.max(), bumpy_values.max()))
        ax.hist(flat_values, bins=bins, range=(lo, hi), density=True, alpha=0.55, label="flat")
        ax.hist(bumpy_values, bins=bins, range=(lo, hi), density=True, alpha=0.55, label="bumpy")
        ax.set_title(field, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(fields) :]:
        ax.axis("off")
    axes_flat[0].legend(loc="best", fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_ood_bars(summary: pd.DataFrame, title: str, output: Path) -> None:
    ordered = summary.sort_values("psi_flat_quantile_bins", ascending=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.barh(y, ordered["psi_flat_quantile_bins"], color="#4c78a8", label="PSI")
    ax.set_yticks(y, ordered["channel"])
    ax.set_xlabel("Population stability index vs flat quantile bins")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def format_float(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1000.0 or (abs_value > 0 and abs_value < 0.001):
        return f"{value:.3e}"
    return f"{value:.4f}"


def write_markdown_report(
    output: Path,
    fields: list[str],
    metadata: dict,
    state_summary: pd.DataFrame | None,
    target_summary: pd.DataFrame | None,
) -> None:
    lines = [
        "# HMMWV Flat vs Bumpy 15D Distribution Analysis",
        "",
        "Compared processed-cache distributions for the shared `tire_normal_force_omega` state fields.",
        "",
        "## Inputs",
        "",
        f"- Flat cache: `{metadata['flat_dir']}`",
        f"- Bumpy cache: `{metadata['bumpy_dir']}`",
        f"- Seed: `{metadata['seed']}`",
        f"- Sample per split cap: `{metadata['sample_per_split']}`",
        f"- Flat sampled rows: `{metadata['flat_counts']['train_rows_sampled'] + metadata['flat_counts']['val_rows_sampled']}`",
        f"- Bumpy sampled rows: `{metadata['bumpy_counts']['train_rows_sampled'] + metadata['bumpy_counts']['val_rows_sampled']}`",
        "",
        "## Channels",
        "",
    ]
    lines.extend(f"{i}. `{field}`" for i, field in enumerate(fields))
    lines.extend(
        [
            "",
            "## How to Read the Tables",
            "",
            "- `mean_shift_in_flat_sigma`: bumpy mean minus flat mean, divided by flat std.",
            "- `std_ratio_bumpy_over_flat`: bumpy std divided by flat std.",
            "- `bumpy_outside_flat_p00_5_p99_5_frac`: fraction of bumpy samples outside the flat 0.5%-99.5% envelope. Flat would be about 1% by construction.",
            "- `psi_flat_quantile_bins`: population stability index using 20 flat quantile bins; larger means stronger distribution shift.",
            "",
        ]
    )

    def add_top_table(title: str, summary: pd.DataFrame) -> None:
        top = summary.sort_values("psi_flat_quantile_bins", ascending=False).head(15)
        columns = [
            "channel",
            "psi_flat_quantile_bins",
            "bumpy_outside_flat_p00_5_p99_5_frac",
            "mean_shift_in_flat_sigma",
            "std_ratio_bumpy_over_flat",
            "flat_p50",
            "bumpy_p50",
            "flat_p99",
            "bumpy_p99",
        ]
        lines.extend([f"## {title}", ""])
        lines.append(
            "| channel | PSI | bumpy outside flat 0.5-99.5% | mean shift sigma | std ratio | flat p50 | bumpy p50 | flat p99 | bumpy p99 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in top[columns].iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['channel']}`",
                        format_float(float(row["psi_flat_quantile_bins"])),
                        format_float(float(row["bumpy_outside_flat_p00_5_p99_5_frac"])),
                        format_float(float(row["mean_shift_in_flat_sigma"])),
                        format_float(float(row["std_ratio_bumpy_over_flat"])),
                        format_float(float(row["flat_p50"])),
                        format_float(float(row["bumpy_p50"])),
                        format_float(float(row["flat_p99"])),
                        format_float(float(row["bumpy_p99"])),
                    ]
                )
                + " |"
            )
        lines.append("")

    if state_summary is not None:
        add_top_table("State Distribution Shift Ranking", state_summary)
        lines.extend(
            [
                "State plots:",
                "",
                "- `states_histograms.png`",
                "- `states_ood_rank.png`",
                "",
            ]
        )
    if target_summary is not None:
        add_top_table("Delta Target Distribution Shift Ranking", target_summary)
        lines.extend(
            [
                "Target plots:",
                "",
                "- `targets_histograms.png`",
                "- `targets_ood_rank.png`",
                "",
            ]
        )
    lines.extend(
        [
            "Full numeric tables are in `states_summary.csv` and `targets_summary.csv` when generated.",
            "",
        ]
    )
    output.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat = DatasetSpec(name="flat", root=args.flat_dir)
    bumpy = DatasetSpec(name="bumpy", root=args.bumpy_dir)
    fields = validate_compatible(flat, bumpy)
    metadata: dict = {
        "flat_dir": str(flat.root),
        "bumpy_dir": str(bumpy.root),
        "seed": args.seed,
        "sample_per_split": args.sample_per_split,
        "fields": fields,
        "arrays": args.arrays,
    }

    summaries: dict[str, pd.DataFrame] = {}
    first_counts_recorded = False
    for array_index, array_kind in enumerate(args.arrays):
        flat_sample, flat_counts = sample_rows(
            flat.root,
            array_kind,
            sample_per_split=args.sample_per_split,
            seed=args.seed + 7919 * array_index,
        )
        bumpy_sample, bumpy_counts = sample_rows(
            bumpy.root,
            array_kind,
            sample_per_split=args.sample_per_split,
            seed=args.seed + 17 + 7919 * array_index,
        )
        if not first_counts_recorded:
            metadata["flat_counts"] = flat_counts
            metadata["bumpy_counts"] = bumpy_counts
            first_counts_recorded = True
        rows = distribution_rows(fields, flat_sample, bumpy_sample)
        summary = pd.DataFrame(rows)
        summary = summary.sort_values("psi_flat_quantile_bins", ascending=False)
        summary.to_csv(args.output_dir / f"{array_kind}_summary.csv", index=False)
        summaries[array_kind] = summary
        plot_histograms(
            fields=fields,
            flat_sample=flat_sample,
            bumpy_sample=bumpy_sample,
            title=f"HMMWV {array_kind}: flat vs bumpy sampled distributions",
            output=args.output_dir / f"{array_kind}_histograms.png",
            bins=args.hist_bins,
        )
        plot_ood_bars(
            summary=summary,
            title=f"HMMWV {array_kind}: OOD rank vs flat",
            output=args.output_dir / f"{array_kind}_ood_rank.png",
        )

    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    write_markdown_report(
        output=args.output_dir / "README.md",
        fields=fields,
        metadata=metadata,
        state_summary=summaries.get("states"),
        target_summary=summaries.get("targets"),
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
