from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write sharded configs for large HMMWV data generation.")
    parser.add_argument("--plan-dir", type=Path, default=Path("artifacts/datasets/hmmwv_turn_300g_plan"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/datasets/hmmwv_turn_300g_shards"))
    parser.add_argument("--num-shards", type=int, default=96)
    parser.add_argument("--episodes-per-shard", type=int, default=1000)
    return parser.parse_args()


def family_counts(total: int) -> dict[str, int]:
    counts = {
        "multi_steer": int(total * 0.30),
        "sustained_turn": int(total * 0.20),
        "sine_steer": int(total * 0.15),
        "chirp_steer": int(total * 0.15),
        "doublet_steer": int(total * 0.10),
    }
    counts["steer_brake"] = total - sum(counts.values())
    return counts


def speed_band(shard_index: int) -> dict[str, Any]:
    bands = [
        {
            "name": "low",
            "throttle_peak_range": [0.14, 0.30],
            "steering_amplitude_range": [0.20, 0.90],
            "duration_s_range": [42.0, 70.0],
        },
        {
            "name": "medium",
            "throttle_peak_range": [0.24, 0.48],
            "steering_amplitude_range": [0.12, 0.75],
            "duration_s_range": [38.0, 66.0],
        },
        {
            "name": "fast",
            "throttle_peak_range": [0.38, 0.70],
            "steering_amplitude_range": [0.05, 0.42],
            "duration_s_range": [34.0, 58.0],
        },
        {
            "name": "mixed",
            "throttle_peak_range": [0.10, 0.62],
            "steering_amplitude_range": [0.05, 0.82],
            "duration_s_range": [36.0, 68.0],
        },
    ]
    return bands[shard_index % len(bands)]


def base_config(dataset_name: str, output_subdir: str, seed: int) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "chrono_data_root": "chrono/data",
        "vehicle_data_root": "chrono/data/vehicle",
        "output_subdir": output_subdir,
        "simulation": {
            "step_size_s": 0.002,
            "tire_step_size_s": 0.001,
            "record_step_s": 0.01,
            "driver_sample_step_s": 0.01,
            "validation_ratio": 0.2,
        },
        "vehicle": {
            "model": "HMMWV_Full",
            "contact_method": "SMC",
            "chassis_fixed": False,
            "init": {"x_m": 0.0, "y_m": 0.0, "z_m": 1.6},
            "engine_model": "SHAFTS",
            "transmission_model": "AUTOMATIC_SHAFTS",
            "drive_type": "AWD",
            "steering_type": "PITMAN_ARM",
            "tire_model": "TMEASY",
        },
        "terrain": {
            "type": "rigid",
            "length_m": 900.0,
            "width_m": 900.0,
            "friction": 0.9,
            "restitution": 0.01,
            "young_modulus_pa": 20000000.0,
        },
        "logging": {"include_tire_channels": False},
        "scenario_generator": {
            "seed": seed,
            "shuffle_seed": seed + 1,
            "warmup_s": 2.5,
            "families": [],
        },
    }


def family_config(
    shard_index: int,
    family_name: str,
    count: int,
    template: str,
    band: dict[str, Any],
) -> dict[str, Any]:
    prefix = f"s{shard_index:03d}_{family_name}"
    cfg: dict[str, Any] = {
        "name": family_name,
        "template": template,
        "family_label": family_name,
        "scenario_prefix": prefix,
        "count": count,
        "duration_s_range": band["duration_s_range"],
        "throttle_peak_range": band["throttle_peak_range"],
        "steering_amplitude_range": band["steering_amplitude_range"],
        "throttle_rise_s_range": [0.4, 1.8],
        "throttle_hold_s_range": [2.0, 8.0],
        "coast_probability": 0.25,
    }
    if family_name == "multi_steer":
        cfg.update(
            {
                "event_count_range": [7, 16],
                "event_gap_s_range": [0.3, 2.8],
                "steer_rise_s_range": [0.12, 0.8],
                "steer_hold_s_range": [0.35, 3.2],
                "steer_return_s_range": [0.12, 0.8],
                "reverse_pulse_probability": 0.45,
                "brake_pulse_count_range": [0, 4],
                "brake_peak_range": [0.10, 0.60],
            }
        )
    elif family_name == "sustained_turn":
        cfg.update(
            {
                "steer_start_offset_s_range": [0.6, 2.5],
                "steer_rise_s_range": [0.2, 1.2],
                "steer_hold_s_range": [10.0, 34.0],
                "steer_return_s_range": [0.4, 1.6],
            }
        )
    elif family_name == "sine_steer":
        cfg.update({"frequency_hz_range": [0.08, 0.9], "end_margin_s_range": [0.5, 2.0]})
    elif family_name == "chirp_steer":
        cfg.update(
            {
                "start_frequency_hz_range": [0.05, 0.25],
                "end_frequency_hz_range": [0.45, 1.45],
                "end_margin_s_range": [0.5, 2.0],
            }
        )
    elif family_name == "doublet_steer":
        cfg.update(
            {
                "pulse_width_s_range": [0.4, 1.8],
                "pulse_gap_s_range": [0.1, 0.9],
                "second_pulse_ratio_range": [0.7, 1.35],
            }
        )
    elif family_name == "steer_brake":
        cfg.update({"brake_peak_range": [0.12, 0.65], "steer_hold_s_range": [1.0, 5.0]})
    return cfg


def write_configs(plan_dir: Path, output_root: Path, num_shards: int, episodes_per_shard: int) -> list[Path]:
    config_dir = plan_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    counts = family_counts(episodes_per_shard)
    templates = {
        "multi_steer": "multi_steer",
        "sustained_turn": "step_steer",
        "sine_steer": "sine_steer",
        "chirp_steer": "chirp_steer",
        "doublet_steer": "doublet_steer",
        "steer_brake": "steer_brake",
    }

    manifest: list[dict[str, Any]] = []
    for shard_index in range(num_shards):
        dataset_name = f"hmmwv_turn_300g_s{shard_index:03d}"
        output_subdir = str(output_root / f"shard_{shard_index:03d}")
        cfg = base_config(dataset_name, output_subdir, seed=2026052300 + 17 * shard_index)
        band = speed_band(shard_index)
        cfg["scenario_generator"]["families"] = [
            family_config(shard_index, family_name, count, templates[family_name], band)
            for family_name, count in counts.items()
        ]

        path = config_dir / f"shard_{shard_index:03d}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        paths.append(path)
        manifest.append(
            {
                "shard_index": shard_index,
                "dataset_name": dataset_name,
                "config_path": str(path),
                "output_subdir": output_subdir,
                "speed_band": band["name"],
                "episodes": episodes_per_shard,
                "families": counts,
            }
        )

    (plan_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def main() -> int:
    args = parse_args()
    paths = write_configs(
        plan_dir=args.plan_dir,
        output_root=args.output_root,
        num_shards=args.num_shards,
        episodes_per_shard=args.episodes_per_shard,
    )
    print(f"wrote {len(paths)} shard configs under {args.plan_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
