"""Write an HMMWV CRM collection config.

The scenario sampling reuses the rigid-terrain generator family mix and
profile templates: multi-steer, sustained turn, sine steer, chirp steer,
doublet steer, and steer+brake.  Durations are shortened for finite CRM terrain
cost and boundary safety; the driver command mechanism remains the same
profile -> DataDriverEntry -> ChDataDriver path used by the rigid collector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prepare_hmmwv_300g_generation import family_config, family_counts


SEED_BASE = 2026061900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=Path("artifacts/datasets/hmmwv_crm_100_plan"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/datasets/hmmwv_crm_100"))
    parser.add_argument("--dataset-name", type=str, default="hmmwv_crm_100")
    parser.add_argument("--config-name", type=str, default="crm100")
    parser.add_argument("--scenario-prefix-root", type=str, default="crm100")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--duration-min-s", type=float, default=12.0)
    parser.add_argument("--duration-max-s", type=float, default=18.0)
    parser.add_argument("--terrain-length-m", type=float, default=150.0)
    parser.add_argument("--terrain-width-m", type=float, default=150.0)
    parser.add_argument("--crm-spacing-m", type=float, default=0.08)
    parser.add_argument("--boundary-margin-m", type=float, default=5.0)
    parser.add_argument(
        "--chrono-threads",
        type=int,
        default=12,
        help="Number of Chrono MBD threads to use during CRM collection.",
    )
    parser.add_argument(
        "--chrono-data-root",
        type=str,
        default=None,
        help="Override Chrono data root, e.g. $CONDA_PREFIX/share/chrono/data.",
    )
    return parser.parse_args()


def base_config(args: argparse.Namespace) -> dict[str, Any]:
    chrono_data_root = args.chrono_data_root or "chrono/data"
    return {
        "dataset_name": str(args.dataset_name),
        "chrono_data_root": chrono_data_root,
        "vehicle_data_root": str(Path(chrono_data_root) / "vehicle") if args.chrono_data_root else "chrono/data/vehicle",
        "output_subdir": str(args.output_dir),
        "simulation": {
            "step_size_s": 5e-4,
            "tire_step_size_s": 5e-4,
            "record_step_s": 0.01,
            "driver_sample_step_s": 0.01,
            "validation_ratio": 0.2,
            "chrono_threads": int(args.chrono_threads),
            "collision_threads": 1,
            "eigen_threads": 1,
        },
        "vehicle": {
            "model": "HMMWV_Full",
            "contact_method": "SMC",
            "chassis_fixed": False,
            "init": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.7},
            "engine_model": "SHAFTS",
            "transmission_model": "AUTOMATIC_SHAFTS",
            "drive_type": "AWD",
            "steering_type": "PITMAN_ARM",
            "tire_model": "RIGID_MESH",
        },
        "terrain": {
            "type": "crm",
            "length_m": float(args.terrain_length_m),
            "width_m": float(args.terrain_width_m),
            "depth_m": 0.25,
            "center_m": [0.0, 0.0, 0.0],
            "boundary_margin_m": float(args.boundary_margin_m),
            "initial_spacing_m": float(args.crm_spacing_m),
            "active_domain_m": [2.0, 2.0, 1.0],
            "active_domain_delay_s": 0.1,
            "soil": {
                "density": 1700.0,
                "cohesion": 5000.0,
                "friction": 0.8,
                "young_modulus_pa": 1_000_000.0,
                "poisson_ratio": 0.3,
                "mu_I0": 0.04,
                "average_diam_m": 0.005,
            },
            "sph": {
                "integration_scheme": "RK2",
                "d0_multiplier": 1.0,
                "free_surface_threshold": 2.0,
                "artificial_viscosity": 0.5,
                "shifting_method": "NONE",
                "shifting_ppst_push": 1.0,
                "shifting_ppst_pull": 1.0,
                "viscosity_method": "ARTIFICIAL_BILATERAL",
                "boundary_method": "ADAMI",
                "num_proximity_search_steps": 4,
            },
        },
        "logging": {"include_tire_channels": True, "tire_force_source": "crm_fsi"},
        "scenario_generator": {
            "seed": SEED_BASE,
            "shuffle_seed": SEED_BASE + 1,
            "warmup_s": 0.2,
            "families": [],
        },
    }


def crm_band(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": "crm_mixed",
        "throttle_peak_range": [0.10, 0.62],
        "steering_amplitude_range": [0.05, 0.82],
        "duration_s_range": [float(args.duration_min_s), float(args.duration_max_s)],
    }


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = base_config(args)
    counts = family_counts(int(args.episodes))
    templates = {
        "multi_steer": "multi_steer",
        "sustained_turn": "step_steer",
        "sine_steer": "sine_steer",
        "chirp_steer": "chirp_steer",
        "doublet_steer": "doublet_steer",
        "steer_brake": "steer_brake",
    }
    band = crm_band(args)
    families = []
    for family_name, count in counts.items():
        if count <= 0:
            continue
        family = family_config(0, family_name, count, templates[family_name], band)
        family["scenario_prefix"] = f"{args.scenario_prefix_root}_{family_name}"
        family["family_label"] = family_name
        families.append(family)
    cfg["scenario_generator"]["families"] = families
    return cfg


def main() -> int:
    args = parse_args()
    config_dir = args.plan_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args)
    config_name = args.config_name[:-5] if args.config_name.endswith(".json") else args.config_name
    config_path = config_dir / f"{config_name}.json"
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    manifest = {
        "dataset_name": cfg["dataset_name"],
        "config_path": str(config_path),
        "output_subdir": cfg["output_subdir"],
        "episodes": int(args.episodes),
        "families": family_counts(int(args.episodes)),
        "terrain": cfg["terrain"],
    }
    (args.plan_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote CRM config: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
