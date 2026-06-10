"""Write shard configs for the 10 GB rigid-terrain tire-force dataset.

Reuses the family mix, parameter ranges, and speed-band rotation of the 300 GB
generation (`prepare_hmmwv_300g_generation.py`) so the new dataset covers the
same excitation distribution, just less densely:

- 4 shards, one per speed band (low / medium / fast / mixed)
- 256 episodes per shard by default (~10 GB raw with tire channels enabled,
  which roughly 2.7x the per-row width of the 300 GB datasets)
- `logging.include_tire_channels: true` (the point of this dataset)
- fresh seed base so episodes are new draws, not repeats of the 300 GB pool

Also writes a 12-episode smoke shard (mixed band) for local/cluster checkout.

For clusters without the chrono source checkout, pass
``--chrono-data-root "$CONDA_PREFIX/share/chrono/data"`` (the official pychrono
conda package ships the full data tree there).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prepare_hmmwv_300g_generation import base_config, family_config, family_counts, speed_band

SEED_BASE = 2026061100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=Path("artifacts/datasets/hmmwv_tire_rigid_10g_plan"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/datasets/hmmwv_tire_rigid_10g_shards"))
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--episodes-per-shard", type=int, default=256)
    parser.add_argument("--smoke-episodes", type=int, default=12)
    parser.add_argument(
        "--chrono-data-root",
        type=str,
        default=None,
        help="Override chrono data root (e.g. $CONDA_PREFIX/share/chrono/data on a cluster).",
    )
    return parser.parse_args()


def shard_config(
    shard_index: int,
    dataset_name: str,
    output_subdir: str,
    episodes: int,
    chrono_data_root: str | None,
) -> dict[str, Any]:
    cfg = base_config(dataset_name, output_subdir, seed=SEED_BASE + 17 * shard_index)
    cfg["logging"]["include_tire_channels"] = True
    if chrono_data_root:
        cfg["chrono_data_root"] = chrono_data_root
        cfg["vehicle_data_root"] = str(Path(chrono_data_root) / "vehicle")

    templates = {
        "multi_steer": "multi_steer",
        "sustained_turn": "step_steer",
        "sine_steer": "sine_steer",
        "chirp_steer": "chirp_steer",
        "doublet_steer": "doublet_steer",
        "steer_brake": "steer_brake",
    }
    band = speed_band(shard_index)
    families = []
    for family_name, count in family_counts(episodes).items():
        if count <= 0:
            continue
        family = family_config(shard_index, family_name, count, templates[family_name], band)
        family["scenario_prefix"] = f"t10_s{shard_index:03d}_{family_name}"
        families.append(family)
    cfg["scenario_generator"]["families"] = families
    return cfg


def main() -> int:
    args = parse_args()
    config_dir = args.plan_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for shard_index in range(args.num_shards):
        dataset_name = f"hmmwv_tire_rigid_10g_s{shard_index:03d}"
        output_subdir = str(args.output_root / f"shard_{shard_index:03d}")
        cfg = shard_config(
            shard_index, dataset_name, output_subdir, args.episodes_per_shard, args.chrono_data_root
        )
        path = config_dir / f"shard_{shard_index:03d}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        manifest.append(
            {
                "shard_index": shard_index,
                "dataset_name": dataset_name,
                "config_path": str(path),
                "output_subdir": output_subdir,
                "speed_band": speed_band(shard_index)["name"],
                "episodes": args.episodes_per_shard,
                "families": family_counts(args.episodes_per_shard),
            }
        )

    # smoke shard: a handful of episodes from the widest (mixed) band
    smoke_index = 3  # band rotation: 3 -> "mixed"
    smoke_cfg = shard_config(
        smoke_index,
        "hmmwv_tire_rigid_10g_smoke",
        str(args.output_root / "smoke"),
        args.smoke_episodes,
        args.chrono_data_root,
    )
    smoke_cfg["scenario_generator"]["seed"] = SEED_BASE + 9999
    smoke_cfg["scenario_generator"]["shuffle_seed"] = SEED_BASE + 10000
    for family in smoke_cfg["scenario_generator"]["families"]:
        family["scenario_prefix"] = family["scenario_prefix"].replace("t10_s003", "t10_smoke")
    smoke_path = config_dir / "smoke.json"
    smoke_path.write_text(json.dumps(smoke_cfg, indent=2) + "\n")

    (args.plan_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.num_shards} shard configs + smoke config under {args.plan_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
