"""Write shard configs for the 300 GB rigid-terrain tire-force dataset.

Scaled-up version of the 10 GB tire-force set (`prepare_hmmwv_tire10g_
generation.py`): identical family mix, parameter ranges, and speed-band
rotation, just 32x more shards.

- 128 shards rotating through the low / medium / fast / mixed bands
  (32 shards per band), 256 episodes each -> ~2.4 GB per shard, ~307 GB total
- `logging.include_tire_channels: true`
- fresh seed base, so episodes are new draws: no overlap with either the
  10 GB tire set or the original 300 GB (no-tire-channel) pool

No smoke shard here; use the 10 GB pipeline's smoke config for checkout, it
covers the same distribution.

For clusters without the chrono source checkout, pass
``--chrono-data-root "$CONDA_PREFIX/share/chrono/data"``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prepare_hmmwv_300g_generation import base_config, family_config, family_counts, speed_band

SEED_BASE = 2026061300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=Path("artifacts/datasets/hmmwv_tire_rigid_300g_plan"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/datasets/hmmwv_tire_rigid_300g_shards"))
    parser.add_argument("--num-shards", type=int, default=128)
    parser.add_argument("--episodes-per-shard", type=int, default=256)
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
        family["scenario_prefix"] = f"t300_s{shard_index:03d}_{family_name}"
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
        dataset_name = f"hmmwv_tire_rigid_300g_s{shard_index:03d}"
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

    (args.plan_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.num_shards} shard configs under {args.plan_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
