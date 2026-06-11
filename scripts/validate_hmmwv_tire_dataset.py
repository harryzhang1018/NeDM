"""Validate a collected HMMWV dataset that includes tire-force channels.

Checks, over a sample of episode CSVs:

1. dataset_index.json exists and episode files match it
2. every expected tire channel column is present
3. no NaN/inf anywhere
4. median summed wheel-frame Fz is within a band around the vehicle weight
5. slip ratios are bounded for wheels in ground contact (catches sign/radius
   convention regressions; airborne wheels spin freely at near-zero hub speed,
   so their slip ratio is unbounded and only reported, not checked)

Usage:
    python scripts/validate_hmmwv_tire_dataset.py --dataset-dir <shard_dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

WHEEL_NAMES = ("tire_fl", "tire_fr", "tire_rl", "tire_rr")
# HMMWV_Full as configured in this pipeline (measured: 2573.1 kg)
DEFAULT_WEIGHT_N = 25242.0
# below this normal force the wheel is (nearly) airborne and slip is undefined
CONTACT_FZ_N = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=24, help="Episodes to sample for row checks.")
    parser.add_argument("--weight-n", type=float, default=DEFAULT_WEIGHT_N)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def expected_tire_fields() -> list[str]:
    from nedm.hmmwv_data import tire_field_names

    return tire_field_names()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir
    failures: list[str] = []

    index_path = dataset_dir / "dataset_index.json"
    if not index_path.exists():
        print(f"FAIL: missing {index_path} (shard incomplete or wrong directory)")
        return 1
    index = json.loads(index_path.read_text())
    episodes = index.get("episodes", [])
    csv_paths = sorted((dataset_dir / "episodes").glob("*.csv"))
    print(f"{dataset_dir.name}: index lists {len(episodes)} episodes, {len(csv_paths)} csv files on disk")
    if len(csv_paths) != len(episodes):
        failures.append(f"episode count mismatch: index {len(episodes)} vs disk {len(csv_paths)}")

    rng = random.Random(args.seed)
    sampled = csv_paths if len(csv_paths) <= args.max_episodes else rng.sample(csv_paths, args.max_episodes)

    tire_fields = expected_tire_fields()
    fz_cols = [f"{n}_force_wheel_fz_n" for n in WHEEL_NAMES]
    slip_cols = [f"{n}_slip_ratio" for n in WHEEL_NAMES]

    total_fz_medians = []
    slip_extreme = 0.0
    airborne_slip_extreme = 0.0
    rows_checked = 0
    for path in sampled:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(tire_fields) - set(reader.fieldnames or [])
            if missing:
                failures.append(f"{path.name}: missing tire columns, e.g. {sorted(missing)[:3]}")
                continue
            fz_sums = []
            for row in reader:
                rows_checked += 1
                for key in tire_fields:
                    value = float(row[key])
                    if not math.isfinite(value):
                        failures.append(f"{path.name}: non-finite {key} at sample {row['sample_index']}")
                        break
                fz_sums.append(sum(float(row[c]) for c in fz_cols))
                for fz_col, slip_col in zip(fz_cols, slip_cols):
                    slip = abs(float(row[slip_col]))
                    if float(row[fz_col]) > CONTACT_FZ_N:
                        slip_extreme = max(slip_extreme, slip)
                    else:
                        airborne_slip_extreme = max(airborne_slip_extreme, slip)
            if fz_sums:
                fz_sums.sort()
                total_fz_medians.append(fz_sums[len(fz_sums) // 2])

    if total_fz_medians:
        med = sorted(total_fz_medians)[len(total_fz_medians) // 2]
        ratio = med / args.weight_n
        print(f"checked {rows_checked} rows across {len(sampled)} episodes")
        print(f"median episode-median sum Fz: {med:.0f} N vs weight {args.weight_n:.0f} N (ratio {ratio:.3f})")
        print(f"max |slip ratio| in contact: {slip_extreme:.2f} (airborne, unchecked: {airborne_slip_extreme:.2f})")
        if not 0.85 <= ratio <= 1.15:
            failures.append(f"sum-Fz/weight ratio {ratio:.3f} outside [0.85, 1.15]")
        if slip_extreme > 50:
            failures.append(f"absurd in-contact slip ratio {slip_extreme:.1f} (sign/radius convention regression?)")
    elif not failures:
        failures.append("no rows checked")

    print("\n" + ("FAIL:\n  " + "\n  ".join(failures) if failures else "PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
