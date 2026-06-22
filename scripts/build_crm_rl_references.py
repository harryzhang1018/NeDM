"""Build an RL reference set from raw HMMWV CRM episode CSVs.

Unlike ``build_hmmwv_rl_references.py`` (which slices a processed transition
cache), this reads the episode CSVs directly so you can pick specific CRM
trajectories and reproduce them on CRM terrain in the Chrono eval. The state and
action field layout (and dt) are taken from a dynamics checkpoint so the
resulting reference set is accepted by the tracking env unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.rl.references import ReferenceSet, save_reference_set

POSE_FIELDS = ["pos_x_m", "pos_y_m", "yaw_rad"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crm-dataset-dir", type=Path, default=Path("artifacts/datasets/hmmwv_crm_100"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Dynamics checkpoint whose metadata defines state_fields/action_fields/dt_s.",
    )
    parser.add_argument("--num-references", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument(
        "--episode-ids",
        type=str,
        nargs="*",
        default=None,
        help="Explicit episode ids to use (overrides random sampling).",
    )
    parser.add_argument(
        "--max-segment-steps",
        type=int,
        default=None,
        help="Optional cap on reference length (rows). Default: min length over selected episodes.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint_fields(checkpoint_path: Path) -> tuple[list[str], list[str], float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata") or {}
    if not metadata:
        raise ValueError(f"checkpoint {checkpoint_path} has no embedded metadata")
    return list(metadata["state_fields"]), list(metadata["action_fields"]), float(metadata["dt_s"])


def read_episode_csv(csv_path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    needed = set(columns) | {"time_s"}
    accum: dict[str, list[float]] = {name: [] for name in needed}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            for name in needed:
                accum[name].append(float(row[name]))
    return {name: np.asarray(values, dtype=np.float32) for name, values in accum.items()}


def check_dt(time_s: np.ndarray, dt_s: float, episode_id: str) -> None:
    if time_s.shape[0] < 2:
        raise ValueError(f"{episode_id} has fewer than 2 rows")
    steps = np.diff(time_s)
    # The recorder lands samples on the 0.01 s grid to within one solver substep
    # (5e-4 s) of float accumulation jitter; reject only genuinely wrong cadences.
    if abs(float(np.median(steps)) - dt_s) > 1e-4 or float(np.max(np.abs(steps - dt_s))) > 1e-3:
        raise ValueError(
            f"{episode_id} record spacing median={float(np.median(steps)):.5f}s "
            f"max_dev={float(np.max(np.abs(steps - dt_s))):.5f}s does not match checkpoint dt_s={dt_s}"
        )


def select_episode_ids(args: argparse.Namespace, index: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = index["episodes"]
    by_id = {ep["episode_id"]: ep for ep in episodes}
    if args.episode_ids:
        chosen = []
        for episode_id in args.episode_ids:
            if episode_id not in by_id:
                raise KeyError(f"episode id not in dataset: {episode_id}")
            chosen.append(by_id[episode_id])
        return chosen
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(episodes))
    return [episodes[i] for i in order[: args.num_references]]


def main() -> int:
    args = parse_args()
    state_fields, action_fields, dt_s = load_checkpoint_fields(args.checkpoint.resolve())
    columns = list(state_fields) + list(action_fields) + POSE_FIELDS

    dataset_dir = args.crm_dataset_dir if args.crm_dataset_dir.is_absolute() else REPO_ROOT / args.crm_dataset_dir
    index = json.loads((dataset_dir / "dataset_index.json").read_text())
    selected = select_episode_ids(args, index)

    per_episode: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    for ep in selected:
        csv_path = dataset_dir / ep["csv_path"]
        data = read_episode_csv(csv_path, columns)
        check_dt(data["time_s"], dt_s, ep["episode_id"])
        per_episode.append((ep, data))

    lengths = [data["time_s"].shape[0] for _, data in per_episode]
    segment = min(lengths)
    if args.max_segment_steps is not None:
        segment = min(segment, int(args.max_segment_steps))

    states = np.stack(
        [np.stack([data[f][:segment] for f in state_fields], axis=-1) for _, data in per_episode], axis=0
    ).astype(np.float32)
    actions = np.stack(
        [np.stack([data[f][:segment] for f in action_fields], axis=-1) for _, data in per_episode], axis=0
    ).astype(np.float32)
    poses = np.stack(
        [np.stack([data[f][:segment] for f in POSE_FIELDS], axis=-1) for _, data in per_episode], axis=0
    ).astype(np.float32)

    episode_ids = [ep["episode_id"] for ep, _ in per_episode]
    families = [ep.get("scenario_family", ep["episode_id"]) for ep, _ in per_episode]

    segment_records = []
    union = {"min_x": np.inf, "max_x": -np.inf, "min_y": np.inf, "max_y": -np.inf}
    for index_i, (ep, data) in enumerate(per_episode):
        px, py = poses[index_i, :, 0], poses[index_i, :, 1]
        union["min_x"] = min(union["min_x"], float(px.min()))
        union["max_x"] = max(union["max_x"], float(px.max()))
        union["min_y"] = min(union["min_y"], float(py.min()))
        union["max_y"] = max(union["max_y"], float(py.max()))
        segment_records.append(
            {
                "episode_id": ep["episode_id"],
                "scenario_family": ep.get("scenario_family"),
                "rows_used": int(segment),
                "rows_available": int(data["time_s"].shape[0]),
                "start_vel_body_x_mps": float(data["vel_body_x_mps"][0]) if "vel_body_x_mps" in data else None,
                "bbox_x_m": [float(px.min()), float(px.max())],
                "bbox_y_m": [float(py.min()), float(py.max())],
            }
        )

    reference_set = ReferenceSet(
        states=states,
        actions=actions,
        poses=poses,
        episode_ids=episode_ids,
        scenario_families=families,
        dt_s=dt_s,
        state_fields=state_fields,
        action_fields=action_fields,
        rollout_fields=POSE_FIELDS,
        metadata={
            "source": "crm_episode_csv",
            "crm_dataset_dir": str(dataset_dir),
            "checkpoint": str(args.checkpoint),
            "seed": int(args.seed),
            "segment_steps": int(segment),
            "segments": segment_records,
            "terrain_union_bbox_m": union,
        },
    )
    out_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    save_reference_set(reference_set, out_path)

    print(f"wrote {len(episode_ids)} CRM references -> {out_path}")
    print(f"segment_steps={segment} (~{segment * dt_s:.1f}s)  state_dim={states.shape[-1]}")
    for record in segment_records:
        print(
            f"  {record['episode_id']:32s} fam={record['scenario_family']:14s} "
            f"v0={record['start_vel_body_x_mps']:.3f} m/s  "
            f"x[{record['bbox_x_m'][0]:7.1f},{record['bbox_x_m'][1]:7.1f}] "
            f"y[{record['bbox_y_m'][0]:7.1f},{record['bbox_y_m'][1]:7.1f}]"
        )
    pad = 25.0  # cover the 20 m position-error termination radius plus active-domain reach
    print(
        "union trajectory bbox m: "
        f"x[{union['min_x']:.1f},{union['max_x']:.1f}] y[{union['min_y']:.1f},{union['max_y']:.1f}]  "
        f"=> suggested terrain (with {pad:.0f} m pad): "
        f"length>={union['max_x'] - union['min_x'] + 2 * pad:.0f} "
        f"width>={union['max_y'] - union['min_y'] + 2 * pad:.0f} "
        f"center=({0.5 * (union['min_x'] + union['max_x']):.1f}, {0.5 * (union['min_y'] + union['max_y']):.1f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
