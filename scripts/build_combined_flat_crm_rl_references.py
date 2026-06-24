"""Build a combined flat+CRM RL training reference set.

Takes an existing flat reference set (e.g. the 20-trajectory flat training refs)
unchanged, and appends N CRM trajectories read directly from the CRM episode
CSVs so the mixture-expert dynamics model can be RL-trained against targets from
both terrains in one reference file.

CRM selection rules (mirrors the flat builder's even family round-robin):
  * even distribution across maneuver families (same round-robin order as
    ``nedm.rl.references.DEFAULT_REFERENCE_FAMILIES``);
  * each chosen episode starts at the origin (|x0|,|y0| within --origin-tol) and
    travels more than --min-displacement metres by the end of the segment;
  * episodes that terminated at the terrain boundary are skipped by default;
  * dt and the state/action field layout must match the flat reference set so the
    tracking env accepts the combined file unchanged.

The CRM segment is taken from the *start* of each episode (rest / origin warm
start), so its first ``context_steps`` rows are a valid warm-up context.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.rl.references import (  # noqa: E402
    DEFAULT_REFERENCE_FAMILIES,
    ReferenceSet,
    load_reference_set,
    save_reference_set,
)

POSE_FIELDS = ["pos_x_m", "pos_y_m", "yaw_rad"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--flat-reference",
        type=Path,
        default=Path("artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_train_refs_20_1100_seed_20260607.npz"),
        help="Existing flat reference set to keep unchanged and extend.",
    )
    p.add_argument("--crm-dataset-dir", type=Path, default=Path("artifacts/datasets/hmmwv_crm_2000"))
    p.add_argument("--num-crm", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument(
        "--crm-segment-start",
        choices=["random", "origin"],
        default="random",
        help="random: random mid-episode window (like the flat refs); origin: rest start at (0,0).",
    )
    p.add_argument("--origin-tol", type=float, default=0.5, help="Max |x0|,|y0| (m) for an origin start.")
    p.add_argument(
        "--min-displacement",
        type=float,
        default=10.0,
        help="Min net travel (m) over the segment. Excludes near-immobilized/slow windows.",
    )
    p.add_argument(
        "--max-displacement",
        type=float,
        default=50.0,
        help="Max net travel (m) over the segment (keeps CRM targets in a sensible band).",
    )
    p.add_argument(
        "--allow-boundary",
        action="store_true",
        help="Permit episodes flagged terminated_near_boundary (skipped by default).",
    )
    p.add_argument("--split", type=str, default=None, help="Restrict CRM source to this index split (train/val).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plot-dir", type=Path, default=None, help="If set, write XY trajectory plots here.")
    return p.parse_args()


def read_episode_columns(csv_path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    needed = set(columns) | {"time_s"}
    accum: dict[str, list[float]] = {name: [] for name in needed}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        for row in reader:
            for name in needed:
                accum[name].append(float(row[name]))
    return {name: np.asarray(values, dtype=np.float32) for name, values in accum.items()}


def family_round_robin_order(available: set[str]) -> list[str]:
    ordered = [fam for fam in DEFAULT_REFERENCE_FAMILIES if fam in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def main() -> int:
    args = parse_args()

    flat = load_reference_set(args.flat_reference)
    seg_steps = flat.num_steps  # rows per reference (e.g. 1101)
    state_fields = flat.state_fields
    action_fields = flat.action_fields
    dt_s = flat.dt_s
    print(
        f"flat reference: {flat.num_references} refs x {seg_steps} rows, "
        f"{len(state_fields)}D state, dt={dt_s}"
    )

    dataset_dir = args.crm_dataset_dir if args.crm_dataset_dir.is_absolute() else REPO_ROOT / args.crm_dataset_dir
    index = json.loads((dataset_dir / "dataset_index.json").read_text())
    episodes = index["episodes"]

    # Group eligible episodes by family (need enough rows; skip boundary cutoffs).
    by_family: dict[str, list[dict[str, Any]]] = {}
    for ep in episodes:
        if args.split is not None and ep.get("split") != args.split:
            continue
        if int(ep.get("rows", 0)) < seg_steps:
            continue
        if ep.get("terminated_near_boundary") and not args.allow_boundary:
            continue
        by_family.setdefault(ep["scenario_family"], []).append(ep)

    rng = np.random.default_rng(args.seed)
    for eps in by_family.values():
        rng.shuffle(eps)  # in-place shuffle of dict refs
    window_rng = np.random.default_rng(args.seed + 17)  # window start, decoupled from family shuffle

    families = family_round_robin_order(set(by_family))
    columns = list(state_fields) + list(action_fields) + POSE_FIELDS

    skipped: list[str] = []

    def try_episode(ep: dict[str, Any]):
        """Parse + validate one candidate; return (states, actions, poses, diag) or None.

        In ``random`` mode a mid-episode window is chosen (like the flat refs)
        whose net travel falls inside [min_displacement, max_displacement]; in
        ``origin`` mode the segment is the rest start at (0,0).
        """
        csv_path = dataset_dir / ep["csv_path"]
        try:
            data = read_episode_columns(csv_path, columns)
        except ValueError as exc:
            skipped.append(f"{ep['episode_id']}: {exc}")
            return None

        # dt sanity (recorder lands on 0.01s grid within one substep).
        steps = np.diff(data["time_s"])
        if abs(float(np.median(steps)) - dt_s) > 1e-4 or float(np.max(np.abs(steps - dt_s))) > 1e-3:
            skipped.append(f"{ep['episode_id']}: bad dt cadence")
            return None

        full_states = np.stack([data[f] for f in state_fields], axis=-1).astype(np.float32)
        full_actions = np.stack([data[f] for f in action_fields], axis=-1).astype(np.float32)
        full_poses = np.stack([data[f] for f in POSE_FIELDS], axis=-1).astype(np.float32)
        length = full_poses.shape[0]
        max_start = length - seg_steps
        if max_start < 0:
            skipped.append(f"{ep['episode_id']}: only {length} rows (< {seg_steps})")
            return None

        # Net travel of every candidate window, then keep those inside the band.
        starts = np.arange(0, max_start + 1)
        ends = starts + seg_steps - 1
        disp = np.hypot(
            full_poses[ends, 0] - full_poses[starts, 0],
            full_poses[ends, 1] - full_poses[starts, 1],
        )
        in_band = (disp >= args.min_displacement) & (disp <= args.max_displacement)

        if args.crm_segment_start == "origin":
            x0, y0 = float(full_poses[0, 0]), float(full_poses[0, 1])
            if abs(x0) > args.origin_tol or abs(y0) > args.origin_tol:
                skipped.append(f"{ep['episode_id']}: start ({x0:.2f},{y0:.2f}) not at origin")
                return None
            if not in_band[0]:
                skipped.append(f"{ep['episode_id']}: origin-window travel {float(disp[0]):.1f}m out of band")
                return None
            start = 0
        else:  # random window inside the travel band
            valid = starts[in_band]
            if valid.size == 0:
                skipped.append(
                    f"{ep['episode_id']}: no window in [{args.min_displacement:.0f},{args.max_displacement:.0f}]m"
                )
                return None
            start = int(valid[window_rng.integers(0, valid.size)])

        stop = start + seg_steps
        states = full_states[start:stop]
        actions = full_actions[start:stop]
        poses = full_poses[start:stop]

        if not (np.all(np.isfinite(states)) and np.all(np.isfinite(actions)) and np.all(np.isfinite(poses))):
            skipped.append(f"{ep['episode_id']}: non-finite values")
            return None

        diag = {
            "segment_start_row": int(start),
            "start_x_m": float(poses[0, 0]),
            "start_y_m": float(poses[0, 1]),
            "disp_end_m": float(np.hypot(poses[-1, 0] - poses[0, 0], poses[-1, 1] - poses[0, 1])),
            "mean_speed_mps": float(np.mean(states[:, 0])),
            "rows_available": int(length),
        }
        return states, actions, poses, diag

    selected: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, float]]] = []
    cursor = 0
    active = [fam for fam in families if by_family.get(fam)]
    # Round-robin over families; retry within a family until one candidate passes
    # so that failed-filter candidates do not skew the family balance.
    while len(selected) < args.num_crm and active:
        fam = active[cursor % len(active)]
        picked = None
        while by_family[fam] and picked is None:
            ep = by_family[fam].pop()
            result = try_episode(ep)
            if result is not None:
                picked = (ep, *result)
        if picked is not None:
            selected.append(picked)
        if not by_family[fam]:
            active.remove(fam)
            cursor -= 1
        cursor += 1

    if len(selected) < args.num_crm:
        raise SystemExit(
            f"Only selected {len(selected)}/{args.num_crm} CRM refs after filtering. "
            f"Loosen --min/--max-displacement. First skips: {skipped[:5]}"
        )

    crm_states = np.stack([s for _, s, _, _, _ in selected], axis=0)
    crm_actions = np.stack([a for _, _, a, _, _ in selected], axis=0)
    crm_poses = np.stack([p for _, _, _, p, _ in selected], axis=0)
    crm_ids = [ep["episode_id"] for ep, *_ in selected]
    crm_families = [ep["scenario_family"] for ep, *_ in selected]

    # Combine: flat refs unchanged, CRM appended.
    states = np.concatenate([flat.states, crm_states], axis=0).astype(np.float32)
    actions = np.concatenate([flat.actions, crm_actions], axis=0).astype(np.float32)
    poses = np.concatenate([flat.poses, crm_poses], axis=0).astype(np.float32)
    episode_ids = list(flat.episode_ids) + crm_ids
    scenario_families = list(flat.scenario_families) + crm_families
    domains = ["flat"] * flat.num_references + ["crm"] * len(selected)

    crm_segment_records = [
        {
            "episode_id": ep["episode_id"],
            "scenario_family": ep["scenario_family"],
            "domain": "crm",
            "rows_used": int(seg_steps),
            **diag,
        }
        for ep, _, _, _, diag in selected
    ]

    metadata = {
        "source": "combined_flat_crm",
        "flat_reference": str(args.flat_reference),
        "flat_metadata": {k: flat.metadata.get(k) for k in ("source_processed_root", "source_split", "seed")},
        "crm_dataset_dir": str(dataset_dir),
        "crm_seed": int(args.seed),
        "crm_num": int(len(selected)),
        "crm_segment_start": args.crm_segment_start,
        "crm_origin_tol_m": float(args.origin_tol),
        "crm_min_displacement_m": float(args.min_displacement),
        "crm_max_displacement_m": float(args.max_displacement),
        "crm_allow_boundary": bool(args.allow_boundary),
        "domains": domains,
        "crm_segments": crm_segment_records,
    }

    combined = ReferenceSet(
        states=states,
        actions=actions,
        poses=poses,
        episode_ids=episode_ids,
        scenario_families=scenario_families,
        dt_s=dt_s,
        state_fields=state_fields,
        action_fields=action_fields,
        rollout_fields=flat.rollout_fields,
        metadata=metadata,
    )
    out_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    save_reference_set(combined, out_path)

    print(f"\nwrote {combined.num_references} combined references -> {out_path}")
    print(f"  flat: {flat.num_references}  crm: {len(selected)}  rows/ref: {seg_steps} (~{seg_steps * dt_s:.1f}s)")
    print(f"  crm segment start: {args.crm_segment_start}  travel band: "
          f"[{args.min_displacement:.0f},{args.max_displacement:.0f}] m")
    print(f"  flat family dist: {dict(Counter(flat.scenario_families))}")
    print(f"  crm  family dist: {dict(Counter(crm_families))}")
    print("  CRM selections:")
    for ep, _, _, _, diag in selected:
        print(
            f"    {ep['episode_id']:34s} fam={ep['scenario_family']:14s} "
            f"win@row{diag['segment_start_row']:>4d} start=({diag['start_x_m']:+7.1f},{diag['start_y_m']:+7.1f}) "
            f"disp={diag['disp_end_m']:5.1f}m  meanvx={diag['mean_speed_mps']:5.2f}"
        )
    if skipped:
        print(f"  ({len(skipped)} candidates skipped during selection)")

    if args.plot_dir is not None:
        plot_dir = args.plot_dir if args.plot_dir.is_absolute() else REPO_ROOT / args.plot_dir
        write_plots(plot_dir, combined, domains)
        print(f"\nwrote XY trajectory plots -> {plot_dir}")

    return 0


def write_plots(plot_dir: Path, ref: ReferenceSet, domains: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    n = ref.num_references
    # Re-origin every trajectory to (0,0) for visual comparison.
    xy = ref.poses[:, :, :2] - ref.poses[:, :1, :2]

    # 1) Overlay: flat vs crm.
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, dom, color in ((axes[0], "flat", "tab:blue"), (axes[1], "crm", "tab:orange")):
        for i in range(n):
            if domains[i] != dom:
                continue
            ax.plot(xy[i, :, 0], xy[i, :, 1], color=color, alpha=0.7, lw=1.2)
            ax.plot(0, 0, "k.", ms=4)
            ax.plot(xy[i, -1, 0], xy[i, -1, 1], "rx", ms=5)
        ndom = sum(d == dom for d in domains)
        ax.set_title(f"{dom} ({ndom} refs)  re-origined to (0,0)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Combined RL reference set XY trajectories (n={n})")
    fig.tight_layout()
    fig.savefig(plot_dir / "overlay_flat_vs_crm.png", dpi=130)
    plt.close(fig)

    # 2) Grid of all trajectories (each in its own panel).
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 2.5))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        ax = axes[i]
        color = "tab:blue" if domains[i] == "flat" else "tab:orange"
        ax.plot(xy[i, :, 0], xy[i, :, 1], color=color, lw=1.2)
        ax.plot(0, 0, "k.", ms=4)
        ax.plot(xy[i, -1, 0], xy[i, -1, 1], "rx", ms=5)
        ax.set_title(f"{domains[i]}:{ref.scenario_families[i]}", fontsize=7)
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Combined RL reference set: per-trajectory XY (n={n})")
    fig.tight_layout()
    fig.savefig(plot_dir / "grid_all.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
