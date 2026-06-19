from __future__ import annotations
"""Quick 15D-channel sanity + distribution-shift analysis for the CRM dataset.

Compares CRM (deformable terrain) against the flat/bumpy reference stats already
computed in artifacts/analysis/hmmwv_flat_vs_bumpy_15d_distribution/states_summary.csv.
Early-data sanity pass: NaN/inf, ranges, per-channel shift, delta magnitudes
(the thing that broke bumpy fine-tuning), and the vx/omega rolling-slip signature.
"""
import csv
import json
from pathlib import Path
import numpy as np


def read_cols(path, fields):
    """Return dict field->float array for requested columns (ignores missing)."""
    with open(path, newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        col = {f: header.index(f) for f in fields if f in header}
        data = {f: [] for f in col}
        for row in rdr:
            for f, i in col.items():
                data[f].append(float(row[i]))
    return {f: np.asarray(v, dtype=np.float64) for f, v in data.items()}


def load_ref(path):
    """Read flat/bumpy states_summary.csv into channel->dict."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["channel"]] = {k: (float(v) if v not in ("", None) else float("nan"))
                                 for k, v in r.items() if k != "channel"}
    return out

REPO = Path(__file__).resolve().parents[1]
CRM_DIR = REPO / "artifacts/datasets/hmmwv_crm_100"
REF_CSV = REPO / "artifacts/analysis/hmmwv_flat_vs_bumpy_15d_distribution/states_summary.csv"
OUT_DIR = REPO / "artifacts/analysis/hmmwv_crm_15d_distribution"

STATE_FIELDS = [
    "vel_body_x_mps", "vel_body_y_mps",
    "roll_rad", "pitch_rad",
    "roll_rate_radps", "ang_vel_body_y_radps", "yaw_rate_radps",
    "tire_fl_force_wheel_fz_n", "tire_fr_force_wheel_fz_n",
    "tire_rl_force_wheel_fz_n", "tire_rr_force_wheel_fz_n",
    "tire_fl_spindle_omega_radps", "tire_fr_spindle_omega_radps",
    "tire_rl_spindle_omega_radps", "tire_rr_spindle_omega_radps",
]
OMEGA = [f for f in STATE_FIELDS if "spindle_omega" in f]
FZ = [f for f in STATE_FIELDS if "force_wheel_fz" in f]
EXTRA = ["tire_fl_slip_ratio", "tire_fr_slip_ratio", "tire_rl_slip_ratio", "tire_rr_slip_ratio",
         "tire_fl_deflection_m", "tire_rl_deflection_m", "pos_z_m", "vel_body_z_mps"]
WHEEL_RADIUS = 0.4673  # HMMWV tire effective radius (approx), for slip sanity only


def main():
    # Scan per-episode JSON sidecars directly (robust to a stale dataset_index.json).
    eps = []
    for jp in sorted((CRM_DIR / "episodes").glob("*.json")):
        e = json.loads(jp.read_text())
        e["csv_path"] = f"episodes/{jp.stem}.csv"
        if (CRM_DIR / e["csv_path"]).exists():
            eps.append(e)
    state_rows, delta_rows = [], []
    extra_cols = {c: [] for c in EXTRA}
    per_ep = []
    n_boundary = 0
    for e in eps:
        cols = read_cols(CRM_DIR / e["csv_path"], STATE_FIELDS + EXTRA)
        s = np.stack([cols[f] for f in STATE_FIELDS], axis=1)
        d = np.diff(s, axis=0)  # within-episode deltas only
        state_rows.append(s)
        delta_rows.append(d)
        for c in EXTRA:
            if c in cols:
                extra_cols[c].append(cols[c])
        n_boundary += int(e.get("terminated_near_boundary", False))
        per_ep.append((e["episode_id"], e["scenario_family"], e["split"], len(s),
                       float(np.nanmedian(cols["vel_body_x_mps"]))))
    S = np.concatenate(state_rows)
    D = np.concatenate(delta_rows)
    have_extra = [c for c in EXTRA if extra_cols[c]]
    X = {c: np.concatenate(extra_cols[c]) for c in have_extra} if have_extra else None

    print(f"CRM dataset: {len(eps)} episodes, {S.shape[0]} rows, "
          f"{n_boundary}/{len(eps)} terminated_near_boundary")
    print(f"families: {sorted(set(e['scenario_family'] for e in eps))}")
    splits = {}
    for e in eps:
        splits[e['split']] = splits.get(e['split'], 0) + 1
    print(f"split episode counts: {splits}")

    # ---- 1. sanity ----
    nonfinite = (~np.isfinite(S)).sum(axis=0)
    print("\n== SANITY ==")
    print(f"non-finite cells total: state={int((~np.isfinite(S)).sum())} delta={int((~np.isfinite(D)).sum())}")
    if nonfinite.any():
        for f, n in zip(STATE_FIELDS, nonfinite):
            if n:
                print(f"  NONFINITE {f}: {n}")
    fz = S[:, [STATE_FIELDS.index(f) for f in FZ]]
    print(f"Fz negative frac: {(fz < 0).mean():.4f}  |  Fz==0 frac: {(fz == 0).mean():.4f}  "
          f"|  Fz max: {fz.max():.0f} N")
    om = S[:, [STATE_FIELDS.index(f) for f in OMEGA]]
    print(f"omega negative frac: {(om < 0).mean():.4f}  |  omega max: {om.max():.1f} rad/s")

    # ---- 2. ref stats ----
    ref = load_ref(REF_CSV)

    # ---- 3. per-channel comparison table ----
    print("\n== 15D CHANNEL SHIFT (CRM vs flat / bumpy) ==")
    hdr = f"{'channel':30s} {'crm_med':>10s} {'flat_med':>10s} {'bmp_med':>10s} " \
          f"{'crm_std':>10s} {'std/flat':>9s} {'std/bmp':>9s} {'shift_flatσ':>11s}"
    print(hdr)
    rows_out = []
    for i, f in enumerate(STATE_FIELDS):
        col = S[:, i]
        cmed, cstd = np.median(col), col.std()
        fmed, fstd = ref[f]["flat_p50"], ref[f]["flat_std"]
        bmed, bstd = ref[f]["bumpy_p50"], ref[f]["bumpy_std"]
        shift = (col.mean() - ref[f]["flat_mean"]) / (fstd + 1e-12)
        print(f"{f:30s} {cmed:10.3f} {fmed:10.3f} {bmed:10.3f} "
              f"{cstd:10.3f} {cstd/(fstd+1e-12):9.2f} {cstd/(bstd+1e-12):9.2f} {shift:11.2f}")
        rows_out.append(dict(channel=f, crm_med=cmed, crm_std=cstd, crm_mean=col.mean(),
                             flat_med=fmed, flat_std=fstd, bumpy_med=bmed, bumpy_std=bstd,
                             crm_min=col.min(), crm_max=col.max(),
                             std_ratio_crm_over_flat=cstd/(fstd+1e-12),
                             std_ratio_crm_over_bumpy=cstd/(bstd+1e-12),
                             mean_shift_in_flat_sigma=shift))

    # ---- 4. delta magnitudes (impulsiveness) ----
    print("\n== DELTA-STATE (per-step) std & tails  [flat delta std unknown here; raw CRM] ==")
    print(f"{'channel':30s} {'d_std':>12s} {'d_p99.5':>12s} {'d_max':>12s}")
    for i, f in enumerate(STATE_FIELDS):
        dc = D[:, i]
        print(f"{f:30s} {dc.std():12.4f} {np.percentile(np.abs(dc),99.5):12.4f} {np.abs(dc).max():12.4f}")

    # ---- 5. rolling-slip signature ----
    print("\n== LONGITUDINAL SLIP SIGNATURE (deformable-terrain tell) ==")
    vx = S[:, STATE_FIELDS.index("vel_body_x_mps")]
    omega_mean = om.mean(axis=1)
    moving = vx > 2.0
    ratio = vx[moving] / (omega_mean[moving] + 1e-9)
    print(f"vx/omega ratio (moving, vx>2): median={np.median(ratio):.4f}  "
          f"(flat/bumpy rigid ≈ 0.46; lower => wheels spin faster than ground => slip/sinkage)")
    print(f"implied wheel speed omega*R median = {np.median(omega_mean[moving]*WHEEL_RADIUS):.2f} m/s "
          f"vs vx median = {np.median(vx[moving]):.2f} m/s")
    if X is not None:
        for c in have_extra:
            xc = X[c]
            print(f"  {c:28s} median={np.nanmedian(xc):10.4f}  p99.5={np.nanpercentile(np.abs(xc),99.5):10.4f}")

    # ---- 6. speed range ----
    print("\n== SPEED ==")
    print(f"vx: min={vx.min():.2f} median={np.median(vx):.2f} p95={np.percentile(vx,95):.2f} max={vx.max():.2f}  "
          f"(flat med 11.81, bumpy med 9.16)")
    print("per-episode vx medians:")
    for eid, fam, sp, n, vmed in per_ep:
        print(f"  {eid:34s} {fam:16s} {sp:5s} rows={n:5d} vx_med={vmed:6.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "crm_15d_states_summary.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
