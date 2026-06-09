"""Overlay reference vs NN-env-policy vs Chrono-env-policy trajectories for given references.

Re-runs the NN-dynamics rollout for the requested references, loads the matching Chrono-env
rollout npz, and plots XY / position-error / forward-speed overlays so the NN-env and Chrono-env
behaviour can be compared against the same reference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rsl_rl.runners import OnPolicyRunner

from nedm.rl.hmmwv_tracking_env import HMMWVNeuralTrackingEnv


def tnp(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().copy()


def nn_rollouts(run_dir: Path, checkpoint: Path, device: str) -> tuple[list[dict], "HMMWVNeuralTrackingEnv"]:
    env_cfg = json.loads((run_dir / "env_cfg.json").read_text())
    train_cfg = json.loads((run_dir / "train_cfg.json").read_text())
    num_refs = env_cfg_num_refs(env_cfg)
    env_cfg.update({"device": device, "auto_reset": False, "num_envs": num_refs})
    env = HMMWVNeuralTrackingEnv(env_cfg, device=device)
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids, reference_ids=env_ids % env.num_references)
    obs, _ = env.get_observations()
    recs = [{"pose": [], "ref_pose": [], "state": [], "ref_state": []} for _ in range(env.num_envs)]
    with torch.inference_mode():
        for _ in range(int(env.max_episode_length)):
            obs, _, _, _ = env.step(policy(obs))
            ref_state, ref_pose = env.current_reference_state_pose()
            pose, state = env.current_pose(), env.current_state()
            for i in range(env.num_envs):
                recs[i]["pose"].append(tnp(pose[i]))
                recs[i]["ref_pose"].append(tnp(ref_pose[i]))
                recs[i]["state"].append(tnp(state[i]))
                recs[i]["ref_state"].append(tnp(ref_state[i]))
    packed = [{k: np.stack(v, axis=0) for k, v in r.items()} for r in recs]
    return packed, env


def env_cfg_num_refs(env_cfg: dict) -> int:
    from nedm.rl.references import load_reference_set

    return load_reference_set(env_cfg["reference_path"]).num_references


def overlay(ref_index: int, nn_rec: dict, chrono_npz: Path, env, step_dt: float, out_path: Path) -> dict:
    chrono = np.load(chrono_npz)
    vx = env.state_index["vel_body_x_mps"]
    yaw_rate = env.state_index["yaw_rate_radps"]
    ref_pose = nn_rec["ref_pose"]
    nn_pose, ch_pose = nn_rec["pose"], chrono["pose"]
    n = min(len(nn_pose), len(ch_pose), len(ref_pose))
    t = np.arange(n) * step_dt
    nn_err = np.linalg.norm(nn_pose[:n, :2] - ref_pose[:n, :2], axis=1)
    ch_err = np.linalg.norm(ch_pose[:n, :2] - chrono["ref_pose"][:n, :2], axis=1)

    fig, ax = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    a = ax[0, 0]
    a.plot(ref_pose[:, 0], ref_pose[:, 1], color="#0f766e", lw=2.4, label="reference")
    a.plot(nn_pose[:, 0], nn_pose[:, 1], color="#1d4ed8", lw=1.8, label="NN-env policy")
    a.plot(ch_pose[:, 0], ch_pose[:, 1], color="#dc2626", lw=1.8, label="Chrono-env policy")
    a.scatter(*ref_pose[0, :2], color="#0f766e", s=40, zorder=5)
    a.set_title("XY trajectory"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    a.axis("equal"); a.grid(True, alpha=0.3); a.legend()

    a = ax[0, 1]
    a.plot(t, nn_err[:n], color="#1d4ed8", label=f"NN-env (rmse {np.sqrt(np.mean(nn_err**2)):.2f} m)")
    a.plot(t, ch_err[:n], color="#dc2626", label=f"Chrono-env (rmse {np.sqrt(np.mean(ch_err**2)):.2f} m)")
    a.set_title("Position error vs reference"); a.set_xlabel("time [s]"); a.set_ylabel("xy error [m]")
    a.grid(True, alpha=0.3); a.legend()

    a = ax[1, 0]
    a.plot(t, nn_rec["ref_state"][:n, vx], color="#0f766e", lw=2.2, label="reference vx")
    a.plot(t, nn_rec["state"][:n, vx], color="#1d4ed8", label="NN-env vx")
    a.plot(t, chrono["state"][:n, vx], color="#dc2626", label="Chrono-env vx")
    a.set_title("Forward speed"); a.set_xlabel("time [s]"); a.set_ylabel("vx [m/s]")
    a.grid(True, alpha=0.3); a.legend()

    a = ax[1, 1]
    a.plot(t, nn_rec["ref_state"][:n, yaw_rate], color="#0f766e", lw=2.2, label="reference yaw rate")
    a.plot(t, nn_rec["state"][:n, yaw_rate], color="#1d4ed8", label="NN-env yaw rate")
    a.plot(t, chrono["state"][:n, yaw_rate], color="#dc2626", label="Chrono-env yaw rate")
    a.set_title("Yaw rate"); a.set_xlabel("time [s]"); a.set_ylabel("yaw rate [rad/s]")
    a.grid(True, alpha=0.3); a.legend()

    fig.suptitle(f"NN-env vs Chrono-env — ref {ref_index}: {env.reference_names()[ref_index]}", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)
    return {"nn_rmse": float(np.sqrt(np.mean(nn_err**2))), "chrono_rmse": float(np.sqrt(np.mean(ch_err**2)))}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--policy-checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--refs", type=int, nargs="+", required=True)
    p.add_argument("--chrono-dirs", type=Path, nargs="+", required=True,
                   help="One chrono_eval dir per --refs entry (contains chrono_tracking_<idx>.npz).")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    assert len(args.refs) == len(args.chrono_dirs), "need one --chrono-dirs per --refs"
    recs, env = nn_rollouts(args.run_dir.resolve(), args.policy_checkpoint.resolve(), args.device)
    for ref_index, cdir in zip(args.refs, args.chrono_dirs):
        npz = cdir.resolve() / f"chrono_tracking_{ref_index:02d}.npz"
        out = args.output_dir.resolve() / f"overlay_ref{ref_index:02d}.png"
        m = overlay(ref_index, recs[ref_index], npz, env, env.step_dt, out)
        print(f"ref {ref_index:2d}: NN rmse={m['nn_rmse']:.3f} m  Chrono rmse={m['chrono_rmse']:.3f} m  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
