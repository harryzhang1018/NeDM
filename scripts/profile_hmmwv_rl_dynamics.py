from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.rl.defaults import DEFAULT_RL_DYNAMICS_CHECKPOINT, DEFAULT_RL_REFERENCE_PATH
from nedm.rl.hmmwv_tracking_env import HMMWVNeuralTrackingEnv, default_env_cfg, merge_env_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile HMMWV RL NN dynamics inference throughput.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-envs", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--full-step-iters", type=int, default=8)
    parser.add_argument("--substep-iters", type=int, default=20)
    parser.add_argument("--model-iters", type=int, default=20)
    parser.add_argument("--obs-iters", type=int, default=40)
    parser.add_argument("--warmup-iters", type=int, default=4)
    parser.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="highest")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--dynamics-checkpoint", type=Path, default=DEFAULT_RL_DYNAMICS_CHECKPOINT)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_RL_REFERENCE_PATH)
    return parser.parse_args()


def configure_precision(precision: str) -> None:
    torch.set_float32_matmul_precision(precision)
    if torch.cuda.is_available():
        allow_tf32 = precision != "highest"
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def cuda_or_wall_time(fn: Callable[[], None], iters: int, warmup_iters: int, device: torch.device) -> float:
    with torch.inference_mode():
        for _ in range(warmup_iters):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize(device)
            return start.elapsed_time(end) / 1000.0

        start_time = time.perf_counter()
        for _ in range(iters):
            fn()
        return time.perf_counter() - start_time


def predict_next_delta_last_head(model: torch.nn.Module, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    tokens = torch.cat([model.normalize_state(states), model.normalize_action(actions)], dim=-1)
    features = model.backbone(tokens)
    return model.denormalize_target(model.head(features[:, -1, :]))


def nn_substep_last_head(env: HMMWVNeuralTrackingEnv, driver_actions: torch.Tensor) -> None:
    env.action_hist[:, -1, :] = driver_actions
    delta = predict_next_delta_last_head(env.model, env.state_hist, env.action_hist)
    next_state = env.state_hist[:, -1, :] + delta
    env.pose = env._integrate_pose(env.pose, next_state)

    env.state_hist = torch.roll(env.state_hist, shifts=-1, dims=1)
    env.action_hist = torch.roll(env.action_hist, shifts=-1, dims=1)
    env.state_hist[:, -1, :] = next_state
    env.action_hist[:, -1, :] = driver_actions
    env.ref_step_buf = torch.clamp(env.ref_step_buf + 1, max=env.reference_length - 1)


def make_env(args: argparse.Namespace, num_envs: int, device: torch.device) -> HMMWVNeuralTrackingEnv:
    cfg = default_env_cfg()
    cfg.update(
        {
            "num_envs": num_envs,
            "device": str(device),
            "dynamics_checkpoint": str(args.dynamics_checkpoint),
            "reference_path": str(args.reference_path),
            "action_repeat": int(args.action_repeat),
            "auto_reset": True,
        }
    )
    env = HMMWVNeuralTrackingEnv(merge_env_cfg(cfg), device=device)
    if args.compile_model:
        env.model = torch.compile(env.model, mode="reduce-overhead")
    return env


def reset_env(env: HMMWVNeuralTrackingEnv) -> None:
    with torch.inference_mode():
        env.reset()


def profile_env(args: argparse.Namespace, num_envs: int, device: torch.device) -> None:
    env = make_env(args, num_envs, device)
    policy_actions = torch.zeros(num_envs, env.num_actions, dtype=torch.float32, device=device)
    driver_actions = env._scale_policy_actions(policy_actions)

    with torch.inference_mode():
        full_delta = env.model.predict_delta(env.state_hist, env.action_hist)[:, -1, :]
        last_delta = predict_next_delta_last_head(env.model, env.state_hist, env.action_hist)
        max_delta_diff = torch.max(torch.abs(full_delta - last_delta)).item()

    model_full_s = cuda_or_wall_time(
        lambda: env.model.predict_delta(env.state_hist, env.action_hist)[:, -1, :],
        args.model_iters,
        args.warmup_iters,
        device,
    )
    model_last_s = cuda_or_wall_time(
        lambda: predict_next_delta_last_head(env.model, env.state_hist, env.action_hist),
        args.model_iters,
        args.warmup_iters,
        device,
    )

    reset_env(env)
    substep_env_s = cuda_or_wall_time(
        lambda: env._nn_substep(driver_actions),
        args.substep_iters,
        args.warmup_iters,
        device,
    )

    reset_env(env)
    substep_last_s = cuda_or_wall_time(
        lambda: nn_substep_last_head(env, driver_actions),
        args.substep_iters,
        args.warmup_iters,
        device,
    )

    reset_env(env)
    obs_s = cuda_or_wall_time(
        lambda: env._compute_observations(),
        args.obs_iters,
        args.warmup_iters,
        device,
    )

    reset_env(env)
    full_step_s = cuda_or_wall_time(
        lambda: env.step(policy_actions),
        args.full_step_iters,
        args.warmup_iters,
        device,
    )

    policy_steps_per_s = args.full_step_iters / full_step_s
    env_steps_per_s = policy_steps_per_s * num_envs
    nn_substeps_per_s = policy_steps_per_s * num_envs * env.action_repeat

    print(f"\nnum_envs={num_envs}")
    print(f"  model full-output last-slice: {model_full_s / args.model_iters * 1000.0:.3f} ms/call")
    print(f"  model last-head only:         {model_last_s / args.model_iters * 1000.0:.3f} ms/call")
    print(f"  max delta diff:              {max_delta_diff:.6g}")
    print(f"  nn_substep env impl:         {substep_env_s / args.substep_iters * 1000.0:.3f} ms/substep")
    print(f"  nn_substep last-head:        {substep_last_s / args.substep_iters * 1000.0:.3f} ms/substep")
    print(f"  observations:                {obs_s / args.obs_iters * 1000.0:.3f} ms/call")
    print(f"  full env.step:               {full_step_s / args.full_step_iters * 1000.0:.3f} ms/policy step")
    print(f"  throughput:                  {env_steps_per_s:.0f} policy env-steps/s")
    print(f"  NN substep throughput:       {nn_substeps_per_s:.0f} NN env-substeps/s")


def main() -> int:
    args = parse_args()
    configure_precision(args.matmul_precision)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"torch={torch.__version__}")
    print(f"matmul_precision={args.matmul_precision} compile_model={args.compile_model}")
    print(f"dynamics_checkpoint={args.dynamics_checkpoint.resolve()}")
    print(f"reference_path={args.reference_path.resolve()}")
    for num_envs in args.num_envs:
        profile_env(args, int(num_envs), device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
