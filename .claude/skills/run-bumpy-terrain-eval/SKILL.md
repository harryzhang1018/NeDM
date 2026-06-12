---
name: run-bumpy-terrain-eval
description: Run the Chrono HMMWV RL tracking eval on bumpy rigid-heightmap terrain (not flat ground), reproducing the exact per-episode terrain each reference was collected on. Use when asked to evaluate a tracking policy on the bumpy dataset, transfer a flat-trained policy to bumpy terrain, or set up / debug heightmap terrain in the Chrono eval.
---

# Run the Chrono RL tracking eval on bumpy terrain

Evaluates an RL tracking policy in real Chrono on **bumpy rigid-heightmap terrain**, reproducing the exact bump field each reference trajectory was collected on. This is the out-of-regime counterpart to the default flat-ground Chrono eval.

## Why per-episode terrain matters

The bumpy dataset (`artifacts/datasets/hmmwv_bumpy_10g_shards`) was collected with `terrain.type = "rigid_heightmap"`: a 100-map library `assets/bumpy_terrain/bumpy_field_%03d.bmp` (256×256 px stretched over a **500×500 m** patch ≈ 2 m/px, grayscale mapped to height **±0.6 m**). Each episode used a different map, **deterministically chosen from its `episode_id`**:

```
assign_height_map_index(episode_id, 100)  # = int(md5(f"terrain::{episode_id}"), 16) % 100
```

(in `src/nedm/hmmwv_data.py`). A faithful eval must drive each reference over the same map it was recorded on. `HMMWVChronoTrackingEnv._create_sim` does this automatically: it calls `resolve_height_map(config, episode_id)` and passes the BMP to `create_rigid_terrain`. For flat `rigid` configs `resolve_height_map` returns `None`, so the default flat eval is unchanged.

## One-time setup (already built 2026-06-11; rebuild if missing)

1. **Processed cache** from the raw bumpy shards (needed to build references):
   ```bash
   python scripts/build_hmmwv_training_dataset.py \
     --dataset-root artifacts/datasets/hmmwv_bumpy_10g_shards/shard_00{0,1,2,3} \
     --output-dir artifacts/training_datasets/hmmwv_bumpy_10g_seq_v1
   ```
2. **Rest-start reference set** (zero-speed start so Chrono can warm-start — see the chrono-rl-reference-rest-start memory):
   ```bash
   python scripts/build_hmmwv_rl_references.py \
     --processed-dataset-dir artifacts/training_datasets/hmmwv_bumpy_10g_seq_v1 \
     --split train --num-references 20 --segment-nn-steps 1100 --no-random-segment-start \
     --output artifacts/rl_reference_sets/hmmwv_bumpy_refs_20_1100_rest_start.npz
   ```
   Bumpy data has only 6 families (sustained_turn, sine/doublet/multi/chirp_steer, steer_brake) — no launch_brake/step_steer/aggressive_*.
3. **Eval Chrono config** `configs/hmmwv_bumpy_eval.json` = `configs/hmmwv_overfit_v1.json` with **only** the `terrain` block swapped to the bumpy heightmap block (vehicle+simulation blocks are byte-identical to the collector config). Keep the 500×500 patch — that sets the bump wavelength.
4. **Side run-dir** `<RUN>/_bumpycfg/` = copy of the run's `env_cfg.json`+`train_cfg.json`, with `reference_path` → the bumpy npz and `termination.max_position_error_m` → 20.0 (relax the eval bound; training bound was 1 m). This is the override mechanism — there is no `--reference-path` flag.

## Run it

`eval_hmmwv_rl_chrono_tracking.py` reads `reference_path` from `--run-dir`'s `env_cfg.json` and the heightmap terrain from `--chrono-config`. Run **one reference per process** (a multi-ref loop native-crashes with `stack smashing detected`), with the `0.3` steering clamp that prevents the unrelated steering-reversal divergences, and the nedm env's libstdc++ on the path (else pychrono fails to import with `CXXABI_1.3.15 not found`):

```bash
RUN=artifacts/rl_runs/hmmwv_rl_tracking_v07_8192env_16steps_term1m_20260608
OUT=$RUN/chrono_eval_model1999_bumpy_pychrono10_steerlimit03
export LD_LIBRARY_PATH=/home/harry/anaconda3/envs/nedm/lib:$LD_LIBRARY_PATH

for i in $(seq 0 19); do
  /home/harry/anaconda3/envs/nedm/bin/python scripts/eval_hmmwv_rl_chrono_tracking.py \
    --run-dir "$RUN/_bumpycfg" \
    --policy-checkpoint "$RUN/model_1999.pt" \
    --chrono-config configs/hmmwv_bumpy_eval.json \
    --steering-rate-limit 0.3 \
    --reference-index "$i" \
    --output-dir "$OUT"
done
```

Each process prints one `{"steps": ...}` metrics line; aggregate them into `summary.json` yourself. A process occasionally dies in native code **during plotting, after** `chrono_tracking_NN.npz` is saved — recover that ref's metrics from the npz (`pose` vs `ref_pose` XY RMSE) instead of rerunning. Watch the box during the loop; an unattended batch hard-froze the machine once.

## Verify the terrain is actually bumpy (do not get fooled)

Every `bumpy_field` BMP is **flat (height ≈ 0) at the center/origin** — the spawn point is deliberately level. Consequences that look like bugs but are not:

- `RigidTerrain.GetHeight(origin)` returns **0**, and the vehicle settles to the **same** chassis z (~0.57 m) on every map. Neither means the heightmap failed to load.
- A slow reference (e.g. a tight sustained_turn that travels only a few metres) barely leaves the flat origin, so its roll/pitch stays tiny — also not a bug.

To actually confirm the heightmap is applied, drive ~50 m and check that `chassis_z − terrain_height(x,y)` stays constant (~0.59 m ride height) while `terrain_height` swings. Collected fast episodes (~20 m/s) show `pos_z` swinging 0.29→0.82 m — the ±0.5 m heightmap signature.

## Interpreting results

The flat-trained `model_1999` regresses on bumpy terrain: mean XY RMSE 0.26 → **1.46 m**, median 0.22 → **0.35 m**, **4/20 diverge** (high-speed sine/multi/doublet/chirp steering). Both the frozen v07 NN dynamics model and the policy only saw flat-terrain tire dynamics — the fix is finetuning both on the bumpy dataset, then re-running this eval to measure recovery.
