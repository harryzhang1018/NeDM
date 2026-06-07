# HMMWV Model Checkpoints

The repository keeps trained `.pt` checkpoints with Git LFS, but does not commit raw datasets or processed `.npy` training caches.

After cloning:

```bash
git lfs install
git lfs pull
```

Useful checkpoints:

| Model | Checkpoint | Notes |
|---|---|---|
| `v04_long_baseline_b32` | `artifacts/training_runs/hmmwv_transformer_v04_long_baseline_b32/checkpoints/best_val.pt` | Short-context model with better mean/max rollout robustness in the first sweep. |
| `v07_context128_b64` | `artifacts/training_runs/hmmwv_transformer_v07_context128_b64/checkpoints/best_val.pt` | Long-context model with best median rollout error in the first sweep. |
| `v3_turn_300g` | `artifacts/training_runs/hmmwv_transformer_v3_turn_300g/checkpoints/best_val.pt` | V3 model trained on the large turn dataset. |
| `v12_wide512_b48` | `artifacts/training_runs/hmmwv_transformer_v12_wide512_b48/checkpoints/best_val.pt` | Lowest one-step validation loss in the v04-v18 sweep, but weaker long-horizon rollouts. |
| `v18_wide384_context96_b48` | `artifacts/training_runs/hmmwv_transformer_v18_wide384_context96_b48/checkpoints/best_val.pt` | Low validation loss with wider/longer-context architecture. |

Minimal model load:

```python
from pathlib import Path

import torch

from nedm.training.model import HMMWVDynamicsModel

checkpoint_path = Path(
    "artifacts/training_runs/hmmwv_transformer_v07_context128_b64/checkpoints/best_val.pt"
)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
metadata = checkpoint["metadata"]
model_config = checkpoint["config"]["model"]

model = HMMWVDynamicsModel(
    state_dim=len(metadata["state_fields"]),
    action_dim=len(metadata["action_fields"]),
    target_dim=len(metadata["state_fields"]),
    transformer_cfg=model_config,
    normalization=metadata["normalization"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

The checkpoint contains the model config and normalization metadata needed for inference. The large generated data roots remain local-only:

- `artifacts/datasets/`
- `artifacts/training_datasets/`
