# GalaxeaVLA Inference Runtime Guide

This repository now includes a reusable PyTorch inference runtime for later robot deployment.

## What was added

- `src/galaxea_fm/inference/runtime.py`
  - Loads model config, data config, checkpoint, dataset stats, and PaliGemma weights.
  - Applies the same processor and normalizer logic used by training and offline evaluation.
  - Returns postprocessed robot actions in structured form.
- `src/galaxea_fm/inference/schemas.py`
  - Defines the observation schema consumed by inference.
  - Supports loading image inputs from `.png`, `.jpg`, or `.npy`.
- `configs/inference/r1lite_g0plus_pytorch.yaml`
  - Deployment-oriented runtime config template.
- `scripts/server_inference.py`
  - CLI entrypoint for dummy inference and JSON-based inference requests.

## Runtime config

Edit [configs/inference/r1lite_g0plus_pytorch.yaml](/Users/play/Documents/Project/GalaxeaVLA/configs/inference/r1lite_g0plus_pytorch.yaml) on the Linux inference host:

```yaml
model_config_path: configs/model/vla/g0plus.yaml
data_config_path: configs/data/r1lite/arm_torso_velocity_chassis.yaml
checkpoint_path: checkpoints/G0Plus_3B_base/model_state_dict.pt
dataset_stats_path: checkpoints/G0Plus_3B_base/dataset_stats.json
paligemma_path: data/google/paligemma-3b-pt-224
device: cuda
strict_checkpoint_loading: false
seed: 42
```

## Observation JSON contract

Use `scripts/server_inference.py --dump_example_json /tmp/request.json` to generate a template.

Expected structure:

```json
{
  "instruction": "Pick up the red cup and place it on the table",
  "coarse_task": "Pick and place",
  "idx": 0,
  "images": {
    "head_rgb": "/absolute/path/to/head_rgb.png",
    "left_wrist_rgb": "/absolute/path/to/left_wrist_rgb.png",
    "right_wrist_rgb": "/absolute/path/to/right_wrist_rgb.png"
  },
  "state": {
    "left_arm": [[0, 0, 0, 0, 0, 0]],
    "left_gripper": [[0]],
    "right_arm": [[0, 0, 0, 0, 0, 0]],
    "right_gripper": [[0]],
    "torso": [[0, 0, 0, 0]],
    "chassis": [[0, 0, 0]]
  }
}
```

Notes:

- Image tensors are expected per camera as `[obs_steps, C, H, W]`.
- When loading from image files, the runtime converts them to `[1, C, H, W]`.
- State tensors are expected per key as `[obs_steps, dim]`.
- The current real-robot config uses `obs_steps=1`.

## CLI usage

Generate a request template without loading weights:

```bash
python scripts/server_inference.py \
  --dump_example_json /tmp/request.json
```

Run dummy inference:

```bash
python scripts/server_inference.py \
  --config configs/inference/r1lite_g0plus_pytorch.yaml
```

Run inference on a JSON request:

```bash
python scripts/server_inference.py \
  --config configs/inference/r1lite_g0plus_pytorch.yaml \
  --observation_json /path/to/request.json \
  --save_result_json /path/to/result.json
```

## Python integration

Later, a robot middleware layer can call the runtime directly:

```python
from galaxea_fm.inference import (
    GalaxeaInferenceRuntime,
    InferenceRuntimeConfig,
    PolicyObservation,
)

runtime = GalaxeaInferenceRuntime(
    InferenceRuntimeConfig.from_file("configs/inference/r1lite_g0plus_pytorch.yaml")
)
observation = PolicyObservation.from_json("/path/to/request.json")
result = runtime.predict(observation)
```

The recommended future integration boundary is:

- middleware receives robot observations
- middleware converts them to `PolicyObservation`
- runtime returns structured action tensors
- middleware publishes those actions back to the robot

This keeps robot IO separate from model inference.
