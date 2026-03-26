#!/usr/bin/env python3
"""
Minimal inference demo for GalaxeaVLA G0Plus model
This script demonstrates how to load a model and run a single inference step.

Usage:
    python inference_demo.py --ckpt_path /path/to/checkpoint.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Import model and processor
from galaxea_fm.models.galaxea_zero.galaxea_zero_policy import GalaxeaZeroPolicy
from galaxea_fm.processors.galaxea_zero_processor import GalaxeaZeroProcessor
from transformers import AutoTokenizer


def create_dummy_batch(device="cuda"):
    """Create a dummy batch for testing inference"""
    batch_size = 1
    obs_steps = 1

    # Dummy RGB images (3 cameras: head, left_wrist, right_wrist)
    dummy_images = {
        "head_rgb": torch.randn(batch_size, obs_steps, 3, 224, 224).to(device),
        "left_wrist_rgb": torch.randn(batch_size, obs_steps, 3, 224, 224).to(device),
        "right_wrist_rgb": torch.randn(batch_size, obs_steps, 3, 224, 224).to(device),
    }

    # Dummy instruction
    instruction = "Pick up the red cup"

    # Dummy proprioception (joint positions, gripper states, etc.)
    # Shape: (batch_size, obs_steps, proprio_dim=21)
    dummy_proprio = torch.randn(batch_size, obs_steps, 21).to(device)

    batch = {
        "observation.images": dummy_images,
        "instruction": [instruction],
        "observation.state": dummy_proprio,
    }

    return batch


def load_model(ckpt_path: str, device="cuda"):
    """Load G0Plus model from checkpoint"""
    print(f"Loading model from: {ckpt_path}")

    # Create model instance (configuration should match your checkpoint)
    model = GalaxeaZeroPolicy(
        pretrained_model_path="/data/google/paligemma-3b-pt-224",
        action_dim=26,  # 2 x [6 arm joints + 1 gripper] + 6 torso + 6 chassis
        proprio_dim=21,
        cond_steps=1,
        horizon_steps=10,
        num_input_images=3,  # head + left_wrist + right_wrist
    )

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    print("Model loaded successfully!")
    return model


def main():
    parser = argparse.ArgumentParser(description="G0Plus Inference Demo")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda/cpu)"
    )
    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Using device: {args.device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if args.device == 'cuda' else 'N/A'}")

    # Load model
    model = load_model(args.ckpt_path, device=args.device)

    # Create dummy input batch
    print("\nCreating dummy input batch...")
    batch = create_dummy_batch(device=args.device)

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        output = model.predict_action(batch)

    # Print results
    if "action" in output:
        predicted_action = output["action"]
        print(f"\nInference successful!")
        print(f"Output action shape: {predicted_action.shape}")
        print(f"Action values (first timestep): {predicted_action[0, 0, :].cpu().numpy()}")
        print(f"\nAction breakdown (26-dim):")
        print(f"  - Left arm joints (6):  {predicted_action[0, 0, :6].cpu().numpy()}")
        print(f"  - Left gripper (1):     {predicted_action[0, 0, 6].item():.4f}")
        print(f"  - Right arm joints (6): {predicted_action[0, 0, 7:13].cpu().numpy()}")
        print(f"  - Right gripper (1):    {predicted_action[0, 0, 13].item():.4f}")
        print(f"  - Torso velocity (6):   {predicted_action[0, 0, 14:20].cpu().numpy()}")
        print(f"  - Chassis velocity (6): {predicted_action[0, 0, 20:26].cpu().numpy()}")
    else:
        print("Warning: No 'action' key in output!")
        print(f"Available keys: {output.keys()}")

    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Replace dummy data with real robot observations")
    print("2. Use the predicted actions to control your robot")
    print("3. See scripts/eval_open_loop.py for batch inference examples")


if __name__ == "__main__":
    main()
