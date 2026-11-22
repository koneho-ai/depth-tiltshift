#!/usr/bin/env python3
"""Generate a monocular depth map with Depth Anything V2 (Hugging Face).

Usage:
    .venv/bin/python make_depth.py work/input.jpg --out work/depth
    .venv/bin/python make_depth.py work/input.jpg --model depth-anything/Depth-Anything-V2-Large-hf

Outputs:
    <out>.npy  float32 depth, 0..1, 1 = closest (keep this, no banding)
    <out>.png  8-bit preview, bright = close
    <out>_16.png  16-bit depth, bright = close

Default model is Small (24.8M params) so it runs on CPU in this VM.
Large (335M) is sharper; use it when you have RAM/time to spare.
"""
import argparse
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input image (jpg/png)")
    ap.add_argument("--out", default="work/depth", help="output prefix (no extension)")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    args = ap.parse_args()

    img = Image.open(args.input).convert("RGB")
    print(f"input {args.input} {img.size}")

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model)
    model.eval()

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        # predicted_depth is (batch, H, W) — no channel dim.
        depth = model(**inputs).predicted_depth[0]

    # Upscale to full input resolution.
    depth = torch.nn.functional.interpolate(
        depth[None, None], size=img.size[::-1], mode="bilinear", align_corners=False
    )[0, 0].numpy().astype(np.float32)

    # Normalize to 0..1. Depth Anything outputs disparity-like values:
    # larger = closer, so bright = close after this normalization.
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    np.save(f"{args.out}.npy", depth)
    Image.fromarray((depth * 255).astype(np.uint8)).save(f"{args.out}.png")
    import cv2  # Pillow rejects explicit I;16 mode here; cv2 writes 16-bit PNG fine
    cv2.imwrite(f"{args.out}_16.png", (depth * 65535).astype(np.uint16))
    print(f"saved {args.out}.npy / .png / _16.png")
    print(f"depth min {depth.min():.3f} max {depth.max():.3f} mean {depth.mean():.3f}")
    h, w = depth.shape
    for name, (fx, fy) in {"center": (0.5, 0.5), "lower-mid": (0.5, 0.65),
                           "foreground": (0.5, 0.9), "horizon": (0.5, 0.35)}.items():
        print(f"  depth at {name} ({fx}, {fy}): {depth[int(fy * h), int(fx * w)]:.3f}")


if __name__ == "__main__":
    main()
