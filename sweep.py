#!/usr/bin/env python3
"""Focus-sweep video: pull focus from near to far, one frame per focus depth.

Layers are prepared once (focus-independent) and each frame only re-blurs
and re-composites, so a sweep costs ~1 render + N cheap frames.

Usage:
    .venv/bin/python sweep.py work/canyon.jpg work/canyon_depth.npy \
        --out-dir work/sweep --frames 60 --focus-near 0.70 --focus-far 0.03 \
        --aperture 3.0 --shape polygon --blades 6
    ffmpeg -y -framerate 24 -i work/sweep/frame_%03d.png -vf scale=1264:2048 \
        -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p \
        -movflags +faststart work/canyon_focus_sweep.mp4
"""
import argparse
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from imload import load_bgr
from render import prepare_layers, render_focus


def smoothstep(t):
    return t * t * (3 - 2 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("depth")
    ap.add_argument("--out-dir", default="work/sweep")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--focus-near", type=float, default=0.70)
    ap.add_argument("--focus-far", type=float, default=0.03)
    ap.add_argument("--aperture", type=float, default=3.0)
    ap.add_argument("--bands", type=int, default=12)
    ap.add_argument("--r-max", type=float, default=30.0)
    ap.add_argument("--shape", choices=["disk", "polygon"], default="polygon")
    ap.add_argument("--blades", type=int, default=6)
    ap.add_argument("--edge-soft", type=float, default=0.4,
                    help="apodization: fraction of the disc radius over "
                    "which the kernel edge falls off (0 = flat pillbox)")
    ap.add_argument("--saturation", type=float, default=1.2)
    ap.add_argument("--contrast", type=float, default=1.05)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    try:
        img = load_bgr(args.input)
    except ValueError:
        raise SystemExit(f"cannot read {args.input}")
    depth = np.load(args.depth).astype(np.float32)
    if depth.shape[:2] != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_LINEAR)

    t0 = time.time()
    drel, lin, layers = prepare_layers(img, depth, 0.0, 0.0,
                                       args.bands, 0.25)
    print(f"layers prepared in {time.time() - t0:.1f}s "
          f"(depth range {drel.min():.3f}..{drel.max():.3f})", flush=True)

    for i in range(args.frames):
        out_path = os.path.join(args.out_dir, f"frame_{i:03d}.png")
        if os.path.exists(out_path):
            print(f"frame {i:03d} exists, skipping", flush=True)
            continue
        t = i / max(args.frames - 1, 1)
        focus = args.focus_near + (args.focus_far - args.focus_near) * smoothstep(t)
        ft0 = time.time()
        frame = render_focus(lin, layers, focus, args.aperture, args.r_max,
                             args.shape, 0.0, args.blades, args.edge_soft)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil = ImageEnhance.Color(pil).enhance(args.saturation)
        pil = ImageEnhance.Contrast(pil).enhance(args.contrast)
        pil.save(out_path)
        print(f"frame {i:03d}/{args.frames - 1} focus {focus:.3f} "
              f"({time.time() - ft0:.1f}s)", flush=True)

    print(f"done: {args.frames} frames at {args.fps} fps = "
          f"{args.frames / args.fps:.1f}s", flush=True)


if __name__ == "__main__":
    main()
