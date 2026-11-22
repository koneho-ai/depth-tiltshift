#!/usr/bin/env python3
"""Batch tilt-shift for the Immich 'Tilt & Shift' album.

Per image (in its work/<stem>/ dir):
  1. working copy, long edge 3000 (JPG q2)
  2. depth map at 3000 (Depth Anything V2 Small)
  3. still, focus on the subject (from focus.json), hex aperture 3
  4. working copy, long edge 1920 + depth resized to match
  5. focus-sweep frames, 180 frames @ 24 fps (7.5 s), hex aperture 6,
     focus near->far from the depth percentiles of that image
  6. H.264 encode, frames deleted after
  7. still + video uploaded to the NAS (koneho/depth-tiltshift/<stem>/)

Every step skips work whose output already exists, so the batch is
resumable: re-run it and it picks up where it stopped. A failed image is
logged and skipped, not fatal to the batch.

Usage:
    .venv/bin/python process_album.py [--only STEM] [--no-upload]
    .venv/bin/python process_album.py --stills-only
    .venv/bin/python process_album.py --album-json list.json \
        --focus-json focus.json --work-dir work --smb-dir ~/smb

The album JSON is {"assets": [{"id", "filename", "type"}, ...]}; the
focus JSON maps filename stems to [fx, fy] focus points (fractions of
width/height). Use build_sheets.py to pick focus points from contact
sheets. The NAS upload needs the SMB helper scripts in --smb-dir and a
reachable share; pass --no-upload to skip it.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
VENV_PY = os.path.join(HERE, ".venv", "bin", "python")
ALBUM_JSON = os.path.expanduser("~/workspace/immich-api/tilt-shift-album.json")
FOCUS_JSON = os.path.join(HERE, "work", "focus.json")
SMB_DIR = os.path.expanduser("~/workspace/synology-smb")
LOG = os.path.join(WORK, "batch.log")

SWEEP_FRAMES = 180
SWEEP_FPS = 24


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, what):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed: {(r.stdout + r.stderr)[-500:]}")
    return r.stdout


def ensure_forwarder():
    """Start the SMB forwarder if port 14445 is not listening."""
    import socket
    s = socket.socket()
    try:
        s.connect(("127.0.0.1", 14445))
        s.close()
        return
    except OSError:
        pass
    subprocess.Popen(
        [os.path.join(SMB_DIR, ".venv", "bin", "python"), "forwarder.py",
         "14445", "hola.chihuahua-major.ts.net", "445"],
        cwd=SMB_DIR, stdout=open("/tmp/forwarder.log", "ab"),
        stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(30):
        time.sleep(1)
        try:
            s = socket.socket()
            s.connect(("127.0.0.1", 14445))
            s.close()
            return
        except OSError:
            continue
    raise RuntimeError("SMB forwarder did not come up")


def make_working_copy(src, dst, edge):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    # Load via imload (ffmpeg-decodes JXL/HEIC, incl. tiled HEIC grids)
    # and scale with PIL: ffmpeg -vf scale conflicts with the tile-grid
    # complex filtergraph, and -filter_complex [0:v] grabs a single tile.
    from imload import load_pil
    from PIL import Image
    img = load_pil(src)
    w, h = img.size
    if max(w, h) > edge:
        if w >= h:
            nw, nh = edge, round(h * edge / w)
        else:
            nw, nh = round(w * edge / h), edge
        nw -= nw % 2
        nh -= nh % 2
        img = img.resize((nw, nh), Image.LANCZOS)
    img.save(dst, quality=92)


def make_depth(img_path, out_prefix):
    if os.path.exists(out_prefix + ".npy"):
        return
    run([VENV_PY, os.path.join(HERE, "make_depth.py"), img_path,
         "--out", out_prefix], f"depth {img_path}")


def resize_depth(src_npy, dst_npy, img_path):
    if os.path.exists(dst_npy):
        return
    d = np.load(src_npy).astype(np.float32)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    d2 = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    np.save(dst_npy, d2)


def render_still(img, depth_npy, out, fx, fy):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return
    run([VENV_PY, os.path.join(HERE, "render.py"), img, depth_npy,
         "--out", out, "--focus-xy", str(fx), str(fy),
         "--shape", "polygon", "--blades", "6",
         "--aperture", "3", "--r-max", "60", "--edge-soft", "0.4"],
        f"still {out}")


def render_sweep(img, depth_npy, frames_dir, video_out):
    if os.path.exists(video_out) and os.path.getsize(video_out) > 0:
        return
    d = np.load(depth_npy).astype(np.float32)
    near = float(np.percentile(d, 98))
    far = float(np.percentile(d, 2))
    if near - far < 0.05:
        near = min(1.0, far + 0.05)
    run([VENV_PY, os.path.join(HERE, "sweep.py"), img, depth_npy,
         "--out-dir", frames_dir, "--frames", str(SWEEP_FRAMES),
         "--fps", str(SWEEP_FPS), "--focus-near", f"{near:.3f}",
         "--focus-far", f"{far:.3f}",
         "--aperture", "6", "--r-max", "90",
         "--shape", "polygon", "--blades", "6", "--edge-soft", "0.4"],
        f"sweep frames {frames_dir}")
    run(["ffmpeg", "-y", "-v", "error", "-framerate", str(SWEEP_FPS),
         "-i", os.path.join(frames_dir, "frame_%03d.png"),
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", video_out],
        f"encode {video_out}")
    shutil.rmtree(frames_dir, ignore_errors=True)


def upload_to_nas(stem, files):
    existing = [f for f in files if os.path.exists(f)]
    if not existing:
        return
    ensure_forwarder()
    run([os.path.join(SMB_DIR, ".venv", "bin", "python"),
         os.path.join(SMB_DIR, "smb_upload.py"),
         "--share", "koneho", "--remote-dir", f"depth-tiltshift/{stem}"]
        + existing, f"NAS upload {stem}")


def process(asset, focus, args):
    stem = os.path.splitext(asset["filename"])[0]
    d = os.path.join(WORK, stem)
    original = os.path.join(d, asset["filename"])
    if not os.path.exists(original):
        raise RuntimeError(f"original missing: {original}")
    t0 = time.time()
    img3000 = os.path.join(d, "work_3000.jpg")
    depth3000 = os.path.join(d, "depth_3000")
    img1920 = os.path.join(d, "work_1920.jpg")
    depth1920 = os.path.join(d, "depth_1920.npy")
    still = os.path.join(d, f"{stem}_still.jxl")
    video = os.path.join(d, f"{stem}_sweep.mp4")
    frames_dir = os.path.join(d, "sweep_frames")

    make_working_copy(original, img3000, 3000)
    make_depth(img3000, depth3000)
    fx, fy = focus.get(stem, [0.5, 0.5])
    render_still(img3000, depth3000 + ".npy", still, fx, fy)
    if not args.stills_only:
        make_working_copy(original, img1920, 1920)
        resize_depth(depth3000 + ".npy", depth1920, img1920)
        render_sweep(img1920, depth1920, frames_dir, video)
    if not args.no_upload:
        upload_to_nas(stem, [still, video] if not args.stills_only else [still])
    log(f"{stem}: done in {time.time() - t0:.0f}s")


def main():
    global ALBUM_JSON, FOCUS_JSON, WORK, SMB_DIR, LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="process only this stem")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--stills-only", action="store_true")
    ap.add_argument("--album-json", default=ALBUM_JSON)
    ap.add_argument("--focus-json", default=FOCUS_JSON)
    ap.add_argument("--work-dir", default=WORK)
    ap.add_argument("--smb-dir", default=SMB_DIR)
    args = ap.parse_args()

    ALBUM_JSON = os.path.expanduser(args.album_json)
    FOCUS_JSON = os.path.expanduser(args.focus_json)
    WORK = os.path.expanduser(args.work_dir)
    SMB_DIR = os.path.expanduser(args.smb_dir)
    LOG = os.path.join(WORK, "batch.log")

    with open(ALBUM_JSON) as f:
        assets = json.load(f)["assets"]
    focus = {}
    if os.path.exists(FOCUS_JSON):
        with open(FOCUS_JSON) as f:
            focus = json.load(f)
    if args.only:
        assets = [a for a in assets
                  if os.path.splitext(a["filename"])[0] == args.only]
        if not assets:
            raise SystemExit(f"no asset with stem {args.only}")
    log(f"batch start: {len(assets)} images "
        f"({'stills only' if args.stills_only else 'still + sweep'})")
    failed = []
    for i, a in enumerate(assets, 1):
        stem = os.path.splitext(a["filename"])[0]
        log(f"[{i}/{len(assets)}] {stem}")
        try:
            process(a, focus, args)
        except Exception as e:
            log(f"{stem}: FAILED: {e}")
            failed.append(stem)
    log(f"batch done: {len(assets) - len(failed)} ok, "
        f"{len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
