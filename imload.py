"""Image loading for the depth-tiltshift tools, including JPEG XL and HEIC.

OpenCV and Pillow cannot read .jxl or .heic on their own (pillow-jxl has
no wheel for this platform, pillow-heif is not installed), but the system
ffmpeg ships with the libjxl decoder and a native HEVC decoder, and ffmpeg
is already a dependency of this project (video encoding). So .jxl and
.heic/.heif files are decoded through ffmpeg into an in-memory PNG, then
loaded normally. Everything else goes through the usual loaders.

Notes:
- Output is always 8-bit sRGB: ffmpeg decodes to rgb24, so 10/12/16-bit
  JXL/HEIC is scaled down and alpha is dropped. That matches what the
  depth model and the renderer consume.
- Wide-gamut (Display P3) JXL/HEIC is passed through without a gamut
  remap, so colours may be slightly off; depth and bokeh are unaffected.
- Animated JXL / multi-image HEIC: only the first frame is used.
"""
import io
import subprocess

import cv2
import numpy as np
from PIL import Image


def _decode_via_ffmpeg(path):
    """Decode a .jxl/.heic file to PNG bytes via ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-frames:v", "1", "-pix_fmt", "rgb24",
        "-f", "image2pipe", "-vcodec", "png", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to decode {path}: "
            f"{proc.stderr.decode(errors='replace').strip()}")
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg produced no output for {path}")
    return proc.stdout


def _needs_ffmpeg(path):
    return str(path).lower().endswith((".jxl", ".jxls", ".heic", ".heif"))


def load_pil(path):
    """Load an image as a PIL RGB image. .jxl/.heic via ffmpeg."""
    if _needs_ffmpeg(path):
        return Image.open(io.BytesIO(_decode_via_ffmpeg(path))).convert("RGB")
    return Image.open(path).convert("RGB")


def load_bgr(path):
    """Load an image as a uint8 BGR numpy array (OpenCV convention).

    .jxl/.heic via ffmpeg. Raises SystemExit-style ValueError when the
    file cannot be read, matching cv2.imread's None contract.
    """
    if _needs_ffmpeg(path):
        data = np.frombuffer(_decode_via_ffmpeg(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img
