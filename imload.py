"""Image loading for the depth-tiltshift tools, including JPEG XL and HEIC.

OpenCV and Pillow cannot read .jxl or .heic on their own. .jxl goes
through ffmpeg's libjxl decoder into an in-memory PNG (ffmpeg is already
a dependency of this project for video encoding). .heic/.heif goes
through pillow-heif (libheif) — NOT ffmpeg: these files are tiled HEIF
grids (4000x3000 of 512x512 tiles), and ffmpeg's grid stitching is
version-dependent — ffmpeg 7.1.4 silently decodes only the first tile
(512x512), which produced cropped 512x512 stills and sweeps for two
images on 2026-11-24, while ffmpeg 8.1.2 stitches correctly. libheif
stitches the grid on every version, so HEIC requires pillow-heif
(`pip install pillow-heif`) and fails loudly if it is missing.

Notes:
- Output is always 8-bit sRGB: 10/12/16-bit JXL/HEIC is scaled down and
  alpha is dropped. That matches what the depth model and the renderer
  consume.
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
    return str(path).lower().endswith((".jxl", ".jxls"))


def _is_heic(path):
    return str(path).lower().endswith((".heic", ".heif"))


def _load_heic_pil(path):
    """Load a HEIC/HEIF via pillow-heif (libheif stitches tiled grids).

    Raises a clear error if pillow-heif is missing — falling back to
    ffmpeg would silently produce a single-tile crop on ffmpeg < 8.
    """
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        raise RuntimeError(
            f"HEIC file {path} needs pillow-heif "
            "(pip install pillow-heif); ffmpeg's HEIF grid support is "
            "version-dependent and old versions decode only one tile")
    register_heif_opener()
    return Image.open(path).convert("RGB")


def load_pil(path):
    """Load an image as a PIL RGB image. .jxl via ffmpeg, .heic via libheif."""
    if _is_heic(path):
        return _load_heic_pil(path)
    if _needs_ffmpeg(path):
        return Image.open(io.BytesIO(_decode_via_ffmpeg(path))).convert("RGB")
    return Image.open(path).convert("RGB")


def load_bgr(path):
    """Load an image as a uint8 BGR numpy array (OpenCV convention).

    .jxl via ffmpeg, .heic via libheif. Raises SystemExit-style
    ValueError when the file cannot be read, matching cv2.imread's None
    contract.
    """
    if _is_heic(path):
        img = cv2.cvtColor(np.array(_load_heic_pil(path)),
                           cv2.COLOR_RGB2BGR)
    elif _needs_ffmpeg(path):
        data = np.frombuffer(_decode_via_ffmpeg(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img


def save_jxl(img, path, distance=1.0, effort=7):
    """Encode a PIL RGB image as JPEG XL via ffmpeg's libjxl encoder.

    `distance` is the Butteraugli target: 1.0 is visually lossless
    (the libjxl default), 0 is mathematically lossless. Pillow cannot
    write JXL on this platform, so the image goes through ffmpeg as PNG
    on stdin. 8-bit sRGB in, 8-bit JXL out.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    cmd = [
        "ffmpeg", "-v", "error", "-i", "pipe:0",
        "-c:v", "libjxl", "-distance", str(distance),
        "-effort", str(effort), "-y", str(path),
    ]
    proc = subprocess.run(cmd, input=buf.getvalue(),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to encode {path}: "
            f"{proc.stderr.decode(errors='replace').strip()}")


def save_pil(img, path, quality=92):
    """Save a PIL RGB image, inferring the format from the extension.

    `.jxl` goes through ffmpeg/libjxl (distance 1.0, visually
    lossless); anything else uses Pillow directly.
    """
    if str(path).lower().endswith((".jxl", ".jxls")):
        save_jxl(img, path)
    else:
        img.save(path, quality=quality)
