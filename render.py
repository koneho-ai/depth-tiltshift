#!/usr/bin/env python3
"""Depth-based tilt-shift: layered DOF with disk bokeh.

Implements the Kraus & Strengert (2007) / Wadhwa et al. (2018) layered
renderer; see README.md for the full reference list.

Why a naive blur-stack is wrong: blending Gaussian-blurred copies per pixel
is a gather using the *destination* pixel's blur. That mixes colours across
depth edges, so foreground colour bleeds into background bokeh and vice
versa. Defocus is a property of the *source* point, and a finite aperture
sees background that the pinhole view hides, so the correct fix is:

  1. Work in linear light (blur in sRGB darkens highlights).
  2. Split the image into depth bands (uniform in diopters = disparity here,
     since our depth map is disparity-like), with soft overlapping tent
     weights (Wadhwa Eq. 9).
  3. Per band: cull nearer pixels (foreground occluders), inpaint the holes
     with push-pull pyramid fill (Strengert et al. 2006) so the blur has
     plausible background to draw from, and set filled alpha to 1.
  4. Blur each band's premultiplied RGBA with a DISK kernel (the aperture
     imaged), not a Gaussian. The disc is flat in the centre and falls off
     over its outer edge (apodized, see disk_kernel): a perfectly flat disc
     with a hard rim makes a blurred foreground object's silhouette read as
     a visible outline against sharp background. Optional polygon/rim for
     shaped bokeh.
  5. Composite far -> near with the premultiplied over operator.

Tilt support: bands and CoC are computed in plane-relative depth
d' = d - (tilt_x*x + tilt_y*y), so a tilted focus plane becomes flat in d'
while occlusion ordering (by true depth) stays consistent. tilt=0 is pure
depth-of-field blur: one subject distance in focus, everything else blurred
by its depth distance from it.

Usage:
    python render.py photo.jpg depth.npy \
        --out out.jpg --focus-xy 0.5 0.5 --aperture 3
"""
import argparse
import cv2
import numpy as np
from PIL import Image, ImageEnhance

from imload import load_bgr


# --- colour space -----------------------------------------------------------

def srgb_to_linear(c):
    """sRGB [0,1] -> linear light, piecewise EOTF."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c,
                    1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


# --- kernels ----------------------------------------------------------------

def disk_kernel(r, shape="disk", rim=0.0, blades=6, edge_soft=0.4):
    """Normalized disk (pillbox) kernel of radius r px, 1-px antialiased edge.

    The intensity profile (IDF) is flat in the centre and falls off over
    the outer ``edge_soft`` fraction of the radius (cosine/Hann falloff,
    zero slope at both ends) -- the soft-edged, centre-weighted disc of
    an under-corrected lens ("creamy" bokeh, Bühler & Wexler 2002).
    A perfectly flat disc (edge_soft=0) makes a blurred foreground
    object's silhouette read as a visible outline against sharp
    background: the layer alpha falls almost linearly past the edge and
    hits zero with a kink (measured 2026-11-22 on the canyon sweep:
    alpha 0.52 at the spire edge, 0.21 at +15 px, 0.01 at +30 px -- the
    correct flat-disc profile, but the eye reads the kink as an edge).
    Apodizing the edge makes the feather S-shaped with a smooth landing,
    so the blurred object dissolves into the background with no outline.
    edge_soft=0 recovers the flat pillbox (geometric bokeh discs, hard
    rim). Optional bright rim (over-corrected spherical aberration look)
    or an N-blade polygon aperture (apothem form).
    """
    k = int(np.ceil(r))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    rad = np.sqrt(xx ** 2 + yy ** 2).astype(np.float32)
    if shape == "polygon":
        ang = np.arctan2(yy, xx)
        ap = np.pi / blades
        rr = r * np.cos(ap) / np.cos((ang + ap / 2) % (2 * ap) - ap)
        w = (rad <= rr).astype(np.float32)
        w *= np.clip(rr + 0.5 - rad, 0, 1)  # antialiased edge
        r_edge = rr  # falloff follows the polygon's local radius
    else:
        w = (rad <= r).astype(np.float32)
        w *= np.clip(r + 0.5 - rad, 0, 1)  # antialiased edge
        r_edge = r
    if edge_soft > 0:
        t = np.clip((rad - r_edge * (1 - edge_soft))
                    / (r_edge * edge_soft + 1e-6), 0, 1)
        w *= 0.5 * (1 + np.cos(np.pi * t))  # Hann falloff, smooth landing
    if rim:
        w *= 1.0 + rim * np.clip((rad / max(r, 1e-6) - 0.7) / 0.3, 0, 1) ** 2
    return (w / w.sum()).astype(np.float32)


# --- hole filling -----------------------------------------------------------

def push_pull_fill(img, valid):
    """Fill invalid pixels by pull (weighted downsample) + push (upsample).

    img: HxWxC float32, valid: HxW float32 0/1. CPU analogue of Strengert
    et al.'s pyramidal interpolation of scattered data (Kraus & Strengert
    2007 §3.2). Valid pixels are returned unchanged.
    """
    pyr_img, pyr_w = [img * valid[..., None]], [valid.astype(np.float32)]
    while min(pyr_img[-1].shape[:2]) > 8 and pyr_w[-1].mean() < 0.999 \
            and len(pyr_img) < 9:
        pyr_img.append(cv2.pyrDown(pyr_img[-1]))
        pyr_w.append(cv2.pyrDown(pyr_w[-1]))
    out = pyr_img[-1] / np.maximum(pyr_w[-1], 1e-6)[..., None]
    for lvl in range(len(pyr_img) - 2, -1, -1):
        up = cv2.pyrUp(out, dstsize=(pyr_img[lvl].shape[1], pyr_img[lvl].shape[0]))
        w = pyr_w[lvl][..., None]
        coarse = pyr_img[lvl] / np.maximum(w, 1e-6)
        out = np.where(w > 0.5, coarse, up)
    return np.where(valid[..., None] > 0.5, img, out)


# --- main renderer ----------------------------------------------------------

def prepare_layers(img_bgr, depth, tilt_x=0.0, tilt_y=0.0, bands=12,
                   overlap=0.25):
    """Focus-independent part of the pipeline (steps 0-1, 3-5).

    Returns (drel, lin, layers): plane-relative depth, linearized image, and
    layers sorted far -> near as (center, rgba_filled) with premultiplied
    float32 RGBA, holes push-pull filled. Only the per-band blur radius
    depends on focus, so a focus sweep reuses these layers for every frame.
    """
    h, w = depth.shape
    # Step 1: despeckle depth (edge-aware smoothing is ideal; median is safe
    # because our depth comes from a smooth bilinear upsample).
    try:
        depth = cv2.medianBlur(depth.astype(np.float32), 3)
    except cv2.error:
        pass

    # Plane-relative depth: a tilted focus plane becomes flat in d'.
    xs, ys = np.meshgrid(np.linspace(-1, 1, w, dtype=np.float32),
                         np.linspace(-1, 1, h, dtype=np.float32))
    drel = depth - (tilt_x * xs + tilt_y * ys)

    # Step 0: linearize.
    lin = srgb_to_linear(img_bgr.astype(np.float32) / 255.0)

    # Step 3: bands uniform in d' (diopters).
    edges = np.linspace(drel.min(), drel.max(), bands + 1)
    layers = []
    for j in range(bands):
        lo, hi = edges[j], edges[j + 1]
        eta = max(overlap * (hi - lo), 1e-9)
        a = np.clip(1 + np.minimum(drel - lo, hi - drel) / eta, 0, 1)
        center = 0.5 * (lo + hi)
        rgba = np.dstack([lin * a[..., None], a]).astype(np.float32)
        # Step 4: occluders are pixels NEARER than this band (d' above
        # the band + taper). Farther pixels stay transparent: they are
        # behind this layer and handled by farther layers. (The report's
        # sketch used a<0.01 as the hole, which wrongly includes farther
        # pixels; culling only nearer pixels is the Kraus §3.1 rule.)
        hole = (drel > hi + eta).astype(np.uint8)
        if hole.any():
            # Step 5: push-pull fill the occluder footprint ONLY -- do
            # NOT dilate the hole. An earlier version dilated by r_j,
            # which made hallucinated background opaque for r_j past the
            # silhouette; the blur then spread it another r_j, giving a
            # ~2*r_j wide opaque halo around the rock (measured 2026-11-22:
            # band 1 alpha 0.4-1.0 up to 36 px from the rock edge). With
            # no dilation the disk blur alone gives the correct partial
            # occlusion: alpha 0.5 at the silhouette, 0 at r_j outside.
            # Every band is filled, including the in-focus one: its fill
            # only shows through where nearer bands feather, and there it
            # is sharp focus-depth content, which is what a real lens sees.
            content = (a > 0.01).astype(np.float32)
            valid = content * (1 - hole.astype(np.float32))
            if valid.mean() > 0.001:
                filled = push_pull_fill(rgba, valid)
                # Filled region is hallucinated background at this depth:
                # opaque, un-premultiplied colour re-premultiplied by 1.
                fa = np.maximum(filled[..., 3], 1e-6)
                hm = hole > 0
                rgba[hm, :3] = np.clip(filled[hm, :3] / fa[hm, None], 0, None)
                rgba[hm, 3] = 1.0
        layers.append((center, rgba))
    layers.sort(key=lambda t: t[0])  # ascending d' = far first
    return drel, lin, layers


def render_focus(lin, layers, focus, aperture=3.0, r_max=30.0,
                 shape="disk", rim=0.0, blades=6, edge_soft=0.4):
    """Per-frame part of the pipeline (steps 6-8): blur each pre-filled
    layer by its focus-dependent radius and composite far -> near.

    Strength maps diopter distance to disk radius in px: strength =
    aperture * 20, so aperture 5 gives r=20px at |dd|=0.2, the same reach
    as v1's max Gaussian sigma, with r_max as a cap.
    """
    strength = aperture * 20.0
    acc_rgb = np.zeros_like(lin)
    acc_a = np.zeros(lin.shape[:2], dtype=np.float32)
    for center, rgba in layers:
        r_j = min(float(strength * abs(center - focus)), r_max)
        # Step 6: same disk blur on premultiplied RGB and A together.
        b = rgba if r_j < 0.5 else cv2.filter2D(
            rgba, -1, disk_kernel(r_j, shape, rim, blades, edge_soft),
            borderType=cv2.BORDER_REFLECT)
        # Step 7: premultiplied over.
        acc_rgb = b[..., :3] + (1 - b[..., 3:4]) * acc_rgb
        acc_a = b[..., 3] + (1 - b[..., 3]) * acc_a

    rgb = acc_rgb / np.maximum(acc_a, 1e-6)[..., None]
    out = linear_to_srgb(np.clip(rgb, 0, 1))
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def dof_bokeh(img_bgr, depth, focus_xy=(0.5, 0.62), focus=None,
              tilt_x=0.0, tilt_y=0.0, aperture=3.0, bands=12, r_max=30.0,
              overlap=0.25, shape="disk", rim=0.0, blades=6,
              edge_soft=0.4):
    """img_bgr: HxWx3 uint8 BGR. depth: HxW float32, 1 = close (disparity-like).

    Returns HxWx3 uint8 BGR.
    """
    drel, lin, layers = prepare_layers(img_bgr, depth, tilt_x, tilt_y,
                                       bands, overlap)
    if focus is None:
        fx, fy = focus_xy
        focus = float(drel[int(fy * drel.shape[0]), int(fx * drel.shape[1])])
    print(f"focus (plane-relative depth) {focus:.3f} at {focus_xy}, "
          f"tilt_x {tilt_x} tilt_y {tilt_y}")
    return render_focus(lin, layers, focus, aperture, r_max, shape, rim,
                        blades, edge_soft)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("depth", help="depth .npy from make_depth.py (0..1, 1 = close)")
    ap.add_argument("--out", default="out.jpg")
    ap.add_argument("--focus", type=float, default=None)
    ap.add_argument("--focus-xy", type=float, nargs=2, default=[0.5, 0.62])
    ap.add_argument("--tilt-x", type=float, default=0.0)
    ap.add_argument("--tilt-y", type=float, default=0.0)
    ap.add_argument("--aperture", type=float, default=3.0)
    ap.add_argument("--bands", type=int, default=12)
    ap.add_argument("--r-max", type=float, default=30.0)
    ap.add_argument("--shape", choices=["disk", "polygon"], default="disk")
    ap.add_argument("--rim", type=float, default=0.0,
                    help="bright-rim strength 0..1 (soap-bubble bokeh)")
    ap.add_argument("--blades", type=int, default=6)
    ap.add_argument("--edge-soft", type=float, default=0.4,
                    help="apodization: fraction of the disc radius over "
                    "which the kernel edge falls off (0 = flat pillbox, "
                    "hard rim; 0.4 = soft creamy edge, default)")
    ap.add_argument("--saturation", type=float, default=1.2)
    ap.add_argument("--contrast", type=float, default=1.05)
    args = ap.parse_args()

    try:
        img = load_bgr(args.input)
    except ValueError:
        raise SystemExit(f"cannot read {args.input}")
    depth = np.load(args.depth).astype(np.float32)
    if depth.shape[:2] != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_LINEAR)

    out = dof_bokeh(img, depth, focus_xy=args.focus_xy, focus=args.focus,
                    tilt_x=args.tilt_x, tilt_y=args.tilt_y,
                    aperture=args.aperture, bands=args.bands,
                    r_max=args.r_max, shape=args.shape, rim=args.rim,
                    blades=args.blades, edge_soft=args.edge_soft)
    cv2.imwrite(args.out, out, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Miniature colour lift (same as v1).
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Color(pil).enhance(args.saturation)
    pil = ImageEnhance.Contrast(pil).enhance(args.contrast)
    pil.save(args.out, quality=92)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
