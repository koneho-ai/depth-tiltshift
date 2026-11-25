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
     since our depth map is disparity-like), with TRIANGLE weights centred
     on each band centre (a partition of unity, Wadhwa Eq. 9): every pixel
     is a blend of its two nearest bands, so the sharp zone slides
     continuously through depth instead of snapping from slice to slice.
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

from imload import load_bgr, save_pil


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
        # Antialiased edge ONLY -- no hard (rad <= rr) mask. The mask
        # made pixels enter the kernel abruptly at weight 0.5 as r grew,
        # so the blur jumped in pixel steps as focus swept (measured
        # 2026-11-22: frame-to-frame diffs modulating 2x with the kernel
        # grid). The clip alone is continuous in r: a pixel's weight
        # ramps 0 -> 1 as the edge passes it.
        w = np.clip(rr + 0.5 - rad, 0, 1)
        r_edge = rr  # falloff follows the polygon's local radius
    else:
        w = np.clip(r + 0.5 - rad, 0, 1)  # antialiased edge, no hard mask
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

def prepare_layers(img_bgr, depth, tilt_x=0.0, tilt_y=0.0, bands=48):
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

    # Step 3: bands uniform in d' (diopters), with TRIANGLE weights
    # centred on each band centre -- a partition of unity, so every
    # pixel is a blend of its two nearest bands and the sharp zone
    # slides continuously through depth as focus moves. The old
    # trapezoid weights (flat top 1.0 across each band, tapering only
    # 0.25x the band width outside) made each pixel belong to exactly
    # one band, so only band centres could ever be sharp: as focus
    # swept, sharpness snapped from slice to slice with dead zones
    # between (measured 2026-11-22 on DSC_9801: frame-to-frame diffs
    # spiking 4x every time focus crossed a centre). 48 bands keeps
    # the midpoint softness under ~1.5 px at aperture 6 and the
    # sweep's frame-to-frame change within 1.6x (measured).
    centers = np.linspace(drel.min(), drel.max(), bands)
    s = ((drel.max() - drel.min()) / max(bands - 1, 1)) if bands > 1 else 1.0
    s = max(float(s), 1e-9)
    layers = []
    for c in centers:
        a = np.clip(1 - np.abs(drel - c) / s, 0, 1)
        center = float(c)
        rgba = np.dstack([lin * a[..., None], a]).astype(np.float32)
        # Step 4: occluders are pixels NEARER than this band's triangle
        # support (d' above c + s). Farther pixels stay transparent:
        # they are behind this layer and handled by farther layers.
        # (The report's sketch used a<0.01 as the hole, which wrongly
        # includes farther pixels; culling only nearer pixels is the
        # Kraus §3.1 rule.)
        hole = (drel > c + s).astype(np.uint8)
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


def _blur_rgba(rgba, r, shape, rim, blades, edge_soft):
    """Disk-blur premultiplied RGBA, at a resolution matched to r.

    Large radii are blurred at reduced resolution and upsampled: the
    content is already that blurry, so nothing visible is lost, and a
    1/4-res blur costs 1/16 of full res. This keeps many-band renders
    affordable (the pyramidal idea from Kraus & Strengert 2007).
    """
    if r < 0.5:
        return rgba
    if r <= 12:
        scale = 1.0
    elif r <= 24:
        scale = 0.5
    else:
        scale = 0.25
    if scale == 1.0:
        return cv2.filter2D(
            rgba, -1, disk_kernel(r, shape, rim, blades, edge_soft),
            borderType=cv2.BORDER_REFLECT)
    h, w = rgba.shape[:2]
    small = cv2.resize(rgba, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    blurred = cv2.filter2D(
        small, -1, disk_kernel(r * scale, shape, rim, blades, edge_soft),
        borderType=cv2.BORDER_REFLECT)
    return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)


def varmatch_weight(drel, centers, j, focus, strength, r_max):
    """Variance-matched band weight for band j (smooth-DOF research, proposal 2).

    The linear triangle weights mix two adjacent bands' disk blurs, but a
    mixture of two disks is NOT a disk of the interpolated radius: its
    second moment (variance) is systematically too large, by
    t*(1-t)*dr^2 at blend fraction t. Near focus the adjacent radii differ
    hugely in relative terms (0 vs 2.6 vs 5.2 px at aperture 6), so the
    effective blur pulses once per band as depth crosses band centres --
    the striping artifact (measured 2026-11-24: period exactly one band
    spacing, ~3 gray levels on smooth sand).

    This remaps the weights so the mixture's second moment equals the
    target disk's: in the interval [c_j, c_{j+1}], w'_{j+1} =
    (r(d)^2 - r_j^2) / (r_{j+1}^2 - r_j^2), clipped to [0, 1]. The clip
    handles the interval containing the focus, where the target radius
    dips below both bands' radii and no convex mixture can match it
    (there the smaller-radius band is used whole, as before). Partition of
    unity is preserved: w'_j + w'_{j+1} = 1 in every interval. Where two
    adjacent bands have equal radii (focus exactly midway between
    centres, or both capped at r_max) the mixture is already exact and
    the linear weight is kept.
    """
    c = centers[j]
    s = centers[1] - centers[0] if len(centers) > 1 else 1.0

    def rad(cc):
        return min(strength * abs(cc - focus), r_max)

    r_j = rad(c)
    r_d = np.minimum(strength * np.abs(drel - focus), r_max)
    w = np.zeros(drel.shape, dtype=np.float32)
    if j > 0:
        c_lo = centers[j - 1]
        r_lo = rad(c_lo)
        m = (drel >= c_lo) & (drel <= c)
        denom = r_j * r_j - r_lo * r_lo
        if abs(denom) > 1e-12:
            w[m] = np.clip((r_d[m] ** 2 - r_lo * r_lo) / denom, 0, 1)
        else:
            w[m] = (drel[m] - c_lo) / s
    if j < len(centers) - 1:
        c_hi = centers[j + 1]
        r_hi = rad(c_hi)
        # Include the left endpoint (>= c): at d == c the below-interval
        # also writes 1, so there is no conflict, and excluding it would
        # leave band 0 with weight 0 at drel == drel.min() (e.g. sky
        # pixels at exactly 0.0), zeroing the whole composite there.
        m = (drel >= c) & (drel <= c_hi)
        denom = r_hi * r_hi - r_j * r_j
        if abs(denom) > 1e-12:
            w[m] = np.clip(1 - (r_d[m] ** 2 - r_j * r_j) / denom, 0, 1)
        else:
            w[m] = 1 - (drel[m] - c) / s
    return w


def render_focus(lin, layers, focus, aperture=3.0, r_max=30.0,
                 shape="disk", rim=0.0, blades=6, edge_soft=0.4,
                 varmatch=False, drel=None):
    """Per-frame part of the pipeline (steps 6-8): blur each pre-filled
    layer by its focus-dependent radius and composite far -> near.

    Strength maps diopter distance to disk radius in px: strength =
    aperture * 20, so aperture 5 gives r=20px at |dd|=0.2, the same reach
    as v1's max Gaussian sigma, with r_max as a cap.
    """
    strength = aperture * 20.0
    acc_rgb = np.zeros_like(lin)
    acc_a = np.zeros(lin.shape[:2], dtype=np.float32)
    centers = [c for c, _ in layers]
    s = centers[1] - centers[0] if len(centers) > 1 else 1.0
    if varmatch and drel is None:
        raise ValueError("varmatch needs drel (plane-relative depth)")
    for j, (center, rgba) in enumerate(layers):
        r_j = min(float(strength * abs(center - focus)), r_max)
        if varmatch:
            # Rescale this layer's premultiplied RGBA by w'/w (the
            # variance-matched weight over the triangle weight), so the
            # layer carries the remapped weight into the blur. Holes keep
            # factor 1 (their fill stays opaque); pixels with negligible
            # triangle weight keep factor 1 (their contribution is ~0
            # either way, and it avoids a 0/0 at the support edge).
            # The multiply copies the layer; the cached layers are reused
            # across sweep frames and must not be mutated.
            wprime = varmatch_weight(drel, centers, j, focus, strength,
                                     r_max)
            w_tri = np.clip(1 - np.abs(drel - center) / s, 0, 1)
            factor = np.ones(drel.shape, dtype=np.float32)
            m = w_tri > 1e-3
            factor[m] = wprime[m] / w_tri[m]
            rgba = rgba * factor[..., None]
        # Step 6: same disk blur on premultiplied RGB and A together.
        b = _blur_rgba(rgba, r_j, shape, rim, blades, edge_soft)
        # Step 7: premultiplied over.
        acc_rgb = b[..., :3] + (1 - b[..., 3:4]) * acc_rgb
        acc_a = b[..., 3] + (1 - b[..., 3]) * acc_a

    rgb = acc_rgb / np.maximum(acc_a, 1e-6)[..., None]
    out = linear_to_srgb(np.clip(rgb, 0, 1))
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _tap_pattern(shape="disk", blades=6, n_taps=25):
    """Fixed unit-shape tap offsets for the scatter-as-gather.

    A centre tap plus a sunflower (Fibonacci) spiral, which gives near
    uniform area coverage of the unit disk. For a polygon aperture each
    offset's radius is remapped to the polygon boundary in its direction
    (same vertex convention as _polygon_kernel: vertices at
    (i+0.5)*2*pi/blades), so the taps fill the polygon instead of the
    disk. Returns (K,2) float32 offsets with |offset| <= 1 in the
    shape's own metric.
    """
    offs = [(0.0, 0.0)]
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_taps - 1):
        rho = np.sqrt((i + 0.5) / (n_taps - 1))
        ang = i * golden
        offs.append((rho * np.cos(ang), rho * np.sin(ang)))
    offs = np.array(offs, dtype=np.float32)
    if shape == "polygon":
        # Boundary radius of the regular n-gon (circumradius 1) in
        # direction theta: apothem / cos(phi), phi = angle to the
        # nearest edge-midpoint direction (edge midpoints at i*2*pi/n).
        n = blades
        theta = np.arctan2(offs[:, 1], offs[:, 0])
        phi = (theta + np.pi / n) % (2 * np.pi / n) - np.pi / n
        r_bound = np.cos(np.pi / n) / np.cos(phi)
        r = np.hypot(offs[:, 0], offs[:, 1])
        scale = np.where(r > 0, r_bound, 1.0)
        offs = offs * scale[:, None]
    return offs


def _gather_dof(lin, drel, focus, strength, r_max, shape="disk", blades=6,
                edge_soft=0.4, n_taps=25, energy=True):
    """Per-pixel scatter-as-gather DOF (Hammon 2008, GPU Gems 3 Ch. 28;
    Jimenez 2014).

    Each output pixel gathers K taps laid out in its own CoC disk
    (radius r_p, a continuous function of depth -- no bands, so no
    banding). A tap contributes with weight apod(|p-q| / r_q), the
    source-side apodized disk profile -- the tap's light only reaches
    pixels inside its OWN disk, fading to its rim (Hammon's rule: a
    sample contributes only if its blur disk actually covers the
    destination pixel). With energy=True the weight is additionally
    divided by r_q^2, the energy dilution over the tap's disk area (a
    heavily blurred source is a faint wash, a sharp one concentrated)
    -- the physically exact scatter weight. Measured 2026-11-24 on the
    beach frame, energy=True biases the result toward the focus plane's
    brightness on depth gradients (~8 gray levels darker on the sand),
    because the gather samples only within r_p and misses the far wash
    the normalisation would otherwise balance; energy=False (Hammon's
    original form) avoids that bias at the cost of undiluted foreground
    spill, which is small in the near-focus zone. Normalised per pixel,
    so a flat field is preserved exactly. The radius is capped at r_max
    like the layered pipeline. Returns the gathered linear RGB image.
    """
    h, w = drel.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r_p = np.minimum(strength * np.abs(drel - focus), r_max).astype(np.float32)
    offs = _tap_pattern(shape, blades, n_taps)
    acc = np.zeros_like(lin)
    wsum = np.zeros((h, w), dtype=np.float32)
    for k in range(len(offs)):
        dx, dy = float(offs[k, 0]), float(offs[k, 1])
        rho = float(np.hypot(dx, dy))
        sx = xx + dx * r_p
        sy = yy + dy * r_p
        col = cv2.remap(lin, sx, sy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        d_tap = cv2.remap(drel, sx, sy, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
        r_q = np.minimum(strength * np.abs(d_tap - focus), r_max)
        # Source-side apodization: flat to (1-edge_soft) of the source
        # disk, squared-cosine falloff to the rim, zero outside. Same
        # profile as the layered kernel, evaluated on the SOURCE disk
        # (the tap's light fades toward the rim of its own disk).
        dist = rho * r_p
        t = np.clip((dist / np.maximum(r_q, 1e-6) - (1 - edge_soft))
                    / max(edge_soft, 1e-6), 0, 1)
        apod = (0.5 + 0.5 * np.cos(np.pi * t)) ** 2
        if energy:
            r_qc = np.maximum(r_q, 0.5)
            wt = apod / (r_qc * r_qc)
        else:
            wt = apod
        acc += col * wt[..., None]
        wsum += wt
    out = acc / np.maximum(wsum, 1e-6)[..., None]
    bad = wsum < 1e-6
    out[bad] = lin[bad]
    return out


def _guided_filter(guide, p, r=15, eps=1e-3):
    """Edge-preserving smoothing (He et al. 2013, gray guide).

    guide: HxW float32 (luminance of the sharp image); p: HxW float32
    single channel. Returns the filtered channel. In flat areas the
    output is the local mean (stripes averaged out); at strong edges
    (local variance >> eps) the output follows the guide, so colour
    does not smear across silhouettes. O(1) in r (box filters), unlike
    a bilateral whose diameter truncates the kernel -- a bilateral
    with d=31 (radius 15) truncates a sigma-15 kernel at 1 sigma and
    attenuates the 30-row stripes only ~4x, leaking them back into
    the hybrid (measured 2026-11-24: sand-strip amp 0.38 -> 1.41).
    eps=1e-3 sits between the stripes' variance (~4e-5, smoothed) and
    a real silhouette's (~0.05, preserved).
    """
    k = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.blur(guide, k)
    mean_p = cv2.blur(p, k)
    mean_II = cv2.blur(guide * guide, k)
    mean_Ip = cv2.blur(guide * p, k)
    var_I = mean_II - mean_I * mean_I
    cov = mean_Ip - mean_I * mean_p
    a = cov / (var_I + eps)
    b = mean_p - a * mean_I
    a = cv2.blur(a, k)
    b = cv2.blur(b, k)
    return a * guide + b


def render_focus_hybrid(lin, drel, layers, focus, aperture=3.0, r_max=30.0,
                        shape="disk", rim=0.0, blades=6, edge_soft=0.4,
                        zone_r=16.0, fade_r=20.0, n_taps=49, energy=True):
    """Hybrid DOF (smooth-DOF research, proposal 1): per-pixel
    scatter-as-gather near focus + the layered pipeline for the far
    field.

    The layered renderer's banding is worst near the focus plane, and
    that is exactly where a per-pixel gather is cheapest (small radii
    need few taps) and where its occlusion errors are smallest (small
    CoC). So: the gather result is used where the local CoC radius is
    below zone_r, the layered result where it is above zone_r + fade_r,
    with a cosine cross-fade in between (a smooth function of radius,
    so no transition band). The far field keeps the layered pipeline's
    culling + inpainting, which is where occlusion matters most.
    The gather ignores `rim` (bright-rim bokeh is layered-only).

    Zone calibration (2026-11-24, beach frame 175612 f120): the first
    zone (8/8 px, 25 taps) fixed the sand strip but left bands around
    the kid, whose sand spans CoC radii 0-29 px -- the r > 16 sand was
    still layered. Widening to zone_r=16, fade_r=20 (gather to 16 px,
    fading out by 36 px) with 49 taps removes them (30-row stripe amp
    1.26 -> 0.36 gray there, 3.04 -> 0.38 on the strip). A K-tap gather
    cannot suppress fine texture like a dense disk of radius >> sqrt(K)
    (measured: effective texture radius saturates ~3.7 px at 25 taps
    for nominal R = 8-16 px), but its EDGE blur matches the disk
    exactly (10-90% transition 8 px at R=8, 18 px at R=16) -- so
    silhouettes blur correctly and only fine texture stays slightly
    sharper than the layered version, which reads as natural grain.

    Two edge fixes (2026-11-24, same frame, user-spotted): (1) the
    gather under-spills foreground colour at depth discontinuities
    (kid's outline 11 px vs the layered 27 px) -- a radius-scaled
    depth-edge mask hands edges back to the layered pipeline (the mask
    extends ~one CoC radius from the edge, using the radius at the
    nearest edge pixel, propagated via distance-transform labels;
    edges are real depth jumps >= 0.08 with visible colour contrast,
    so pebbles stay with the gather); (2) the tonal correction's blur must be edge-preserving
    (guided filter, guide = sharp luminance) -- a Gaussian smeared the
    kid's pink 30-45 px onto the sand as a visible glow, and a
    bilateral truncates its kernel at its diameter and leaks the
    stripes back (sand-strip amp 0.38 -> 1.41).
    """
    layered = render_focus(lin, layers, focus, aperture, r_max, shape, rim,
                           blades, edge_soft)
    # render_focus returns sRGB uint8; the gather is linear float. The
    # cross-fade must happen in linear light, so convert the layered
    # result back (the 8-bit round trip costs <=0.5 LSB, invisible).
    layered_lin = srgb_to_linear(layered.astype(np.float32) / 255.0)
    strength = aperture * 20.0
    gather = _gather_dof(lin, drel, focus, strength, r_max, shape, blades,
                         edge_soft, n_taps, energy=energy)
    # Tonal correction: the gather's brightness drifts from the layered
    # result on depth/brightness gradients (measured 2026-11-24 on the
    # beach frame: ~8 gray levels darker than the layered render in the
    # near-focus sand, ~3 darker than the sharp image). The drift is
    # low-frequency, while the banding it replaces is a 30-row periodic
    # modulation, so force the gather's low frequencies to match the
    # layered result's and keep the gather's (stripe-free) high
    # frequencies: gather + (blur(layered) - blur(gather)). The blur
    # must be EDGE-PRESERVING (guided filter, guide = sharp luminance):
    # a plain Gaussian smears bright foreground colour across
    # silhouettes -- measured 2026-11-24, the kid's pink bled 30-45 px
    # onto the sand (+15-27 gray) as a visible glow -- and a bilateral
    # truncates its kernel at its diameter (d=31 keeps only +-1 sigma
    # of a sigma-15 kernel), attenuating the stripes just ~4x and
    # leaking them back into the hybrid (sand-strip amp 0.38 -> 1.41).
    # This also makes the cross-fade tonally seamless.
    guide = (0.299 * lin[..., 2] + 0.587 * lin[..., 1]
             + 0.114 * lin[..., 0]).astype(np.float32)
    low_lay = np.stack([_guided_filter(guide, layered_lin[..., c])
                        for c in range(3)], axis=-1)
    low_gat = np.stack([_guided_filter(guide, gather[..., c])
                        for c in range(3)], axis=-1)
    gather = gather + (low_lay - low_gat)
    r_p = np.minimum(strength * np.abs(drel - focus), r_max)
    t = np.clip((r_p - zone_r) / fade_r, 0, 1)
    fade = 0.5 + 0.5 * np.cos(np.pi * t)  # 1 (gather) -> 0 (layered)
    # Depth-edge mask: the gather under-spills foreground colour at
    # depth discontinuities (measured 2026-11-24 at the kid's
    # silhouette: outline transition 11 px in the hybrid vs 27 px in
    # the layered render -- the foreground taps are diluted 1/r_q^2
    # and the background pixel's own taps dominate, so the edge reads
    # too sharp). The layered pipeline gets depth edges right (its
    # spill is a dense disk blur), so hand depth edges back to it.
    # The mask extends about one CoC radius from the edge (the spill
    # width), using the radius at the nearest edge pixel (the
    # foreground's), propagated exactly via distance-transform labels
    # -- NOT by dilating the radius map, which leaks large foreground
    # radii across the whole frame and reintroduces banding on smooth
    # sand (measured: sand-strip 30-row amp 0.38 -> 1.41 with the
    # dilated mask). An edge is a real depth JUMP (morphological
    # gradient of drel > 0.08 in a 7x7 window) with visible colour
    # contrast (Sobel > 25, dilated 5x5): the jump test separates the
    # kid/ice (jump >= 0.1) from pebbles (<= 0.05) with no
    # connectivity needed -- the earlier Sobel-gradient edge
    # fragmented along the silhouette (colour peak 45, neighbours 39,
    # 1) and a component-size filter then deleted the kid's edge
    # entirely (mask 0.00 at the spill points). The radius at the edge
    # is dilated 15x15 so the foreground's radius (the kid's 38 px)
    # reaches across the 8-px-wide edge band to the background side.
    drange = cv2.morphologyEx(drel, cv2.MORPH_GRADIENT,
                              np.ones((7, 7), np.uint8))
    gray_u8 = cv2.cvtColor((np.clip(linear_to_srgb(lin), 0, 1) * 255
                            ).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    cgx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    cgy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    grad_c = np.sqrt(cgx * cgx + cgy * cgy) / 8.0
    cedge = (grad_c > 25.0).astype(np.uint8)
    cedge = cv2.dilate(cedge, np.ones((5, 5), np.uint8))
    edge = ((drange > 0.08) & (cedge == 1)).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(
        1 - edge, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    h, w = drel.shape
    r_edge = (r_p * edge).astype(np.float32)
    r_edge = cv2.dilate(r_edge, np.ones((15, 15), np.float32))
    # DIST_LABEL_PIXEL labels are 1-based ordinals of the zero pixels
    # in row-major scan order (verified 2026-11-24), NOT linear
    # indices -- map through the edge-pixel list. Some images have NO
    # depth edges at all (smooth depth field, no foreground occluder):
    # then eys is empty, r_at_edge is empty, and the clip below yields
    # -1, which crashes on the empty array (IndexError, hit on 8 redo
    # images). With no edges there is nothing to hand back to the
    # layered pipeline, so the edge mask is just zero.
    eys, exs = np.where(edge == 1)
    if len(eys) == 0:
        edge_mask = np.zeros_like(drel, dtype=np.float32)
    else:
        r_at_edge = r_edge[eys, exs]
        r_nearest = r_at_edge[np.clip(labels.reshape(-1), 1, len(eys)) - 1
                              ].reshape(h, w)
        edge_mask = np.clip((r_nearest + 6.0 - dist) / 12.0, 0, 1)
    gather_w = fade * (1.0 - edge_mask)
    out_lin = (gather * gather_w[..., None]
               + layered_lin * (1 - gather_w[..., None]))
    out = linear_to_srgb(np.clip(out_lin, 0, 1))
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def dof_bokeh(img_bgr, depth, focus_xy=(0.5, 0.62), focus=None,
              tilt_x=0.0, tilt_y=0.0, aperture=3.0, bands=48, r_max=30.0,
              shape="disk", rim=0.0, blades=6,
              edge_soft=0.4, varmatch=False, hybrid=True,
              zone_r=16.0, fade_r=20.0, n_taps=49):
    """img_bgr: HxWx3 uint8 BGR. depth: HxW float32, 1 = close (disparity-like).

    Returns HxWx3 uint8 BGR.
    """
    drel, lin, layers = prepare_layers(img_bgr, depth, tilt_x, tilt_y,
                                       bands)
    if focus is None:
        fx, fy = focus_xy
        focus = float(drel[int(fy * drel.shape[0]), int(fx * drel.shape[1])])
    print(f"focus (plane-relative depth) {focus:.3f} at {focus_xy}, "
          f"tilt_x {tilt_x} tilt_y {tilt_y}")
    if hybrid:
        return render_focus_hybrid(lin, drel, layers, focus, aperture,
                                   r_max, shape, rim, blades, edge_soft,
                                   zone_r, fade_r, n_taps)
    return render_focus(lin, layers, focus, aperture, r_max, shape, rim,
                        blades, edge_soft, varmatch=varmatch, drel=drel)


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
    ap.add_argument("--bands", type=int, default=48)
    ap.add_argument("--r-max", type=float, default=30.0)
    ap.add_argument("--shape", choices=["disk", "polygon"], default="disk")
    ap.add_argument("--rim", type=float, default=0.0,
                    help="bright-rim strength 0..1 (soap-bubble bokeh)")
    ap.add_argument("--blades", type=int, default=6)
    ap.add_argument("--edge-soft", type=float, default=0.4,
                    help="apodization: fraction of the disc radius over "
                    "which the kernel edge falls off (0 = flat pillbox, "
                    "hard rim; 0.4 = soft creamy edge, default)")
    ap.add_argument("--varmatch", action="store_true",
                    help="variance-matched band weights (EXPERIMENTAL, "
                    "largely ineffective: remaps the triangle weights so "
                    "each two-band mixture matches the target disk's "
                    "second moment, but that only cuts the near-focus "
                    "striping ~17%% -- the kernel-shape error remains; "
                    "see varmatch_weight)")
    ap.add_argument("--hybrid", dest="hybrid", action="store_true",
                    default=True,
                    help="hybrid DOF (default): per-pixel "
                    "scatter-as-gather near focus + layered pipeline "
                    "for the far field (removes the per-band striping; "
                    "see render_focus_hybrid)")
    ap.add_argument("--no-hybrid", dest="hybrid", action="store_false",
                    help="use the pure layered pipeline (faster, but "
                    "stripes smooth surfaces near the focus plane)")
    ap.add_argument("--zone-r", type=float, default=16.0,
                    help="hybrid: gather used fully below this CoC radius "
                    "(px), layered above zone-r + fade-r")
    ap.add_argument("--fade-r", type=float, default=20.0,
                    help="hybrid: cross-fade width in CoC radius (px)")
    ap.add_argument("--taps", type=int, default=49,
                    help="hybrid: gather tap count (25 is sparse; 49 "
                    "matches the wide zone)")
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
                    blades=args.blades, edge_soft=args.edge_soft,
                    varmatch=args.varmatch, hybrid=args.hybrid,
                    zone_r=args.zone_r, fade_r=args.fade_r, n_taps=args.taps)
    # Miniature colour lift (same as v1), then save (.jxl supported).
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Color(pil).enhance(args.saturation)
    pil = ImageEnhance.Contrast(pil).enhance(args.contrast)
    save_pil(pil, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
