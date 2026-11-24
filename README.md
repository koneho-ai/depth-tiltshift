# depth-tiltshift

Depth-aware tilt-shift and bokeh rendering from a single photo. Give it a
picture, it estimates a depth map with a monocular depth model, then renders
physically plausible depth of field: a sharp subject at one distance, real
disk (or polygonal) bokeh everywhere else, and no colour bleeding across
depth edges.

Unlike the classic tilt-shift trick (a gradient blur that only knows image
height), this blur knows *distance*: a tall rock stays sharp top to bottom
when it sits at the focus distance, while waves at the same image height but
a different distance blur out.

## How it works

A naive blur-stack (blend Gaussian-blurred copies per pixel) uses the
*destination* pixel's blur radius. Defocus is a property of the *source*
point, so that mixes colours across depth edges: foreground colour bleeds
into background bokeh and vice versa. This tool implements the layered
depth-of-field renderer from the literature (Kraus & Strengert 2007,
Wadhwa et al. 2018):

1. **Linearize** — blur and composite in linear light, not sRGB, so
   highlights keep their energy.
2. **Depth bands** — split the image into 48 bands uniform in diopters
   (disparity), with triangle weights centred on each band centre (a
   partition of unity): every pixel is a blend of its two nearest bands,
   so the sharp zone slides continuously through depth as focus moves
   instead of snapping from slice to slice. Large blur radii are
   computed at reduced resolution and upsampled (pyramidal blur), which
   keeps 48 bands affordable.
3. **Cull + inpaint** — per band, remove nearer pixels (foreground
   occluders) and push-pull inpaint the background hidden behind them, so
   the blur has plausible background to draw from. Only the occluder
   footprint is filled; dilating the hole first creates a visible halo
   around silhouettes.
4. **Disk blur** — blur each band's premultiplied RGBA with a uniform disk
   kernel (the aperture imaged), optionally polygonal for a bladed iris, or
   with a bright rim for soap-bubble bokeh. Gaussian is the wrong kernel.
5. **Composite far to near** with the premultiplied over operator. The
   in-focus band is composited from the full-resolution sharp image.

Tilt is supported too: bands and blur are computed in plane-relative depth,
so a tilted focus plane (the classic miniature look) stays consistent with
occlusion. With tilt set to zero you get pure depth-of-field blur.

## Requirements

- Python 3.11+
- `ffmpeg` on your `PATH` — needed for the focus-sweep video, and for
  JPEG XL (`.jxl`) and HEIC (`.heic`/`.heif`) input, decoded through
  ffmpeg's libjxl decoder and native HEVC decoder (check with
  `ffmpeg -decoders | grep -E 'libjxl|hevc'`)
- About 2 GB RAM for a 1–3 MP image; CPU-only is fine

## Install

```bash
python -m venv .venv
source .venv/bin/activate

# Torch first (CPU build; ~200 MB). GPU users: install the CUDA build
# matching your driver from https://pytorch.org instead.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

The depth model (`depth-anything/Depth-Anything-V2-Small-hf`) downloads
from Hugging Face on first use (~100 MB) and is cached afterwards.

## Quickstart

```bash
# 1. Depth map for your photo (writes photo_depth.npy/.png/_16.png)
python make_depth.py photo.jpg --out photo_depth

# 2. Render: focus on the middle of the frame, aperture 3
python render.py photo.jpg photo_depth.npy --out out.jpg \
    --focus-xy 0.5 0.5 --aperture 3

# 3. Focus-sweep video: pull focus from near (0.70) to far (0.03)
python sweep.py photo.jpg photo_depth.npy --out-dir frames \
    --frames 60 --focus-near 0.70 --focus-far 0.03 \
    --aperture 3 --shape polygon --blades 6
ffmpeg -y -framerate 24 -i frames/frame_%03d.png \
    -vf scale=1264:2048 -c:v libx264 -preset slow -crf 16 \
    -pix_fmt yuv420p -movflags +faststart sweep.mp4
```

Timings on CPU (no GPU): depth ~20 s for a 1 MP image, a still render
~4–9 s, a 60-frame sweep a few minutes (layers are prepared once and reused
for every frame).

## Usage

### `make_depth.py` — depth map

```
python make_depth.py INPUT.jpg --out PREFIX [--model MODEL]
```

- `INPUT.jpg` — any photo (JPG/PNG/JXL/HEIC).
- `--out PREFIX` — output prefix; writes `PREFIX.npy` (float32 depth,
  0..1, **1 = closest**), `PREFIX.png` (8-bit preview, bright = close) and
  `PREFIX_16.png` (16-bit depth). Keep the `.npy` — it has no banding.
- `--model` — Hugging Face depth model. Default is
  `depth-anything/Depth-Anything-V2-Small-hf` (fast on CPU).
  `depth-anything/Depth-Anything-V2-Large-hf` is sharper but much slower
  on CPU.

The script prints sampled depths (centre, lower-mid, foreground, horizon)
to help you pick a focus point.

### `render.py` — still image

```
python render.py INPUT.jpg DEPTH.npy --out OUT.jpg [options]
```

`--out` can end in `.jxl` for a JPEG XL still (visually lossless,
roughly half the size of the JPEG at the same quality).

| Option | Default | Meaning |
|---|---|---|
| `--focus-xy X Y` | `0.5 0.62` | Focus point as fractions of width/height (0..1). The focus depth is sampled from the depth map there. Ignored if `--focus` is given. |
| `--focus D` | — | Focus depth directly (0..1, 1 = close). |
| `--tilt-x TX` | `0.0` | Horizontal tilt of the focus plane, in depth units across the frame. |
| `--tilt-y TY` | `0.0` | Vertical tilt. Positive tilts the plane so the top of the frame focuses farther away. |
| `--aperture A` | `3.0` | Blur strength. Disk radius in px is `A · 20 · |depth − focus|`, capped by `--r-max`. 3 is gentle, 5 is strong. |
| `--bands K` | `48` | Depth bands. 48 keeps a focus sweep smooth (frame-to-frame change within 1.6x); fewer bands make the sharp zone snap from slice to slice. |
| `--r-max R` | `30.0` | Maximum blur radius in px. |
| `--shape` | `disk` | `disk` (wide-open aperture) or `polygon` (bladed iris). |
| `--blades N` | `6` | Polygon sides when `--shape polygon` (6 = hexagon). |
| `--rim R` | `0.0` | Bright-rim strength 0..1 (soap-bubble bokeh). |
| `--edge-soft S` | `0.4` | Apodization: fraction of the disc radius over which the kernel edge falls off. `0` is a flat pillbox (geometric bokeh discs, hard rim); `0.4` is the soft creamy edge of an under-corrected lens. A flat disc makes a blurred foreground object's silhouette read as a visible outline against sharp background. |
| `--saturation S` | `1.2` | Colour saturation lift for the miniature look. |
| `--contrast C` | `1.05` | Contrast lift. |

Examples:

```bash
# Sharp subject, everything else blurred (no tilt)
python render.py photo.jpg photo_depth.npy --out out.jpg \
    --focus-xy 0.35 0.45 --aperture 3

# Classic miniature look: focus band across the middle, tilted plane
python render.py photo.jpg photo_depth.npy --out mini.jpg \
    --focus-xy 0.5 0.62 --tilt-y 0.30 --aperture 5

# Hexagonal bokeh, strong blur
python render.py photo.jpg photo_depth.npy --out hex.jpg \
    --focus-xy 0.5 0.5 --aperture 5 --shape polygon --blades 6
```

### `sweep.py` — focus-pull video frames

```
python sweep.py INPUT.jpg DEPTH.npy --out-dir FRAMES [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--frames N` | `60` | Number of frames. |
| `--fps F` | `24` | Frames per second (only used for the duration message; pass the same value to ffmpeg). |
| `--focus-near D` | `0.70` | Focus depth of the first frame (near). |
| `--focus-far D` | `0.03` | Focus depth of the last frame (far). |
| `--aperture`, `--bands`, `--r-max`, `--shape`, `--blades`, `--edge-soft`, `--saturation`, `--contrast` | as `render.py` | Same meaning. |

Focus follows a smoothstep ease, so the pull lingers at each end. Frames
are written as `FRAMES/frame_000.png` … and existing frames are skipped,
so a run can be resumed after an interruption. Encode with ffmpeg:

```bash
ffmpeg -y -framerate 24 -i FRAMES/frame_%03d.png \
    -vf scale=W:H -c:v libx264 -preset slow -crf 16 \
    -pix_fmt yuv420p -movflags +faststart sweep.mp4
```

Replace `W:H` with your image size rounded to even numbers
(`yuv420p` needs even dimensions). `-crf 16` is visually near-lossless;
use `-crf 18`–`20` for a smaller file.

### Batch processing — `process_album.py` + `build_sheets.py`

To run a whole set of photos (a still + a sweep video per image):

```bash
# 1. Contact sheets, 8 photos per sheet, to pick each subject
python build_sheets.py album.json   # writes work/sheets/sheet_N.jpg

# 2. Record one focus point per photo in work/focus.json:
#    {"DSC_9758": [0.78, 0.55], "DSC_9801": [0.4, 0.6], ...}

# 3. Run the batch (resumable: skips outputs that already exist)
python process_album.py --album-json album.json
```

`album.json` is `{"assets": [{"id", "filename", "type"}, ...]}` and each
original is expected at `work/<stem>/<filename>` (one directory per photo,
so renders stay with their source). Per image the batch makes a 3000px
working copy + depth map, a still focused on the subject (hexagonal
aperture 3, saved as `.jxl`), a 1920px focus sweep (180 frames at 24 fps, 7.5 s,
hexagonal aperture 6, near→far from that image's depth percentiles), and
optionally uploads the still, the video, the depth maps (`.npy` + PNGs) and
the sweep frames to a NAS share (`--no-upload` skips that step;
`--only STEM` runs a single image; `--stills-only` skips the videos).
Sweep frames are kept locally and uploaded under `<stem>/frames/`. The
upload goes through the SMB helper scripts in `--smb-dir` (default
`~/workspace/synology-smb`); by default they talk to a forwarder on
localhost:14445, and on a host that can reach the NAS directly you can
point them at it with `--smb-host <nas-host> --smb-port 445` (which also
skips starting the forwarder). A still
that fails at 48 bands is retried at 24, then 12 bands — a 4:3 image at
3000px needs ~5 GB of band layers, which can be OOM-killed.

Timings on CPU: roughly 1 minute for depth + still, ~10 minutes for the
sweep video per image at 1920px.

## Tips

- **Picking focus.** Run `make_depth.py` first and read its sampled
  depths, or open the `*_16.png` depth preview. Focus on a subject, not on
  empty background: the depth at a sky pixel is 0 and everything else will
  blur into mush.
- **Tilt vs no tilt.** Tilt when the sharp slice should follow the ground
  (a beach, a street, a valley floor). No tilt when one subject should be
  crisp at its own distance (a rock, a person, a spire).
- **Bokeh shape needs highlights.** Disk vs hexagon only shows on small
  bright highlights (lights, sun glints, white water). On a scene with
  none, the two look almost the same.
- **Foreground edges.** When focus is far, a near object should melt into
  the background with no outline at all: its blurred image spreads past
  its own silhouette as a semi-transparent veil fading smoothly to zero.
  If the edge reads as a line, the blur radius is too small compared to
  the object — raise `--aperture` (and `--r-max`, so the cap doesn't bind)
  and keep `--edge-soft` at its 0.4 default. A flat disc (`--edge-soft 0`)
  makes the veil's outer boundary visible as a kink.
- **LDR limits.** Blurring a clipped (JPEG) highlight spreads its energy
  over the disc, so bokeh discs look dimmer than on a real camera. HDR
  input would fix this; it is not implemented.
- **Depth edges.** The depth map is upsampled from the model's native
  resolution, so silhouettes are a few pixels soft. A faint fringe can
  remain at high-contrast edges; edge-aware (guided) depth upsampling
  would tighten it.
- **JPEG XL and HEIC.** `.jxl` and `.heic`/`.heif` input works in all
  scripts, decoded via ffmpeg (see `imload.py`). Everything is converted
  to 8-bit sRGB, so 16-bit JXL/HEIC is scaled down and alpha is dropped;
  animated JXL and multi-image HEIC use the first frame. Tiled HEIC
  grids (as produced by phones) are stitched by the decoder; scaling is
  done with PIL afterwards, because ffmpeg's `-vf scale` conflicts with
  the tile-grid filtergraph. Wide-gamut (Display P3) files are passed
  through without a gamut remap, so colours may be slightly off — depth
  and bokeh are unaffected.

## Files

| File | What it is |
|---|---|
| `make_depth.py` | Depth map from a photo (Depth Anything V2). |
| `render.py` | Layered DOF renderer (stills). Also usable as a module: `prepare_layers()` once, then `render_focus()` per focus. |
| `sweep.py` | Focus-sweep frame renderer for video. |
| `imload.py` | Image loading shared by the scripts, including `.jxl` and `.heic` via ffmpeg. |
| `process_album.py` | Batch driver: still + sweep video per image, resumable, optional NAS upload. |
| `build_sheets.py` | Contact sheets (8 photos each) for picking focus points. |
| `requirements.txt` | Pinned dependencies (torch installed separately, see above). |

## References

- Kraus, M. & Strengert, M. (2007). Depth-of-Field Rendering by Pyramidal
  Image Processing. *Computer Graphics Forum* 26(3):645–654.
  https://www.cs.cit.tum.de/fileadmin/w00cfj/cg/Research/Publications/2007/Depth-of-Field_Rendering/eg07.pdf
- Wadhwa, N. et al. (2018). Synthetic Depth-of-Field with a Single-Camera
  Mobile Phone. *ACM Trans. Graph.* 37(4):64 (SIGGRAPH 2018).
  https://arxiv.org/abs/1806.04171
- Barsky, B. A. et al. (2005). Elimination of artifacts due to occlusion
  and discretization problems in image space blurring techniques.
  *Graphical Models* 67(6):584–599.
- Lee, S., Kim, G. J. & Choi, S. (2008). Real-Time Depth-of-Field
  Rendering Using Point Splatting on Per-Pixel Layers. *Computer Graphics
  Forum* 27(7):1955–1962.
- Demers, J. (2004). Depth of Field: A Survey of Techniques. *GPU Gems*,
  ch. 23.
- Strengert, M., Kraus, M. & Ertl, T. (2006). Pyramid Methods in
  GPU-Based Image Processing. VMV 2006.
- Potmesil, M. & Chakravarty, I. (1981). A Lens and Aperture Camera Model
  for Synthetic Image Generation. *Computer Graphics* 15(3):297–305.
- Peng, J. et al. (2022). MPIB: An MPI-Based Bokeh Rendering Framework
  for Realistic Partial Occlusion Effects. ECCV 2022.
  https://arxiv.org/abs/2207.08403

## License

MIT. See [LICENSE](LICENSE).
