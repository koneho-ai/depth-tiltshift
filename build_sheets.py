#!/usr/bin/env python3
"""Build contact sheets of the Tilt & Shift album for focus picking.

Usage:
    python3 build_sheets.py [album.json]

Makes a 512px preview of each original, then montages 8 per sheet
(2 cols x 4 rows) with the filename stem as a label. Sheets go to
work/sheets/sheet_N.jpg. Run after download_album.py.
"""
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
SHEETS = os.path.join(WORK, "sheets")
CELL_W, CELL_H, LABEL_H = 512, 384, 28
COLS, ROWS = 2, 4
PER_SHEET = COLS * ROWS


def preview(src, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    # Load via imload (ffmpeg-decodes JXL/HEIC, incl. tiled HEIC grids)
    # and scale with PIL: ffmpeg -vf scale conflicts with the tile-grid
    # complex filtergraph, and -filter_complex [0:v] grabs a single tile.
    from imload import load_pil
    img = load_pil(src)
    img.thumbnail((512, 512))
    img.save(dst, quality=85)


def main():
    list_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/workspace/immich-api/tilt-shift-album.json")
    with open(list_path) as f:
        assets = json.load(f)["assets"]
    os.makedirs(SHEETS, exist_ok=True)
    os.makedirs(os.path.join(SHEETS, "previews"), exist_ok=True)

    items = []
    for a in assets:
        stem = os.path.splitext(a["filename"])[0]
        src = os.path.join(WORK, stem, a["filename"])
        if not os.path.exists(src):
            print(f"missing original: {a['filename']}")
            continue
        pv = os.path.join(SHEETS, "previews", stem + ".jpg")
        preview(src, pv)
        items.append((stem, pv))
    print(f"{len(items)} previews ready")

    for s in range(0, len(items), PER_SHEET):
        chunk = items[s:s + PER_SHEET]
        sheet = Image.new("RGB",
                          (COLS * CELL_W, ROWS * (CELL_H + LABEL_H)),
                          (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        for i, (stem, pv) in enumerate(chunk):
            col, row = i % COLS, i // COLS
            x0, y0 = col * CELL_W, row * (CELL_H + LABEL_H)
            img = Image.open(pv)
            img.thumbnail((CELL_W, CELL_H))
            sheet.paste(img, (x0 + (CELL_W - img.width) // 2,
                              y0 + (CELL_H - img.height) // 2))
            draw.text((x0 + 8, y0 + CELL_H + 5), stem, fill=(255, 255, 120))
        out = os.path.join(SHEETS, f"sheet_{s // PER_SHEET + 1}.jpg")
        sheet.save(out, quality=88)
        print(f"saved {out} ({len(chunk)} images)")


if __name__ == "__main__":
    main()
