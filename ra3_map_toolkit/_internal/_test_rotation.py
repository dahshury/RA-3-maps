"""Generate all 5 rotation/flip variants of the 2 II map and a side-by-side PNG."""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image, ImageDraw, ImageFont

from map_processor.core.ra3map import Ra3Map
from map_processor.utils.map_rotation import (
    rotate_context_right_angles,
    flip_context_axis,
)
from minimap_generator import generate_minimap

SRC = Path(r"e:/DL/Projects/Ra3 texture gen/RA 3 maps/RA3 Official maps/2 II/map_mp_2_rao1.map")
OUT = Path(r"e:/DL/Projects/Ra3 texture gen/RA 3 maps/RA3 Official maps/2 II/_rotation_test")
OUT.mkdir(parents=True, exist_ok=True)


VARIANTS = [
    ("orig",     "Original",            lambda c: None),
    ("rot90cw",  "Rotate 90 right (CW)", lambda c: rotate_context_right_angles(c, 90, clockwise=True)),
    ("rot90ccw", "Rotate 90 left (CCW)", lambda c: rotate_context_right_angles(c, 90, clockwise=False)),
    ("rot180",   "Rotate 180",          lambda c: rotate_context_right_angles(c, 180, clockwise=True)),
    ("flipx",    "Flip X (mirror y)",   lambda c: flip_context_axis(c, axis="x")),
    ("flipy",    "Flip Y (mirror x)",   lambda c: flip_context_axis(c, axis="y")),
]


def render_one(label: str, transform):
    m = Ra3Map(str(SRC))
    m.parse()
    ctx = m.get_context()
    transform(ctx)
    img = generate_minimap(ctx)
    out_map = OUT / f"map_mp_2_rao1_{label}.map"
    m.save(str(out_map), compress=True)
    if img is not None:
        img.save(OUT / f"{label}.png")
    return img


images = []
labels = []
for key, label, fn in VARIANTS:
    print(f"-- {label}")
    img = render_one(key, fn)
    if img is None:
        print(f"  ERROR: minimap None for {key}")
        continue
    images.append(img)
    labels.append(label)
    print(f"  saved {key}.png  size={img.size}")
    print(f"  saved map_mp_2_rao1_{key}.map")

# Build a side-by-side comparison sheet
PAD = 12
LABEL_H = 22
TARGET_W = 280  # column width
def fit(img: Image.Image) -> Image.Image:
    ratio = TARGET_W / img.width
    return img.resize((TARGET_W, int(img.height * ratio)), Image.NEAREST)

resized = [fit(im) for im in images]
col_h = max(im.height for im in resized)
sheet_w = PAD + (TARGET_W + PAD) * len(resized)
sheet_h = PAD + LABEL_H + col_h + PAD
sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 28))
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 14)
except Exception:
    font = ImageFont.load_default()

for i, (img, lbl) in enumerate(zip(resized, labels)):
    x = PAD + i * (TARGET_W + PAD)
    draw.text((x, PAD), lbl, fill=(220, 220, 220), font=font)
    sheet.paste(img, (x, PAD + LABEL_H))

sheet_path = OUT / "all_variants.png"
sheet.save(sheet_path)
print(f"\nside-by-side: {sheet_path}")
