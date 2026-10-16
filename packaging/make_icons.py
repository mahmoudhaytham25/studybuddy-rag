"""Generate the StudyBuddy app icons (packaging/icon.png, icon.ico, icon.icns).

Design: dark rounded tile matching the app's #1e1f24 chrome with an amber
(#d4a253) chat-bubble glyph — the accent used throughout _STYLESHEET.

Requires Pillow in the build venv only (icons are committed, so Pillow is
not a project dependency):

    .venv/bin/pip install pillow
    .venv/bin/python packaging/make_icons.py

icon.icns is produced via macOS `iconutil`; on other platforms the step is
skipped (PyInstaller only needs it for the .app bundle anyway).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
SIZE = 1024

# Palette lifted from studybuddy/ui/main_window.py::_STYLESHEET
BG = "#1e1f24"        # window background
BORDER = "#3a3b44"    # hairline borders
ACCENT = "#d4a253"    # amber accent
TEXT = "#e8e6e1"      # near-white text


def render(size: int = SIZE) -> Image.Image:
    """Render the icon at `size` px (drawn at SIZE then downscaled)."""
    s = size / SIZE  # drawing scale
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # -- dark rounded tile --------------------------------------------------
    margin = 28 * s
    tile = [margin, margin, size - margin, size - margin]
    d.rounded_rectangle(tile, radius=200 * s, fill=BG)
    # subtle inner border ring
    inset = margin + 10 * s
    d.rounded_rectangle(
        [inset, inset, size - inset, size - inset],
        radius=185 * s,
        outline=BORDER,
        width=max(1, round(6 * s)),
    )

    # -- amber chat bubble ("ask your notes") --------------------------------
    bx0, by0, bx1, by1 = 0.22, 0.24, 0.78, 0.60  # fractions of size
    bubble = [bx0 * size, by0 * size, bx1 * size, by1 * size]
    d.rounded_rectangle(bubble, radius=110 * s, fill=ACCENT)
    # tail: small triangle hanging off the bottom-left of the bubble
    tail = [
        (0.33 * size, by1 * size - 4 * s),
        (0.52 * size, by1 * size - 4 * s),
        (0.28 * size, 0.78 * size),
    ]
    d.polygon(tail, fill=ACCENT)
    # rounded tip on the tail
    d.ellipse(
        [0.28 * size - 30 * s, 0.78 * size - 30 * s,
         0.28 * size + 30 * s, 0.78 * size + 30 * s],
        fill=ACCENT,
    )

    # -- "text lines" punched out of the bubble in the tile colour -----------
    line_h = 44 * s
    line_r = line_h / 2
    lx0 = 0.30 * size
    for i, frac_w in enumerate((0.40, 0.30)):
        y = (0.345 + i * 0.115) * size
        d.rounded_rectangle(
            [lx0, y, lx0 + frac_w * size, y + line_h],
            radius=line_r,
            fill=BG,
        )

    # -- small amber dot baseline accent under the bubble --------------------
    d.ellipse(
        [0.72 * size, 0.72 * size, 0.78 * size, 0.78 * size],
        fill=ACCENT,
    )

    return img


def main() -> None:
    img = render(SIZE)

    png = OUT / "icon.png"
    img.save(png, format="PNG")
    print(f"wrote {png}")

    ico = OUT / "icon.ico"
    img.save(
        ico,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
               (128, 128), (256, 256)],
    )
    print(f"wrote {ico}")

    if sys.platform == "darwin":
        iconset = OUT / "icon.iconset"
        iconset.mkdir(exist_ok=True)
        for px in (16, 32, 64, 128, 256, 512, 1024):
            render(px).save(
                iconset / f"icon_{px}x{px}.png", format="PNG"
            )
        # iconutil also wants @2x names; map 2x variants onto the same files
        aliases = {
            "icon_16x16@2x.png": "icon_32x32.png",
            "icon_32x32@2x.png": "icon_64x64.png",
            "icon_128x128@2x.png": "icon_256x256.png",
            "icon_256x256@2x.png": "icon_512x512.png",
            "icon_512x512@2x.png": "icon_1024x1024.png",
        }
        for alias, src in aliases.items():
            shutil.copy(iconset / src, iconset / alias)
        subprocess.run(
            ["iconutil", "-c", "icns", "-o", str(OUT / "icon.icns"),
             str(iconset)],
            check=True,
        )
        shutil.rmtree(iconset)
        print(f"wrote {OUT / 'icon.icns'}")
    else:
        print("skipping icon.icns (needs macOS iconutil)")


if __name__ == "__main__":
    main()
