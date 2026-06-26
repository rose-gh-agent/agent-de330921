from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image


INPUT_PATH = Path("/home/user/attachments/rs.jpeg")
OUTPUT_PATH = Path("/home/user/output/rs_blue.png")


def main() -> None:
    image = Image.open(INPUT_PATH).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)

    rgb = arr.astype(np.float32) / 255.0
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    delta = maxc - minc

    value = maxc
    saturation = np.zeros_like(maxc)
    nonzero = maxc > 1e-6
    saturation[nonzero] = delta[nonzero] / maxc[nonzero]

    # Preserve the black background and only retarget the bright cyan/teal metal.
    bright_enough = value > 0.10
    colored_enough = saturation > 0.18
    cyan_blue_dominant = (b >= r + 0.04) & (g >= r + 0.02)
    target_mask = bright_enough & colored_enough & cyan_blue_dominant

    # Fixed blue hue with a small saturation lift keeps the metallic sheen
    # while preserving the original luminance structure.
    target_hue = 230.0 / 360.0
    new_s = np.clip(saturation * 1.08, 0.0, 1.0)

    out = rgb.copy()
    ys, xs = np.where(target_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        rr, gg, bb = colorsys.hsv_to_rgb(target_hue, float(new_s[y, x]), float(value[y, x]))
        out[y, x, 0] = rr
        out[y, x, 1] = gg
        out[y, x, 2] = bb

    result = Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT_PATH)


if __name__ == "__main__":
    main()
