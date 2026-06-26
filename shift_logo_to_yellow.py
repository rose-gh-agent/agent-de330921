#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


INPUT_PATH = Path("/home/user/attachments/rs.jpeg")
OUTPUT_PATH = Path("/home/user/output/rs_yellow.jpeg")
TARGET_HUE = 60.0 / 360.0


def main() -> None:
    image = Image.open(INPUT_PATH).convert("RGB")
    data = np.array(image, dtype=np.float32) / 255.0

    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    v = maxc
    s = np.zeros_like(maxc)
    nonzero = maxc != 0
    s[nonzero] = delta[nonzero] / maxc[nonzero]

    h = np.zeros_like(maxc)
    mask = delta != 0

    rm = mask & (maxc == r)
    h[rm] = ((g[rm] - b[rm]) / delta[rm]) % 6

    gm = mask & (maxc == g)
    h[gm] = (b[gm] - r[gm]) / delta[gm] + 2

    bm = mask & (maxc == b)
    h[bm] = (r[bm] - g[bm]) / delta[bm] + 4

    h = h / 6.0

    non_bg = (s > 0.05) & (v > 0.05)
    h[non_bg] = TARGET_HUE

    h6 = h * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    result = np.zeros_like(data)
    for idx, (ri, gi, bi) in enumerate(((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))):
        mask_i = i == idx
        result[:, :, 0][mask_i] = ri[mask_i]
        result[:, :, 1][mask_i] = gi[mask_i]
        result[:, :, 2][mask_i] = bi[mask_i]

    result[~non_bg] = data[~non_bg]

    result_img = Image.fromarray((result * 255).astype(np.uint8), "RGB")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_img.save(OUTPUT_PATH, quality=95)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
