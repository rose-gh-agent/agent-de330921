#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


INPUT_PATH = Path("/home/user/attachments/rs.jpeg")
OUTPUT_PATH = Path("/home/user/output/rs_yellow.jpeg")


def main() -> None:
    image = cv2.imread(str(INPUT_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load input image: {INPUT_PATH}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Restrict edits to brighter, saturated teal/cyan metal tones.
    color_mask = (s > 45) & (v > 30)
    teal_mask = (h >= 72) & (h <= 125)
    mask = color_mask & teal_mask

    # Remap the teal band into a gold band while preserving value/luminosity.
    # 72..125 in OpenCV hue space becomes roughly 18..34 (yellow/gold).
    if np.any(mask):
        teal_min = 72.0
        teal_max = 125.0
        gold_min = 18.0
        gold_max = 34.0

        normalized = np.clip((h[mask] - teal_min) / (teal_max - teal_min), 0.0, 1.0)
        h[mask] = gold_min + normalized * (gold_max - gold_min)

        # Slight saturation lift improves the metallic gold feel without
        # disturbing the underlying shading or contrast.
        s[mask] = np.clip(s[mask] * 1.12 + 8.0, 0.0, 255.0)

    hsv[:, :, 0] = h
    hsv[:, :, 1] = s
    hsv[:, :, 2] = v

    transformed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    result = image.copy()
    result[mask] = transformed[mask]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(OUTPUT_PATH), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Could not write output image: {OUTPUT_PATH}")

    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
