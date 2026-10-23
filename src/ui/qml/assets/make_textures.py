#!/usr/bin/env python3
"""Regenerate the soft ground textures used by the 3D cluster scene.

    python src/ui/qml/assets/make_textures.py

Each texture is white with an alpha channel; QML tints it with the threat
colour (PrincipledMaterial.baseColor multiplies baseColorMap).

    halo.png     soft rounded-rectangle glow drawn under a vehicle
    chevron.png  ground chevron (points toward image top = forward)
    pool.png     headlamp light pool in front of the ego vehicle
    scene_mask.png  soft-edged alpha mask over the 3D scene in Cluster.qml
                    (size must match the sceneZone item: 1060 x 520)
"""

import os

import cv2
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))


def _save(name, alpha):
    a = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    rgba = np.dstack([np.full_like(a, 255)] * 3 + [a])
    cv2.imwrite(os.path.join(OUT, name), rgba)


def rounded_rect_sdf(w, h, half_w, half_h, radius):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    px = np.abs(xs - (w - 1) / 2.0) - (half_w - radius)
    py = np.abs(ys - (h - 1) / 2.0) - (half_h - radius)
    outside = np.hypot(np.maximum(px, 0), np.maximum(py, 0))
    inside = np.minimum(np.maximum(px, py), 0)
    return outside + inside - radius


def make_halo():
    w, h = 128, 256
    d = rounded_rect_sdf(w, h, 40, 104, 30)
    core = np.clip(1.0 - np.maximum(d, 0) / 22.0, 0, 1) ** 1.6
    rim = np.exp(-((d + 2.0) / 5.0) ** 2) * 0.9
    alpha = np.maximum(core * 0.55, rim)
    alpha[d < -8] = np.maximum(alpha[d < -8], 0.35)
    _save("halo.png", alpha)


def make_chevron():
    w, h = 256, 128
    img = np.zeros((h * 4, w * 4), np.float32)
    s = 4
    pts = np.array([
        [20, 118], [128, 18], [236, 118], [196, 118], [128, 58], [60, 118]
    ], np.float32) * s
    cv2.fillPoly(img, [pts.astype(np.int32)], 1.0, lineType=cv2.LINE_AA)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    _save("chevron.png", img)


def make_pool():
    w, h = 256, 256
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xs - w / 2) / (w / 2)
    ny = (ys - h / 2) / (h / 2)
    r = np.sqrt(nx ** 2 + ny ** 2)
    alpha = np.clip(1.0 - r, 0, 1) ** 2.2
    _save("pool.png", alpha)


def make_scene_mask(w=1060, h=520):
    xs = np.linspace(0, 1, w)
    ys = np.linspace(0, 1, h)

    def ramp(u, a, b):
        t = np.clip((u - a) / (b - a), 0, 1)
        return t * t * (3 - 2 * t)

    ax = np.minimum(ramp(xs, 0.0, 0.16), ramp(1 - xs, 0.0, 0.16))
    ay = np.minimum(ramp(ys, 0.02, 0.26), ramp(1 - ys, 0.0, 0.20))
    _save("scene_mask.png", np.outer(ay, ax))


if __name__ == "__main__":
    make_scene_mask()
    make_halo()
    make_chevron()
    make_pool()
    print("textures written to", OUT)
