"""Lane-network input prep: ImageNet norm plus optional dark-frame CLAHE."""

from __future__ import annotations

import cv2
import numpy as np

from src.inference import lane_filter_config as cfg

IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_NORM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

_clahe = None
_clahe_key = None


def _get_clahe():
    global _clahe, _clahe_key
    tile = int(getattr(cfg, "CLAHE_TILE", 8))
    clip = float(getattr(cfg, "CLAHE_CLIP", 2.5))
    key = (tile, clip)
    if _clahe is None or _clahe_key != key:
        _clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        _clahe_key = key
    return _clahe


def road_luma(bgr: np.ndarray) -> float:
    """Mean HSV-V on the lower 2/3 of the frame (skip bright sky)."""
    h = int(bgr.shape[0])
    road = bgr[h // 3 :, :, :]
    v = cv2.cvtColor(road, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float(np.mean(v))


def apply_clahe_bgr(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on Lab L. Display overlays should keep the unenhanced copy."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = _get_clahe().apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def bgr_to_nchw(bgr: np.ndarray) -> np.ndarray:
    img = bgr[:, :, ::-1].astype(np.float32)
    img = (img - IMG_NORM_MEAN) / IMG_NORM_STD
    img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return np.ascontiguousarray(img)


def prepare_lane_input(resized_bgr: np.ndarray, force_mode: str | None = None):
    """Build NCHW tensor + mask from a model-sized BGR image.

    force_mode: None (config), "off", "conf", "clahe", "gated", or "always".
    """
    luma = road_luma(resized_bgr)
    dark = luma < float(getattr(cfg, "DARK_LUMA_MAX", 125.0))

    if force_mode is not None:
        mode = force_mode
    elif bool(getattr(cfg, "ENABLE_CLAHE_ALWAYS", False)):
        mode = "always"
    elif bool(getattr(cfg, "ENABLE_DARK_CLAHE", False)) and bool(
        getattr(cfg, "ENABLE_ADAPTIVE_CONF", False)
    ):
        mode = "gated"
    elif bool(getattr(cfg, "ENABLE_DARK_CLAHE", False)):
        mode = "clahe"
    elif bool(getattr(cfg, "ENABLE_ADAPTIVE_CONF", False)):
        mode = "conf"
    else:
        mode = "off"

    use_clahe = (mode == "always") or (mode in ("gated", "clahe") and dark)
    use_dark_conf = dark and mode in ("gated", "always", "conf")
    model_bgr = apply_clahe_bgr(resized_bgr) if use_clahe else resized_bgr
    conf = (
        float(getattr(cfg, "DARK_CONF_THRESHOLD", 0.28))
        if use_dark_conf
        else float(cfg.CONF_THRESHOLD)
    )

    h, w = resized_bgr.shape[:2]
    img = bgr_to_nchw(model_bgr)
    mask = np.ascontiguousarray(np.zeros((1, 1, h, w), dtype=np.float32))
    meta = {
        "luma": luma,
        "dark": dark,
        "clahe": use_clahe,
        "conf": conf,
        "mode": mode,
    }
    return img, mask, meta
