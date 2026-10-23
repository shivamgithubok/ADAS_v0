"""Garmin HUD ego-speed: OCR to JSON, then load by frame index.

The lane net runs on *_nohud clips. Speed is read from the matching
with-HUD file (bottom bar, ``NN MPH``) and stored as a per-frame JSON log.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import cv2
import numpy as np

MPH_TO_MPS = 0.44704
_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
_MPH_TEMPLATE_PATH = os.path.join(_ASSET_DIR, "garmin_hud_mph.png")


def detect_hud_top(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return _hud_top(gray)


def crop_garmin_hud(frame: np.ndarray, hood_frac: float = 0.08, hud_top: Optional[int] = None):
    """Remove the Garmin bottom bar and optional hood. Returns (crop, hud_y)."""
    y0 = int(hud_top) if hud_top is not None else detect_hud_top(frame)
    y0 = max(1, min(int(y0), frame.shape[0] - 1))
    content = frame[:y0]
    if hood_frac > 0:
        cut = int(content.shape[0] * (1.0 - float(hood_frac)))
        content = content[: max(cut, 1)]
    return content, y0


def export_nohud_video(src: str, dst: str, hood_frac: float = 0.08):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(src)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    ok, frame0 = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Cannot read {src}")
    crop0, hud_y = crop_garmin_hud(frame0, hood_frac=hood_frac)
    ch, cw = crop0.shape[:2]
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))
    writer.write(crop0)
    n = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop, _ = crop_garmin_hud(frame, hood_frac=hood_frac, hud_top=hud_y)
        if crop.shape[0] != ch or crop.shape[1] != cw:
            crop = cv2.resize(crop, (cw, ch))
        writer.write(crop)
        n += 1
    cap.release()
    writer.release()
    return n, (cw, ch), hud_y


def _hud_top(gray: np.ndarray) -> int:
    h = gray.shape[0]
    rows = []
    for y in range(h - 1, -1, -1):
        if (gray[y] < 40).mean() > 0.55:
            rows.append(y)
        elif rows:
            break
    if len(rows) < 12:
        return int(h * 0.93)
    return int(min(rows))


_MPH_TMPL_H = None


def _mph_template_height() -> int:
    """Native HUD row count. Scaling the MPH glyph to a taller hood-crop kills NCC."""
    global _MPH_TMPL_H
    if _MPH_TMPL_H is None:
        img = cv2.imread(_MPH_TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE) if os.path.isfile(_MPH_TEMPLATE_PATH) else None
        _MPH_TMPL_H = int(img.shape[0]) if img is not None and img.size else 37
    return int(_MPH_TMPL_H)


def _hud_bar(gray: np.ndarray) -> np.ndarray:
    """Bottom Garmin overlay only — do not eat the dark hood as HUD."""
    h = gray.shape[0]
    y0 = _hud_top(gray)
    y0 = max(int(y0), h - 42)
    bar = gray[y0:]
    th = _mph_template_height()
    if bar.shape[0] > th:
        bar = bar[-th:]
    return bar


def _load_digit_templates():
    templates = {}
    if not os.path.isdir(_ASSET_DIR):
        return templates
    for name in os.listdir(_ASSET_DIR):
        if not name.startswith("garmin_digit_") or not name.endswith(".png"):
            continue
        try:
            digit = int(name.replace("garmin_digit_", "").replace(".png", ""))
        except ValueError:
            continue
        img = cv2.imread(os.path.join(_ASSET_DIR, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        templates[digit] = (img > 127).astype(np.uint8) * 255
    return templates


_DIGIT_TEMPLATES = None


def _digit_templates():
    global _DIGIT_TEMPLATES
    if _DIGIT_TEMPLATES is None:
        _DIGIT_TEMPLATES = _load_digit_templates()
    return _DIGIT_TEMPLATES


def _ncc_digit(norm: np.ndarray) -> Optional[int]:
    best_d, best_s = None, 0.50
    n = (norm > 0).astype(np.float32)
    n_std = float(n.std()) + 1e-6
    for digit, tmpl in _digit_templates().items():
        t = cv2.resize(tmpl, (12, 18), interpolation=cv2.INTER_NEAREST)
        t = (t > 0).astype(np.float32)
        t_std = float(t.std()) + 1e-6
        score = float(((n - n.mean()) * (t - t.mean())).mean() / (n_std * t_std))
        if score > best_s:
            best_s, best_d = score, digit
    return best_d


def _classify_digit(bin_img: np.ndarray) -> Optional[int]:
    """Classify a binary digit crop (ink=255) from Garmin HUD font."""
    rs = np.where(bin_img.sum(axis=1) > 0)[0]
    cs = np.where(bin_img.sum(axis=0) > 0)[0]
    if len(rs) < 4 or len(cs) < 2:
        return None
    d = bin_img[rs[0] : rs[-1] + 1, cs[0] : cs[-1] + 1]
    h, w = d.shape
    if h < 6 or w < 2:
        return None
    aspect = w / float(h)
    if aspect < 0.22:
        return 1
    norm = cv2.resize((d > 0).astype(np.uint8) * 255, (12, 18), interpolation=cv2.INTER_NEAREST)
    ncc = _ncc_digit(norm)
    if ncc is not None:
        return ncc
    ink = (norm > 0).astype(np.float32)
    holes = _count_holes(norm)
    top = float(ink[:6].mean())
    mid = float(ink[6:12].mean())
    bot = float(ink[12:].mean())
    left = float(ink[:, :4].mean())
    right = float(ink[:, 8:].mean())
    cy = float(np.average(np.arange(18), weights=ink.sum(axis=1) + 1e-6)) / 17.0

    if holes >= 2:
        return 8
    if holes == 1:
        # Garmin 4: hole + heavy right stem, open left
        if left < 0.38 and right > 0.45:
            return 4
        if bot < 0.28 and top > 0.35:
            return 4
        if cy < 0.48:
            return 9
        if cy > 0.58:
            return 6
        if left > 0.45 and right > 0.45:
            return 0
        return 9 if top >= bot else 6
    if aspect < 0.38 and left > 0.55:
        return 1
    if bot < 0.22 and top > 0.30:
        return 7
    if mid > 0.38 and top > 0.35 and bot > 0.35:
        return 5
    if right > left + 0.08 and bot > 0.30:
        return 3
    if left > right and bot > 0.28:
        return 2
    if bot > top and mid < 0.35:
        return 2
    return 3 if right >= left else 2


def _count_holes(bin_img: np.ndarray) -> int:
    ink = (bin_img > 0).astype(np.uint8)
    bg = (1 - ink).astype(np.uint8)
    if bg.size == 0:
        return 0
    h, w = bg.shape
    ff = bg.copy()
    for x in range(w):
        if ff[0, x]:
            cv2.floodFill(ff, None, (x, 0), 2)
        if ff[h - 1, x]:
            cv2.floodFill(ff, None, (x, h - 1), 2)
    for y in range(h):
        if ff[y, 0]:
            cv2.floodFill(ff, None, (0, y), 2)
        if ff[y, w - 1]:
            cv2.floodFill(ff, None, (w - 1, y), 2)
    trapped = ((bg == 1) & (ff == 1)).astype(np.uint8)
    n, _ = cv2.connectedComponents(trapped)
    return max(0, int(n) - 1)


def _split_digit_blobs(bin_img: np.ndarray):
    col = (bin_img > 0).sum(axis=0)
    ink = col > 0
    runs = []
    on = None
    for i, v in enumerate(ink):
        if v and on is None:
            on = i
        elif (not v) and on is not None:
            if i - on >= 2:
                runs.append((on, i))
            on = None
    if on is not None and len(ink) - on >= 2:
        runs.append((on, len(ink)))
    return runs


def read_hud_speed_mph(frame: np.ndarray, mph_template: Optional[np.ndarray] = None) -> Optional[int]:
    """Read integer MPH from a Garmin HUD frame. Returns None on failure."""
    if frame is None or frame.size == 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    bar = _hud_bar(gray)
    if bar.shape[0] < 10:
        return None
    tmpl = mph_template
    if tmpl is None and os.path.isfile(_MPH_TEMPLATE_PATH):
        tmpl = cv2.imread(_MPH_TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if tmpl is None or tmpl.size == 0 or bar.shape[1] < 8:
        return None
    if bar.shape[0] != tmpl.shape[0]:
        scale = bar.shape[0] / float(tmpl.shape[0])
        tmpl = cv2.resize(tmpl, (max(8, int(round(tmpl.shape[1] * scale))), bar.shape[0]))
    res = cv2.matchTemplate(bar, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.45:
        return None
    mph_x = int(max_loc[0])
    x1 = max(0, mph_x - 4)
    x0 = max(0, mph_x - 46)
    roi = bar[:, x0:x1]
    _, bw = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)
    runs = _split_digit_blobs(bw)
    if not runs:
        return None
    keep = [(a, b) for a, b in runs if b > bw.shape[1] - 28]
    if not keep:
        keep = runs[-2:]
    blobs = [bw[:, a:b] for a, b in keep if (bw[:, a:b]).sum() >= 20]
    if not blobs:
        return None
    blobs = blobs[-2:]
    digits = []
    for blob in blobs:
        d = _classify_digit(blob)
        if d is None:
            return None
        digits.append(d)
    if not digits:
        return None
    value = 0
    for d in digits:
        value = value * 10 + d
    if value < 0 or value > 120:
        return None
    return int(value)


def extract_speed_log(video_path: str, max_frames: int = 0) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tmpl = cv2.imread(_MPH_TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE) if os.path.isfile(_MPH_TEMPLATE_PATH) else None
    mph = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mph.append(read_hud_speed_mph(frame, tmpl))
        i += 1
        if max_frames and i >= max_frames:
            break
    cap.release()
    mph = _fill_speed_gaps(mph)
    mps = [None if v is None else round(float(v) * MPH_TO_MPS, 4) for v in mph]
    valid_n = sum(v is not None for v in mph)
    return {
        "source_video": os.path.abspath(video_path),
        "fps": fps,
        "frame_count": len(mph),
        "unit": "mph",
        "valid_frames": valid_n,
        "mph": mph,
        "mps": mps,
        "notes": "Garmin bottom HUD 'NN MPH'. Pair by frame index with the matching *_nohud.mp4.",
    }


def _fill_speed_gaps(mph, max_jump=6):
    """Zero-order hold; drop OCR spikes larger than max_jump mph."""
    out = list(mph)
    last = None
    for i, v in enumerate(out):
        if v is None:
            out[i] = last
            continue
        if last is not None and abs(int(v) - int(last)) > max_jump:
            out[i] = last
            continue
        last = int(v)
        out[i] = last
    return out


def save_speed_log(log: dict, json_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(json_path)) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def default_speed_json_path(video_path: str) -> str:
    """If video is *_nohud.mp4, JSON lives next to the HUD sibling stem."""
    path = os.path.abspath(video_path)
    stem, _ = os.path.splitext(path)
    stem = re.sub(r"_nohud$", "", stem, flags=re.IGNORECASE)
    return stem + "_speed.json"


def find_speed_json(video_path: str) -> Optional[str]:
    candidates = [
        default_speed_json_path(video_path),
        os.path.splitext(os.path.abspath(video_path))[0] + "_speed.json",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


class EgoSpeedLog:
    """Frame-index lookup of ego speed in m/s."""

    def __init__(self, mph=None, mps=None, fps=30.0):
        self.fps = float(fps)
        if mps is not None:
            self.mps = [None if v is None else float(v) for v in mps]
        elif mph is not None:
            self.mps = [None if v is None else float(v) * MPH_TO_MPS for v in mph]
        else:
            self.mps = []

    @classmethod
    def from_json(cls, json_path: str) -> "EgoSpeedLog":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(mph=data.get("mph"), mps=data.get("mps"), fps=data.get("fps", 30.0))

    @classmethod
    def auto_load(cls, video_path: str) -> Optional["EgoSpeedLog"]:
        path = find_speed_json(video_path)
        if path is None:
            return None
        return cls.from_json(path)

    def get_mph(self, frame_index: int) -> Optional[int]:
        if frame_index < 0 or frame_index >= len(self.mps):
            return None
        v = self.mps[frame_index]
        if v is None:
            return None
        return int(round(float(v) / MPH_TO_MPS))

    def get_mps(self, frame_index: int, min_mps: float = 0.3) -> Optional[float]:
        """Metres per second at this frame.

        `min_mps=0` keeps a true stop as 0.0 (needed for world-frame heading).
        The default 0.3 still treats a parked log as 'no speed' for dash scroll.
        """
        if frame_index < 0 or frame_index >= len(self.mps):
            return None
        v = self.mps[frame_index]
        if v is None or v < float(min_mps):
            return None
        return float(v)
