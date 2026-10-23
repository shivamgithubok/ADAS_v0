"""
BEV car sprite preparation.

Pipeline:
  1. Load asset (JPEG / PNG / WebP / AVIF)
  2. Rotate nose-up using known map (or mirror heuristic)
  3. Cut background (alpha / solid key / GrabCut for checker plates)
  4. Strip white/gray fringe that causes halos on the dark BEV road
  5. Keep only true top-down cars (portrait aspect) — skip rear/side studio shots
  6. Cache cleaned PNGs under data/cache/bev_cars/
"""

from __future__ import annotations

import hashlib
import os

import cv2
import numpy as np
from PIL import Image
from PySide6.QtGui import QImage, QPixmap

# Nose-up clockwise rotations for current assets in data/
KNOWN_ROTATE_CW = {
    'tooooo.avif': 90,  # nose left
    '1.jpeg': 0,        # rear 3/4, already nose-away / up
    '2.jpeg': 0,
    '3.jpeg': 90,       # nose-up after dark-roof crop pipeline
    '4.jpeg': 90,       # nose left
    '5': 0,             # nose up
}

# Skip trucks only — rear/3-4 studio cars (1.jpeg etc.) are allowed via ready/
SKIP_FILES = {
    'truck_1.jpeg', 'truck.jpeg',
}


def _bgra_to_pixmap(bgra: np.ndarray) -> QPixmap | None:
    bgra = np.ascontiguousarray(bgra)
    ch, cw = bgra.shape[:2]
    qimg = QImage(bgra.data, cw, ch, cw * 4, QImage.Format_ARGB32).copy()
    pix = QPixmap.fromImage(qimg)
    return None if pix.isNull() else pix


def _load_bgr_alpha(path: str):
    ext = os.path.splitext(path)[1].lower()
    try_pil = ext in ('.avif', '.heic', '.heif', '') or True
    raw = None if ext in ('.avif', '.heic', '.heif') else cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None and try_pil:
        try:
            rgba = np.array(Image.open(path).convert('RGBA'))
            bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            return bgr, rgba[:, :, 3]
        except Exception:
            return None, None
    if raw is None:
        return None, None
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), None
    if raw.shape[2] == 4:
        return raw[:, :, :3], raw[:, :, 3]
    return raw, None


def _looks_like_checkerboard(bgr: np.ndarray) -> bool:
    """Baked checkerboard plates have high local contrast in the corners."""
    h, w = bgr.shape[:2]
    patches = [
        bgr[: min(48, h), : min(48, w)],
        bgr[: min(48, h), max(0, w - 48):],
        bgr[max(0, h - 48):, : min(48, w)],
        bgr[max(0, h - 48):, max(0, w - 48):],
    ]
    stds = [float(p.std()) for p in patches if p.size]
    means = [float(p.mean()) for p in patches if p.size]
    if not stds:
        return False
    return (np.median(stds) > 18.0) and (120 < np.median(means) < 240)


def _flood_outer(is_bg: np.ndarray) -> np.ndarray:
    h, w = is_bg.shape
    work = (is_bg.astype(np.uint8) * 255)
    ff = np.zeros((h + 2, w + 2), np.uint8)
    sx_step = max(1, w // 24)
    sy_step = max(1, h // 24)
    for x in range(0, w, sx_step):
        for y in (0, h - 1):
            if work[y, x] == 255:
                cv2.floodFill(work, ff, (x, y), 128)
    for y in range(0, h, sy_step):
        for x in (0, w - 1):
            if work[y, x] == 255:
                cv2.floodFill(work, ff, (x, y), 128)
    outer = work == 128
    return outer if outer.any() else is_bg


def _grabcut(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    m = 0.06
    rect = (int(w * m), int(h * m), int(w * (1 - 2 * m)), int(h * (1 - 2 * m)))
    cv2.grabCut(bgr, mask, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
    content = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(content.astype(np.uint8), 8)
    if num > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        content = labels == keep
    return content


def _strip_fringe(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Remove mid-gray checker/anti-alias halo — keep bright white body paint."""
    a = alpha.copy()
    near = cv2.dilate((a == 0).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lum = bgr.mean(axis=2)
    # Mid-gray halo only (not solid white paint >235)
    halo = near & (hsv[:, :, 1] < 45) & (lum > 155) & (lum < 230) & (a > 0)
    a[halo] = 0
    a = cv2.erode(a, np.ones((3, 3), np.uint8), iterations=2)
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return a


def _content_mask(bgr: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is not None and float((alpha == 0).mean()) > 0.05:
        return alpha > 10

    if _looks_like_checkerboard(bgr):
        return _grabcut(bgr)

    border = np.concatenate([
        bgr[0, :, :], bgr[-1, :, :], bgr[:, 0, :], bgr[:, -1, :]
    ]).astype(np.float32)
    ref = np.median(border, axis=0)
    dist = np.linalg.norm(bgr.astype(np.float32) - ref, axis=2)
    white = (bgr[:, :, 0] > 230) & (bgr[:, :, 1] > 230) & (bgr[:, :, 2] > 230)
    black = (bgr[:, :, 0] < 45) & (bgr[:, :, 1] < 45) & (bgr[:, :, 2] < 45)
    mean_c = float(ref.mean())
    if mean_c > 200:
        is_bg = white | (dist < 28)
    elif mean_c < 60:
        is_bg = black
    else:
        is_bg = (dist < 32) | white

    content = ~_flood_outer(is_bg)
    ys, xs = np.where(content)
    h, w = bgr.shape[:2]
    if len(xs) == 0 or content.mean() > 0.80:
        return _grabcut(bgr)
    cw, ch = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    if cw * ch > 0.90 * h * w:
        return _grabcut(bgr)
    return content


def prepare_car_bgra(path: str, max_side: int = 512) -> np.ndarray | None:
    """Return cropped BGRA uint8 array (nose-up, fringe-stripped) or None."""
    name = os.path.basename(path)
    if name in SKIP_FILES:
        return None

    bgr, alpha = _load_bgr_alpha(path)
    if bgr is None:
        return None

    h, w = bgr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    rot = KNOWN_ROTATE_CW.get(name, 0)
    if rot:
        code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }.get(int(rot) % 360)
        if code is not None:
            bgr = cv2.rotate(bgr, code)
            if alpha is not None:
                alpha = cv2.rotate(alpha, code)

    # Checkerboard JPEGs — always GrabCut; solid white studio plates use keying
    if name in ('3.jpeg', '4.jpeg') or _looks_like_checkerboard(bgr):
        content = _grabcut(bgr)
    else:
        content = _content_mask(bgr, alpha)

    alpha_u8 = np.where(content, np.uint8(255), np.uint8(0))
    alpha_u8 = _strip_fringe(bgr, alpha_u8)

    ys, xs = np.where(alpha_u8 > 0)
    if len(xs) == 0:
        return None

    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha_u8
    bgra = bgra[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]

    ch, cw = bgra.shape[:2]
    aspect = cw / float(ch)
    # Portrait top-downs + near-square rear 3/4 (1.jpeg)
    if not (0.30 <= aspect <= 1.05):
        return None
    return np.ascontiguousarray(bgra)


def _cache_path(src_path: str, cache_dir: str) -> str:
    st = os.stat(src_path)
    key = f"{os.path.basename(src_path)}:{st.st_mtime_ns}:{st.st_size}"
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    base = os.path.splitext(os.path.basename(src_path))[0]
    return os.path.join(cache_dir, f"{base}_{digest}.png")


def load_car_pixmap(path: str, cache_dir: str | None = None) -> QPixmap | None:
    """Prepare (or load cached) top-down car pixmap."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cpath = _cache_path(path, cache_dir)
        if os.path.exists(cpath):
            pix = QPixmap(cpath)
            if not pix.isNull():
                return pix

    bgra = prepare_car_bgra(path)
    if bgra is None:
        return None
    pix = _bgra_to_pixmap(bgra)
    if pix is None:
        return None
    if cache_dir:
        cv2.imwrite(_cache_path(path, cache_dir), bgra)
    return pix


def list_image_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        fp = os.path.join(folder, name)
        if not os.path.isfile(fp):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in ('.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp') or ext == '':
            out.append(fp)
    return out


def load_ready_png(path: str) -> QPixmap | None:
    """Load a pre-baked RGBA PNG, crush soft edge halo, trim to opaque bounds."""
    bgr, alpha = _load_bgr_alpha(path)
    if bgr is None or alpha is None:
        return None

    # Remove soft gray anti-alias halo (keeps dark glass + bright body paint)
    m = (alpha > 128).astype(np.uint8) * 255
    lum = bgr.mean(axis=2)
    for _ in range(6):
        near = cv2.dilate((m == 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        edge = near & (m > 0)
        bad = edge & (lum > 50) & (lum < 185)
        if not np.any(bad):
            break
        m[bad] = 0

    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = m
    bgra = bgra[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]
    ch, cw = bgra.shape[:2]
    # Allow near-square rear 3/4 assets (e.g. 1-back.png) as well as portrait top-downs
    if not (0.28 <= cw / float(ch) <= 1.15):
        return None
    return _bgra_to_pixmap(np.ascontiguousarray(bgra))


# Ego-only plates — not used in the traffic sprite pool
EGO_ONLY_NAMES = {
    '1-back.png', '1_back.png', 'ego.png', 'ego_car.png',
}


def load_car_sprite_pool(
    sides_dir: str,
    cache_dir: str,
    extra_paths: list[str] | None = None,
    exclude_names: set[str] | None = None,
) -> tuple[list[QPixmap], list[str]]:
    """
    Load usable top-down / elevated car sprites (no trucks).

    Prefer pre-baked transparent PNGs in sides_dir/ready/ when present —
    those are the cleanest mattes. Fall back to on-the-fly prep of source files.

    Returns (pixmaps, basenames) so callers can prefer a white near-car skin.
    """
    sprites: list[QPixmap] = []
    names: list[str] = []
    seen: set[str] = set()
    skip = set(n.lower() for n in (exclude_names or set())) | set(n.lower() for n in EGO_ONLY_NAMES)

    ready_dir = os.path.join(sides_dir, 'ready')
    ready_paths = list_image_files(ready_dir) if os.path.isdir(ready_dir) else []
    ready_paths = [p for p in ready_paths if p.lower().endswith('.png')]

    if ready_paths:
        for p in ready_paths:
            name = os.path.basename(p)
            if name in seen or name.lower() in skip:
                continue
            seen.add(name)
            pix = load_ready_png(p)
            if pix is not None:
                sprites.append(pix)
                names.append(name)
        if sprites:
            return sprites, names

    paths = list_image_files(sides_dir)
    if extra_paths:
        paths.extend(p for p in extra_paths if os.path.isfile(p))
    for p in paths:
        name = os.path.basename(p)
        if name in seen or name in SKIP_FILES or name.lower() in skip:
            continue
        seen.add(name)
        pix = load_car_pixmap(p, cache_dir=cache_dir)
        if pix is not None:
            sprites.append(pix)
            names.append(name)
    return sprites, names


def find_white_sprite_index(names: list[str]) -> int | None:
    """Prefer true white top-down, then white hatch, else any name containing 'white'."""
    prefer = ('car_white.png', 'car_white_hatch.png')
    lower = [n.lower() for n in names]
    for want in prefer:
        if want in lower:
            return lower.index(want)
    for i, n in enumerate(lower):
        if 'white' in n:
            return i
    return 0 if names else None


def hw_from_pixmap(pixmap: QPixmap | None, half_length_m: float) -> float:
    if pixmap is None or pixmap.height() <= 0:
        return half_length_m * 0.48
    aspect = pixmap.width() / float(pixmap.height())
    return half_length_m * float(np.clip(aspect, 0.38, 0.65))
