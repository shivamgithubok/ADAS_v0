"""Camera-agnostic self-calibration from lane markings.

Used at runtime when no models/calib/<stem>.json exists, so ADAS2 / a live
webcam / a new dashcam do not silently inherit Garmin GRMN6694 numbers.

Only steps 1–2 run here (vanishing point + height from lane width). Those fix
*lateral* placement with no focal length. Forward range still uses a typical
dashcam HFOV prior unless a full `scripts/calibrate_from_video.py --write`
has produced a JSON with a measured f.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.ground_calib import GroundCalibration

# Typical windshield dashcam. Wrong by ~10% on f → 10% range error, zero
# lateral error. Better than copying another camera's pixel horizon.
DEFAULT_HFOV_DEG = 70.0
DEFAULT_HEIGHT_M = 1.35
DEFAULT_LANE_WIDTH_M = 3.7
# Downward pitch so the horizon sits below frame centre on a road scene.
DEFAULT_PITCH_DEG = 4.0


def generic_calib(
    width: int,
    height: int,
    hfov_deg: float = DEFAULT_HFOV_DEG,
    cam_height_m: float = DEFAULT_HEIGHT_M,
    pitch_deg: float = DEFAULT_PITCH_DEG,
) -> GroundCalibration:
    """Square-pixel prior from frame size alone. No Garmin numbers."""
    width, height = int(width), int(height)
    f_px = (0.5 * width) / math_tan_half(hfov_deg)
    u_vp = 0.5 * width
    v_vp = 0.5 * height + f_px * math_tan_half(2.0 * pitch_deg)
    v_vp = float(np.clip(v_vp, 0.35 * height, 0.85 * height))
    return GroundCalibration(
        source_width=width,
        source_height=height,
        f_px=float(f_px),
        u_vp=float(u_vp),
        v_vp=float(v_vp),
        cam_height_m=float(cam_height_m),
        source=f"generic:{width}x{height}:hfov{hfov_deg:.0f}",
    )


def math_tan_half(deg: float) -> float:
    return float(np.tan(np.radians(float(deg)) * 0.5))


def estimate_vanishing_point_on_frame(frame: np.ndarray) -> Optional[Tuple[float, float]]:
    """One-frame Hough VP. Returns None if lane lines are too weak."""
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    r0, r1 = int(0.45 * h), int(0.92 * h)
    mask = np.zeros_like(g)
    mask[r0:r1, :] = 255
    edges = cv2.bitwise_and(cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 50, 150), mask)
    min_len = max(24, int(0.08 * w))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=min_len, maxLineGap=12)
    if lines is None:
        return None
    segs = []
    for line in lines:
        x1, y1, x2, y2 = [int(v) for v in np.asarray(line).reshape(-1)[:4]]
        if abs(y2 - y1) < 8:
            continue
        s = (x2 - x1) / float(y2 - y1)
        if abs(s) > 3.5:
            continue
        segs.append((s, x1 - s * y1))
    pts = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            s1, b1 = segs[i]
            s2, b2 = segs[j]
            if abs(s1 - s2) < 0.25:
                continue
            v = (b2 - b1) / (s1 - s2)
            u = s1 * v + b1
            if 0.25 * h < v < 0.88 * h and 0.15 * w < u < 0.85 * w:
                pts.append((u, v))
    if len(pts) < 4:
        return None
    arr = np.array(pts)
    return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))


def estimate_vanishing_point_from_video(
    video_path: str,
    n_frames: int = 16,
    stride: int = 12,
    start: int = 30,
) -> Optional[Tuple[float, float, int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    us, vs = [], []
    for i in range(n_frames):
        fi = start + i * stride
        if n_total > 0 and fi >= n_total:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        hit = estimate_vanishing_point_on_frame(frame)
        if hit is None:
            continue
        us.append(hit[0])
        vs.append(hit[1])
    cap.release()
    if len(us) < 3:
        return None
    return float(np.median(us)), float(np.median(vs)), len(us)


def bootstrap_calib(
    width: int,
    height: int,
    video_path: Optional[str] = None,
    frame: Optional[np.ndarray] = None,
) -> GroundCalibration:
    """Build a calib for *this* camera.

    JSON is the caller's job (`GroundCalibration.for_video`). This function
    never copies another camera's pixel horizon.
    """
    calib = generic_calib(width, height)
    u_vp, v_vp = calib.u_vp, calib.v_vp
    n = 0
    if frame is not None:
        hit = estimate_vanishing_point_on_frame(frame)
        if hit is not None:
            u_vp, v_vp = hit
            n = 1
    if n == 0 and video_path:
        hit = estimate_vanishing_point_from_video(video_path)
        if hit is not None:
            u_vp, v_vp, n = hit
    if n > 0:
        calib = GroundCalibration(
            source_width=width,
            source_height=height,
            f_px=calib.f_px,
            u_vp=float(u_vp),
            v_vp=float(v_vp),
            cam_height_m=calib.cam_height_m,
            source=f"self-vp:{os_stem(video_path)}:{width}x{height}:n{n}",
        )
    return calib


def os_stem(path: Optional[str]) -> str:
    if not path:
        return "live"
    import os
    return os.path.splitext(os.path.basename(path))[0]
