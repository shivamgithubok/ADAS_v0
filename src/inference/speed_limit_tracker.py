"""Temporal speed-limit state. One box never sets the posted limit."""

from __future__ import annotations

import re

_SPEED_RE = re.compile(r"speedLimit(\d+)", re.IGNORECASE)

# A two-digit plate under this height is a guess. Distant 65/45 flips on
# GRMN6695 were 24–32 px; the readable 50 was ~40 px and up.
SPEED_MIN_H_PX = 36
# Boxes cut by the frame edge lose a digit and the class head jumps (50 → 30).
SPEED_EDGE_PX = 3
# Two speed classes on one physical sign: the frame is not evidence.
SPEED_AMBIG_IOU = 0.50


def class_to_mph(name: str):
    if not name:
        return None
    m = _SPEED_RE.search(str(name))
    return int(m.group(1)) if m else None


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(area_a + area_b - inter)


def _readable_box(det, frame_shape):
    x1, y1, x2, y2 = det["bbox"]
    if (y2 - y1) < SPEED_MIN_H_PX or (x2 - x1) < 8:
        return False
    if frame_shape is None:
        return True
    h, w = int(frame_shape[0]), int(frame_shape[1])
    edge = SPEED_EDGE_PX
    if x1 < edge or y1 < edge or x2 > w - edge or y2 > h - edge:
        return False
    return True


def pick_speed_det(detections, min_conf=0.35, use_class=True, frame_shape=None):
    """Prefer larger, more confident speed-limit boxes; tie-break right of center.

    `det['mph']` is the LPRNet read when OCR succeeds. Otherwise the YOLO
    class digits are used (speedLimit35 and the rest).

    Returns None when nothing readable is present, and ``"ambiguous"`` when
    two different limits overlap on the same sign. An ambiguous frame must
    not move the posted limit.
    """
    cands = []
    for det in detections:
        mph = det.get("mph")
        if mph is None and use_class:
            mph = class_to_mph(det.get("class", "") or det.get("yolo_class", ""))
        if mph is None or det.get("conf", 0.0) < min_conf:
            continue
        if not _readable_box(det, frame_shape):
            continue
        cands.append({**det, "mph": int(mph)})
    if not cands:
        return None
    for i, a in enumerate(cands):
        for b in cands[i + 1 :]:
            if a["mph"] != b["mph"] and _iou(a["bbox"], b["bbox"]) >= SPEED_AMBIG_IOU:
                return "ambiguous"
    best = None
    best_score = -1.0
    for det in cands:
        x1, y1, x2, y2 = det["bbox"]
        area = max(1.0, float(x2 - x1) * float(y2 - y1))
        cx = 0.5 * (x1 + x2)
        score = float(det["conf"]) * (area ** 0.5) + 0.02 * cx
        if score > best_score:
            best_score = score
            best = det
    return best


class SpeedLimitTracker:
    def __init__(self, confirm_hits=3, act_conf=0.55, stale_frames=90, replace_hits=5):
        self.confirm_hits = int(confirm_hits)
        self.act_conf = float(act_conf)
        self.stale_frames = int(stale_frames)
        # Replacing a limit the driver already saw takes longer than the
        # first lock. A passing sign's class head flickers for a few frames.
        self.replace_hits = int(replace_hits)
        self.posted_mph = None
        self.candidate_mph = None
        self.hits = 0
        self.miss = 0
        self.status = "NONE"
        self.conf = 0.0
        self.bbox = None
        self.label = None

    def snapshot(self):
        return {
            "posted_mph": self.posted_mph,
            "candidate_mph": self.candidate_mph,
            "status": self.status,
            "conf": self.conf,
            "bbox": self.bbox,
            "label": self.label,
            "hits": self.hits,
        }

    def update(self, detections, ran_infer=True, min_conf=0.35, use_class=True,
               frame_shape=None):
        best = (
            pick_speed_det(
                detections, min_conf=min_conf, use_class=use_class,
                frame_shape=frame_shape,
            )
            if ran_infer
            else None
        )
        if best == "ambiguous":
            return self.snapshot()
        if best is None:
            if ran_infer:
                self.miss += 1
                if self.posted_mph is not None and self.miss >= self.stale_frames:
                    self.status = "STALE"
            return self.snapshot()

        self.miss = 0
        mph = best["mph"]
        self.conf = float(best["conf"])
        self.bbox = list(best["bbox"])
        self.label = best.get("class")

        if self.posted_mph == mph:
            self.candidate_mph = mph
            self.hits = self.confirm_hits
            self.status = "CONFIRMED"
            return self.snapshot()

        if self.candidate_mph == mph:
            self.hits += 1
        else:
            self.candidate_mph = mph
            self.hits = 1
        need = self.confirm_hits if self.posted_mph is None else self.replace_hits
        # A posted limit stays confirmed until a different value actually
        # replaces it. The candidate is only what the plate is considering.
        self.status = "CANDIDATE" if self.posted_mph is None else "CONFIRMED"
        if self.hits >= need and self.conf >= self.act_conf:
            self.posted_mph = mph
            self.status = "CONFIRMED"
        return self.snapshot()
