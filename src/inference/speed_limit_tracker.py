"""Temporal speed-limit state. One box never sets the posted limit."""

from __future__ import annotations

import re

_SPEED_RE = re.compile(r"speedLimit(\d+)", re.IGNORECASE)


def class_to_mph(name: str):
    if not name:
        return None
    m = _SPEED_RE.search(str(name))
    return int(m.group(1)) if m else None


def pick_speed_det(detections, min_conf=0.35, use_class=True):
    """Prefer larger, more confident speed-limit boxes; tie-break right of center.

    `det['mph']` is the LPRNet read when OCR succeeds. Otherwise the YOLO
    class digits are used (speedLimit35 and the rest).
    """
    best = None
    best_score = -1.0
    for det in detections:
        mph = det.get("mph")
        if mph is None and use_class:
            mph = class_to_mph(det.get("class", ""))
        if mph is None or det.get("conf", 0.0) < min_conf:
            continue
        x1, y1, x2, y2 = det["bbox"]
        area = max(1.0, float(x2 - x1) * float(y2 - y1))
        cx = 0.5 * (x1 + x2)
        score = float(det["conf"]) * (area ** 0.5) + 0.02 * cx
        if score > best_score:
            best_score = score
            best = {**det, "mph": mph}
    return best


class SpeedLimitTracker:
    def __init__(self, confirm_hits=3, act_conf=0.55, stale_frames=90):
        self.confirm_hits = int(confirm_hits)
        self.act_conf = float(act_conf)
        self.stale_frames = int(stale_frames)
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

    def update(self, detections, ran_infer=True, min_conf=0.35, use_class=True):
        best = (
            pick_speed_det(detections, min_conf=min_conf, use_class=use_class)
            if ran_infer
            else None
        )
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
        self.status = "CANDIDATE"
        if self.hits >= self.confirm_hits and self.conf >= self.act_conf:
            self.posted_mph = mph
            self.status = "CONFIRMED"
        return self.snapshot()
