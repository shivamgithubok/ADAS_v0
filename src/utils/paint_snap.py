"""Pull a projected lane polyline onto the white or yellow paint.

The 3D lane is in the OpenLane camera, so on this dashcam the projected line
crosses the real marking. Each sample is moved sideways to the nearest thin
paint run on its own image row. Gaps in a dashed line keep the shift of the
nearest sample that did find paint.
"""

from __future__ import annotations

import cv2
import numpy as np


def lane_paint_mask(frame: np.ndarray, calib=None) -> np.ndarray:
    """1 on thin lane paint. Wide bright things (cars, trees, signs) are dropped."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Top-hat keeps structures thinner than the kernel. A lane stripe is thin.
    # A car, a tree, or a building is not.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    white = (hat > 16) & (gray > 135) & (hsv[:, :, 1] < 110)
    yellow = (
        (hsv[:, :, 0] >= 12)
        & (hsv[:, :, 0] <= 42)
        & (hsv[:, :, 1] > 80)
        & (hsv[:, :, 2] > 120)
        & (hat > 10)
    )
    mask = (white | yellow).astype(np.uint8)
    if calib is not None:
        top = int(min(h - 2, max(0, float(calib.v_vp) + 6)))
    else:
        top = int(0.48 * h)
    mask[:top, :] = 0
    mask[int(0.94 * h) :, :] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        thickness = area / float(max(bw, bh, 1))
        # Lane paint is a few pixels thick. A car or a tree is not.
        if thickness > 14.0:
            mask[labels == i] = 0
    return mask


# Half-width of the paint search. After LANE_X_SCALE the projected line sits
# on the stripe (~0.1 m), so 1.0 m reaches paint without the next lane
# (~3.7 m). 2.2 m was required only while X was 1.24x too wide.
SNAP_WINDOW_M = 1.0
# After the line is on the paint, ignore a snap that leaps to another stripe.
SHIFT_JUMP_M = 0.40
SLOT_MATCH_M = 1.15
# A move bigger than this is the car changing lanes, not the same stripe.
LANE_MOVE_M = 1.50
SLOT_CONFIRM_HITS = 2
SLOT_HOLD = 6
MAX_DRAW_SLOTS = 4
DRAW_FREEZE_FRAMES = 12  # ~0.4 s; hold last strokes while the corridor is UNKNOWN
DRAW_BLEND = 0.40        # weight of this frame's snap; rest is the previous stroke


def _median_shift_m(raw, snapped, calib):
    """Median lateral correction, matched on image row so dropped points do not skew it."""
    if not raw or not snapped:
        return 0.0
    S = np.asarray(snapped, dtype=np.float64)
    if S.ndim != 2 or len(S) < 2:
        return 0.0
    S = S[np.argsort(S[:, 1])]
    shifts = []
    for u, v in raw:
        v = float(v)
        if v < S[0, 1] or v > S[-1, 1]:
            continue
        us = float(np.interp(v, S[:, 1], S[:, 0]))
        shifts.append(_px_to_m(calib, v, us - float(u)))
    if not shifts:
        return 0.0
    return float(np.median(shifts))


def _blend_polylines(prev, new, alpha):
    """Lateral blend of two polylines matched on image row."""
    if not prev or len(prev) < 2:
        return [(float(u), float(v)) for u, v in new]
    if not new or len(new) < 2:
        return [(float(u), float(v)) for u, v in prev]
    P = np.asarray(prev, dtype=np.float64)
    N = np.asarray(new, dtype=np.float64)
    P = P[np.argsort(P[:, 1])]
    N = N[np.argsort(N[:, 1])]
    v0 = max(float(P[0, 1]), float(N[0, 1]))
    v1 = min(float(P[-1, 1]), float(N[-1, 1]))
    if v1 - v0 < 8.0:
        return [(float(u), float(v)) for u, v in new]
    vs = np.linspace(v0, v1, max(len(N), 12))
    pu = np.interp(vs, P[:, 1], P[:, 0])
    nu = np.interp(vs, N[:, 1], N[:, 0])
    a = float(np.clip(alpha, 0.0, 1.0))
    u = (1.0 - a) * pu + a * nu
    return list(zip(u.tolist(), vs.tolist()))


class LaneDrawStabilizer:
    """One persistent stroke per physical lane.

    A new detection updates the matching stroke. It does not add another
    polyline on top. The paint snap remembers its lateral shift and ignores a
    jump onto a different stripe.
    """

    def __init__(self):
        self._slots = []
        self._freeze_age = 0

    def reset(self):
        self._slots = []
        self._freeze_age = 0

    def _match(self, x_m, role, used):
        best_i = None
        best_d = SLOT_MATCH_M
        for i, slot in enumerate(self._slots):
            if i in used:
                continue
            slot_role = slot.get("role")
            if role in ("left", "right", "adj_l", "adj_r") and slot_role in (
                "left", "right", "adj_l", "adj_r"
            ):
                if slot_role != role:
                    continue
            d = abs(float(slot["x"]) - float(x_m))
            if d > LANE_MOVE_M:
                continue
            same = role and slot_role == role
            if same:
                d *= 0.5
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _drawn_from_slots(self):
        drawn = []
        for slot in self._slots:
            if slot["hits"] < SLOT_CONFIRM_HITS or slot["miss"] > SLOT_HOLD:
                continue
            if len(slot["pts"]) >= 2:
                drawn.append({
                    "pts": [(float(u), float(v)) for u, v in slot["pts"]],
                    "role": slot["role"],
                    "x": float(slot["x"]),
                })
        return drawn

    def hold(self):
        """Keep the last strokes on screen; do not match new detections."""
        self._freeze_age += 1
        if self._freeze_age > DRAW_FREEZE_FRAMES:
            for slot in self._slots:
                slot["miss"] = int(slot["miss"]) + 1
            self._slots = [s for s in self._slots if s["miss"] <= SLOT_HOLD]
            return []
        return self._drawn_from_slots()

    def _snap_locked(self, pts, slot, frame, calib, mask):
        """Per-point snap onto the stripe. A locked line will not leap away.

        The drawn geometry is the snap itself. A single sideways shift of the
        raw line stays parallel to the wrong projection and misses the paint.
        """
        prev = 0.0 if slot is None else float(slot["shift_m"])
        on_paint = bool(slot is not None and slot.get("on_paint"))
        raw = [(float(u), float(v)) for u, v in pts]
        if len(raw) < 2:
            return raw, prev, on_paint
        snapped = snap_polyline_to_paint(
            pts, frame, calib, mask=mask, window_m=SNAP_WINDOW_M
        )
        snapped_f = [(float(u), float(v)) for u, v in snapped]
        shift = _median_shift_m(raw, snapped_f, calib)
        if on_paint and abs(shift - float(prev)) > SHIFT_JUMP_M:
            held = [(float(u), float(v)) for u, v in slot["pts"]]
            return held, prev, True
        found = abs(shift) > 0.04 or on_paint
        return snapped_f, shift, bool(found)

    def update(self, items, frame, calib, mask):
        """items: dicts with x (meters), pts (projected pixels), role.

        Returns the strokes to draw, each ``{pts, role, x}``.
        """
        used = set()
        order = sorted(range(len(items)), key=lambda i: abs(float(items[i]["x"])))
        for i in order:
            item = items[i]
            slot_i = self._match(item["x"], item.get("role"), used)
            slot = None if slot_i is None else self._slots[slot_i]
            snapped, shift, on_paint = self._snap_locked(
                item["pts"], slot, frame, calib, mask
            )
            if slot is None:
                if len(self._slots) >= MAX_DRAW_SLOTS:
                    continue
                role = item.get("role") or "other"
                for old in self._slots:
                    if old.get("role") == role and abs(float(old["x"]) - float(item["x"])) > LANE_MOVE_M:
                        old["miss"] = SLOT_HOLD + 1
                self._slots.append({
                    "x": float(item["x"]),
                    "pts": snapped,
                    "shift_m": float(shift),
                    "hits": SLOT_CONFIRM_HITS,
                    "miss": 0,
                    "role": role,
                    "on_paint": bool(on_paint),
                })
                used.add(len(self._slots) - 1)
                continue
            # Sit on this frame's snap, blended with the locked stroke so a
            # dashed-gap miss does not yank the line sideways.
            slot["pts"] = _blend_polylines(slot.get("pts"), snapped, DRAW_BLEND)
            slot["shift_m"] = float(shift)
            slot["on_paint"] = bool(on_paint or slot.get("on_paint"))
            slot["x"] = 0.5 * float(slot["x"]) + 0.5 * float(item["x"])
            slot["hits"] = int(slot["hits"]) + 1
            slot["miss"] = 0
            slot["role"] = item.get("role") or slot["role"]
            used.add(slot_i)

        live_x = [self._slots[i]["x"] for i in used]
        for i, slot in enumerate(self._slots):
            if i not in used:
                if any(abs(slot["x"] - x) < 0.80 for x in live_x):
                    slot["miss"] = SLOT_HOLD + 1
                    continue
                slot["miss"] = int(slot["miss"]) + 1
        self._slots = [s for s in self._slots if s["miss"] <= SLOT_HOLD]
        self._freeze_age = 0
        return self._drawn_from_slots()


def snap_polyline_to_paint(pts, frame, calib=None, mask=None, window_m=None):
    """Return the polyline with each point shifted onto nearby paint.

    `calib` is the dashcam ground calibration. It sets the search width in
    metres so a near-field line can reach the paint without jumping a whole
    lane. Without it, the search widens toward the bottom of the frame.
    """
    if pts is None or len(pts) < 2:
        return list(pts) if pts is not None else []
    if window_m is None:
        window_m = SNAP_WINDOW_M
    h, w = frame.shape[:2]
    if mask is None:
        mask = lane_paint_mask(frame, calib)

    src = [(float(u), float(v)) for u, v in pts]
    order = sorted(range(len(src)), key=lambda i: -src[i][1])
    # Shift is stored in metres when the dashcam calibration is known, so a
    # correction measured on a far dash still lands on the near paint.
    dx_m = [0.0] * len(src)
    du_px = [0.0] * len(src)
    hit = [False] * len(src)
    prev_m = None
    for i in order:
        u, v = src[i]
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= vi < h and 0 <= ui < w):
            continue
        if vi < int(0.40 * h) or vi > int(0.94 * h):
            continue
        half = _half_window_px(calib, vi, h, window_m)
        found = _nearest_paint_u(mask, ui, vi, half)
        if found is None:
            continue
        shift_px = float(found) - float(ui)
        shift_m = _px_to_m(calib, vi, shift_px)
        if prev_m is not None and abs(shift_m - prev_m) > 0.9:
            continue
        dx_m[i] = shift_m
        du_px[i] = shift_px
        hit[i] = True
        prev_m = shift_m

    if not any(hit):
        return [(int(round(u)), int(round(v))) for u, v in src]

    known = [i for i in range(len(src)) if hit[i]]
    if len(known) >= 3:
        vals = [dx_m[i] for i in known]
        for k, i in enumerate(known):
            lo, hi = max(0, k - 1), min(len(vals), k + 2)
            dx_m[i] = float(np.median(vals[lo:hi]))
    for i in range(len(src)):
        if hit[i]:
            continue
        nearest = min(known, key=lambda k: abs(src[k][1] - src[i][1]))
        dx_m[i] = dx_m[nearest]
        du_px[i] = _m_to_px(calib, src[i][1], dx_m[i], fallback_px=du_px[nearest])

    out = []
    for i, (u, v) in enumerate(src):
        uu = u + du_px[i]
        if 0.0 <= uu < w and 0.0 <= v < h:
            out.append((int(round(uu)), int(round(v))))
    return out


def snap_hit_fraction(pts, frame, calib=None, mask=None, window_m=None) -> float:
    """Share of polyline samples that land on paint after the snap search."""
    if not pts:
        return 0.0
    if window_m is None:
        window_m = SNAP_WINDOW_M
    h, _w = frame.shape[:2]
    if mask is None:
        mask = lane_paint_mask(frame, calib)
    hits = 0
    used = 0
    for u, v in pts:
        vi = int(round(v))
        ui = int(round(u))
        if not (int(0.40 * h) <= vi <= int(0.94 * h)):
            continue
        used += 1
        half = _half_window_px(calib, vi, h, window_m)
        if _nearest_paint_u(mask, ui, vi, half) is not None:
            hits += 1
    if used == 0:
        return 0.0
    return hits / float(used)


def _px_to_m(calib, v, shift_px):
    if calib is None:
        return float(shift_px)
    dv = float(v) - float(calib.v_vp)
    if dv < 8.0 or calib.f_px < 1.0:
        return float(shift_px)
    y = float(calib.hf) / dv
    return float(shift_px) * y / float(calib.f_px)


def _m_to_px(calib, v, shift_m, fallback_px=0.0):
    if calib is None:
        return float(fallback_px)
    dv = float(v) - float(calib.v_vp)
    if dv < 8.0 or calib.f_px < 1.0:
        return float(fallback_px)
    y = float(calib.hf) / dv
    return float(shift_m) * float(calib.f_px) / y


def _half_window_px(calib, v, h, window_m):
    if calib is not None:
        dv = float(v) - float(calib.v_vp)
        if dv > 8.0:
            y = float(calib.hf) / dv
            return int(max(10.0, min(140.0, float(calib.f_px) * float(window_m) / max(y, 1.0))))
    t = (float(v) - 0.40 * h) / max(1.0, 0.54 * h)
    t = float(np.clip(t, 0.0, 1.0))
    return int(12 + t * 80)


def _nearest_paint_u(mask, u, v, half):
    h, w = mask.shape[:2]
    if not (0 <= v < h):
        return None
    u0 = max(0, int(u) - int(half))
    u1 = min(w - 1, int(u) + int(half))
    if u1 - u0 < 3:
        return None
    cols = np.flatnonzero(mask[v, u0 : u1 + 1] > 0)
    if cols.size < 2:
        return None
    cols = cols + u0
    groups = []
    cur = [int(cols[0])]
    for c in cols[1:]:
        c = int(c)
        if c - cur[-1] <= 6:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)
    max_run = max(18, int(0.45 * (u1 - u0)))
    best = None
    best_dist = 1e9
    for g in groups:
        if len(g) < 2 or len(g) > max_run:
            continue
        # A run that touches the search edge is usually a car or a shadow, not a stripe.
        if g[0] <= u0 + 1 or g[-1] >= u1 - 1:
            continue
        center = 0.5 * (g[0] + g[-1])
        dist = abs(center - float(u))
        if dist < best_dist:
            best_dist = dist
            best = center
    return best
