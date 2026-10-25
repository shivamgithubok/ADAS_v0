"""Ego-centred lane world: physical lane boundaries tracked relative to the car.

    lane net -> LaneTrackerManager (per-line EKF) -> LaneWorld -> road-state corridor,
                                                                object lane assignment, BEV

The ego vehicle is the origin (x right, y forward, metres, ground plane). The
four boundaries nearest the car are tracked as slots labelled relative to the
ego lane::

        -2      -1    [ego]    +1      +2

Each slot stores its lateral offset at y = 0. All slots share one road shape
``s(y) = c1 y + c2 y^2 + c3 y^3``, so boundary k at range y is
``offset_k + s(y)``: heading and curvature are applied at the range where a
query is made, never at a fixed row.

Per frame:

1. Predict. The car advanced ``ds``; with heading c1 relative to the road every
   boundary moves by ``c1 * ds``.
2. Associate. Each freshly detected line is matched to the nearest predicted
   slot inside a gate, one-to-one. A slot label is never re-decided from "which
   pair looks most central this frame", so a detector re-lock or a frame where
   only the adjacent lane is visible cannot renumber the lanes.
3. Update. The mean innovation of the ego boundaries is ego lateral motion and
   moves every slot, measured or not. A smaller per-slot residual adapts the
   individual lane widths (lanes need not be equal).
4. Ego lane change. Only when a tracked ego boundary has passed the car centre
   by ``LW_LC_MARGIN_M`` for ``LW_LC_DWELL_S``. The labels shift by one; the
   offsets do not move, so nothing in the world jumps.
5. Candidates. A line that persists where no slot matches may replace a slot
   that has gone unmeasured, or one whose own detections are far less
   regular (a flickering ghost while the real paint is seen steadily), but
   only if every resulting lane width stays physically plausible.

Objects get their lane from ``ObjectLaneAssigner``: the boundaries evaluated at
the object's own range, with a range-dependent margin and a few frames of
confirmation before a label changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.inference import lane_filter_config as cfg
from src.inference.postprocess import ANCHOR_Y_STEPS
from src.tracking.lane_frame import C2_MAX, C3_MAX, road_shape
from src.utils.drivable_area import parse_lane_components

SLOTS = (-2, -1, 1, 2)
# A slot matched within this many frames counts as currently observed.
SEEN_FRAMES = 3
HEADING_ALPHA = 0.30
CURVATURE_ALPHA = 0.10
WIDTH_ALPHA = 0.20
CONF_ALPHA = 0.20
POLYLINE_Y = ANCHOR_Y_STEPS[ANCHOR_Y_STEPS <= 40.0 + 1e-6]


def boundary_ids(lane_id: int) -> Dict[int, int]:
    """Road-fixed id of each slot's boundary: lane L lies between boundaries L and L+1.

    Slot labels move on an ego lane change; these ids do not, so anything keyed
    by them (display easing, logs) follows the same physical line.
    """
    L = int(lane_id)
    return {-2: L - 1, -1: L, 1: L + 1, 2: L + 2}


@dataclass(frozen=True)
class LaneHit:
    """Where a point sits in the lane world at its own range."""

    index: int        # 0 ego, -1 left adjacent, +1 right adjacent, ...
    offset_m: float   # signed distance from that lane's centre
    margin_m: float   # distance to the nearer boundary of that lane (inside >= 0)
    left_m: float     # lane boundaries at the point's range
    right_m: float


@dataclass(frozen=True)
class LaneWorldState:
    """Immutable per-frame snapshot. Safe to hand across threads."""

    valid: bool = False
    held: bool = True
    confidence: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    offsets: Tuple[float, float, float, float] = (-5.55, -1.85, 1.85, 5.55)
    measured: Tuple[bool, bool, bool, bool] = (False, False, False, False)
    seen: Tuple[bool, bool, bool, bool] = (False, False, False, False)
    ego_lane_id: int = 0
    lane_changes: int = 0
    event: str = ""
    source: str = "none"
    frame: int = 0
    n_lines: int = 0
    n_matched: int = 0

    # ---------------------------------------------------------------- slots
    def offset(self, k: int) -> float:
        return float(self.offsets[SLOTS.index(int(k))])

    def is_measured(self, k: int) -> bool:
        return bool(self.measured[SLOTS.index(int(k))])

    def is_seen(self, k: int) -> bool:
        return bool(self.seen[SLOTS.index(int(k))])

    @property
    def ego_width(self) -> float:
        return self.offset(1) - self.offset(-1)

    @property
    def ego_center(self) -> float:
        """Ego-lane centre at y = 0 in the ego frame."""
        return 0.5 * (self.offset(-1) + self.offset(1))

    @property
    def ego_offset_in_lane(self) -> float:
        """Car centre relative to its lane centre (+ = right of centre)."""
        return -self.ego_center

    # ------------------------------------------------------------- geometry
    def shape(self, y) -> np.ndarray:
        return road_shape(self.c1, self.c2, self.c3, y, cfg.LW_SHAPE_MAX_Y_M)

    def boundary_x(self, k: int, y) -> np.ndarray:
        return self.offset(k) + self.shape(y)

    def center_x(self, y) -> np.ndarray:
        return self.ego_center + self.shape(y)

    def _boundaries_at(self, y: float) -> List[float]:
        s = float(self.shape(float(y)))
        return [float(o) + s for o in self.offsets]

    def lane_bounds(self, index: int, y: float) -> Tuple[float, float]:
        """(left, right) boundary x of lane `index` at range y."""
        b0, b1, b2, b3 = self._boundaries_at(y)
        idx = int(index)
        if idx == 0:
            return b1, b2
        if idx == -1:
            return b0, b1
        if idx == 1:
            return b2, b3
        if idx < -1:
            w = max(1e-3, b1 - b0)
            hi = b0 - (-idx - 2) * w
            return hi - w, hi
        w = max(1e-3, b3 - b2)
        lo = b3 + (idx - 2) * w
        return lo, lo + w

    def lane_center_x(self, index: int, y: float) -> float:
        lo, hi = self.lane_bounds(index, y)
        return 0.5 * (lo + hi)

    def lane_of(self, x: float, y: float) -> LaneHit:
        """Lane containing (x, y), judged against the boundaries at range y."""
        x = float(x)
        b0, b1, b2, b3 = self._boundaries_at(y)
        if b1 <= x <= b2:
            idx = 0
        elif x < b1:
            if x >= b0:
                idx = -1
            else:
                idx = -1 - int(math.ceil((b0 - x) / max(1e-3, b1 - b0)))
        else:
            if x <= b3:
                idx = 1
            else:
                idx = 1 + int(math.ceil((x - b3) / max(1e-3, b3 - b2)))
        lo, hi = self.lane_bounds(idx, y)
        return LaneHit(idx, x - 0.5 * (lo + hi), min(x - lo, hi - x), lo, hi)

    def boundary_polyline(self, k: int, ys=None) -> np.ndarray:
        """(N, 3) [x, y, 0] samples of slot k, the same layout as a tracked lane."""
        ys = POLYLINE_Y if ys is None else np.asarray(ys, dtype=np.float64)
        xs = self.boundary_x(k, ys)
        return np.column_stack([xs, ys, np.zeros_like(ys)]).astype(np.float32)

    def debug(self) -> dict:
        return {
            "valid": self.valid,
            "conf": round(self.confidence, 2),
            "ego_lane_id": self.ego_lane_id,
            "lane_changes": self.lane_changes,
            "source": self.source,
            "offsets": [round(float(o), 2) for o in self.offsets],
            "measured": "".join("M" if m else ("s" if s else ".")
                                for m, s in zip(self.measured, self.seen)),
            "ego_width": round(self.ego_width, 2),
            "ego_offset": round(self.ego_offset_in_lane, 2),
            "heading_deg": round(math.degrees(math.atan(self.c1)), 2),
            "n_lines": self.n_lines,
            "n_matched": self.n_matched,
            "event": self.event,
        }


class _Slot:
    __slots__ = ("offset", "age", "hits", "ever", "rate")

    def __init__(self, offset: float, ever: bool = False):
        self.offset = float(offset)
        self.age = 0 if ever else 10 ** 6
        self.hits = 1 if ever else 0
        self.ever = bool(ever)
        # Fraction of recent frames this boundary was detected in.
        self.rate = 1.0 if ever else 0.0

    def tick(self, hit: bool) -> None:
        a = cfg.LW_RATE_ALPHA
        self.rate = (1.0 - a) * self.rate + (a if hit else 0.0)

    @property
    def seen(self) -> bool:
        return self.ever and self.age <= SEEN_FRAMES


class LaneWorld:
    """Temporal ego-lane / boundary tracker. See module docstring."""

    def __init__(self) -> None:
        self._lane_id = 0
        self._changes = 0
        self._frame = 0
        self._clear()

    def reset(self) -> None:
        """New video / camera: forget geometry and lane identity."""
        self.__init__()

    def _clear(self) -> None:
        self._valid = False
        self._slots: Dict[int, _Slot] = {}
        self._c = [0.0, 0.0, 0.0]
        self._widths = {"left": None, "ego": None, "right": None}
        self._coast_s = 0.0
        self._lost_match_s = 0.0
        self._lc_t = {-1: 0.0, 1: 0.0}
        self._cands: List[dict] = []
        self._conf = 0.0
        self.matched_lines: Dict[int, np.ndarray] = {}
        self.state = LaneWorldState(ego_lane_id=self._lane_id, lane_changes=self._changes)

    # ------------------------------------------------------------- helpers
    def _shape(self, y):
        return road_shape(self._c[0], self._c[1], self._c[2], y, cfg.LW_SHAPE_MAX_Y_M)

    @staticmethod
    def _samples(line) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        xs, ys, _zs, vis = parse_lane_components(line)
        keep = vis & (ys <= cfg.LW_SHAPE_MAX_Y_M + 1e-6) & (ys > 0.0)
        if int(keep.sum()) < 2:
            return None
        return np.asarray(ys[keep], dtype=np.float64), np.asarray(xs[keep], dtype=np.float64)

    @staticmethod
    def _near(ys: np.ndarray) -> np.ndarray:
        near = ys <= cfg.LW_OFFSET_MAX_Y_M + 1e-6
        return near if int(near.sum()) >= 2 else np.ones_like(ys, dtype=bool)

    def _line_offset(self, ys: np.ndarray, xs: np.ndarray) -> float:
        """Line's lateral offset at y = 0 under the current road shape."""
        m = self._near(ys)
        w = 1.0 / (1.0 + 0.05 * ys[m])
        return float(np.sum(w * (xs[m] - self._shape(ys[m]))) / np.sum(w))

    def _width(self, side: str) -> float:
        w = self._widths.get(side)
        if w is None:
            w = self._widths.get("ego")
        return float(cfg.LW_WIDTH_DEFAULT_M if w is None else w)

    # ---------------------------------------------------------------- init
    def _acquire(self, feats) -> Optional[str]:
        """Pick the ego lane as the boundaries that bracket the car."""
        lines = []
        for _line, ys, xs in feats:
            m = self._near(ys)
            wts = 1.0 / (1.0 + 0.05 * ys[m])
            A = np.column_stack([np.ones(int(m.sum())), ys[m]]) * wts[:, None]
            try:
                a, _b = np.linalg.lstsq(A, xs[m] * wts, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            lines.append((float(a), ys, xs))
        if not lines:
            return None
        lefts = sorted([l for l in lines if l[0] < 0.0], key=lambda l: -l[0])
        rights = sorted([l for l in lines if l[0] >= 0.0], key=lambda l: l[0])
        w_prior = self._width("ego")
        best = None
        for i, left in enumerate(lefts[:3]):
            for j, right in enumerate(rights[:3]):
                w = right[0] - left[0]
                if not (cfg.LW_WIDTH_MIN_M <= w <= cfg.LW_WIDTH_MAX_M):
                    continue
                # Skipping a nearer line costs a whole lane; width is a tie-break.
                cost = i + j + abs(w - w_prior) / w_prior
                if best is None or cost < best[0]:
                    best = (cost, left, right)
        if best is not None:
            _cost, left, right = best
            self._fit_shape_from([(left[1], left[2]), (right[1], right[2])], alpha=(1.0, 1.0))
            o_l = self._line_offset(left[1], left[2])
            o_r = self._line_offset(right[1], right[2])
            how = f"pair L{o_l:+.2f} R{o_r:+.2f}"
        else:
            # One usable side. The car must sit inside the lane it implies.
            single = min(lines, key=lambda l: abs(l[0]))
            if abs(single[0]) > w_prior:
                return None
            self._fit_shape_from([(single[1], single[2])], alpha=(1.0, 1.0))
            o = self._line_offset(single[1], single[2])
            if o < 0.0:
                o_l, o_r = o, o + w_prior
            else:
                o_l, o_r = o - w_prior, o
            how = f"one line {o:+.2f}, width {w_prior:.2f}"
        self._slots = {
            -2: _Slot(o_l - self._width("left")),
            -1: _Slot(o_l),
            1: _Slot(o_r),
            2: _Slot(o_r + self._width("right")),
        }
        self._valid = True
        self._coast_s = 0.0
        self._lost_match_s = 0.0
        self._lc_t = {-1: 0.0, 1: 0.0}
        self._cands = []
        return how

    # --------------------------------------------------------- association
    def _width_ok(self, k: int, x: float) -> bool:
        """Would boundary k at x leave its inner lane a plausible width?"""
        s = self._slots
        inner = {-2: -1, -1: 1, 1: -1, 2: 1}[k]
        w = abs(float(x) - s[inner].offset)
        return cfg.LW_WIDTH_MIN_M <= w <= cfg.LW_WIDTH_MAX_M

    def _associate(self, feats):
        obs = [self._line_offset(ys, xs) for _line, ys, xs in feats]
        pairs = []
        for i, ob in enumerate(obs):
            for k, s in self._slots.items():
                gate = cfg.LW_GATE_M if s.seen else cfg.LW_GATE_UNSEEN_M
                d = abs(ob - s.offset)
                if d <= gate and self._width_ok(k, ob):
                    pairs.append((d, i, k))
        pairs.sort()
        used_i, used_k, matched = set(), set(), []
        for _d, i, k in pairs:
            if i in used_i or k in used_k:
                continue
            used_i.add(i)
            used_k.add(k)
            matched.append((k, obs[i], i))
        unmatched = [(i, obs[i]) for i in range(len(obs)) if i not in used_i]
        return matched, unmatched

    def _apply(self, matched, feats) -> float:
        # Ego lateral motion: weighted mean innovation of boundaries already
        # being tracked. A slot re-found after a gap is adopted directly and
        # must not drag the others.
        num = den = 0.0
        for k, ob, _i in matched:
            s = self._slots[k]
            if not s.seen:
                continue
            w = 1.0 if abs(k) == 1 else 0.5
            num += w * (ob - s.offset)
            den += w
        delta = num / den if den > 0.0 else 0.0
        for s in self._slots.values():
            s.offset += cfg.LW_COMMON_GAIN * delta
        hit = set()
        self.matched_lines = {}
        for k, ob, i in matched:
            s = self._slots[k]
            if s.seen:
                s.offset += cfg.LW_SLOT_GAIN * (ob - s.offset)
            else:
                s.offset = float(ob)
            s.age = 0
            s.hits += 1
            s.ever = True
            hit.add(k)
            self.matched_lines[k] = feats[i][0]
        for k, s in self._slots.items():
            s.tick(k in hit)
            if k not in hit:
                s.age += 1
                s.hits = 0
        return delta

    def _fit_shape_from(self, samples, alpha=(HEADING_ALPHA, CURVATURE_ALPHA)) -> None:
        """Shared heading / curvature from lines that are parallel on the road.

        Each line gets its own intercept, so the fit never feeds tracked slot
        offsets back into the heading. samples: [(ys, xs)].
        """
        n = len(samples)
        if n == 0:
            return
        y_all = np.concatenate([ys for ys, _xs in samples])
        b = np.concatenate([xs for _ys, xs in samples])
        w = 1.0 / (1.0 + 0.05 * y_all)
        full = len(b) >= n + 3 and float(np.ptp(y_all)) >= 15.0
        cols = 3 if full else 1
        if len(b) < n + cols:
            return
        A = np.zeros((len(b), n + cols))
        row = 0
        for i, (ys, _xs) in enumerate(samples):
            A[row:row + len(ys), i] = 1.0
            row += len(ys)
        A[:, n] = y_all
        if full:
            A[:, n + 1] = y_all ** 2
            A[:, n + 2] = y_all ** 3
        try:
            sol = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
        except np.linalg.LinAlgError:
            return
        if not np.all(np.isfinite(sol)):
            return
        c1 = float(sol[n])
        c2 = float(sol[n + 1]) if full else self._c[1]
        c3 = float(sol[n + 2]) if full else self._c[2]
        a1, a2 = alpha
        self._c[0] = (1.0 - a1) * self._c[0] + a1 * c1
        self._c[1] = float(np.clip((1.0 - a2) * self._c[1] + a2 * c2, -C2_MAX, C2_MAX))
        self._c[2] = float(np.clip((1.0 - a2) * self._c[2] + a2 * c3, -C3_MAX, C3_MAX))

    def _fit_shape(self, matched, feats) -> None:
        self._fit_shape_from([(feats[i][1], feats[i][2]) for _k, _ob, i in matched])

    # ------------------------------------------------------ slot upkeep
    def _update_widths(self) -> None:
        s = self._slots
        for side, (a, b) in (("left", (-2, -1)), ("ego", (-1, 1)), ("right", (1, 2))):
            if s[a].age == 0 and s[b].age == 0:
                w = s[b].offset - s[a].offset
                if cfg.LW_WIDTH_MIN_M <= w <= cfg.LW_WIDTH_MAX_M:
                    prev = self._widths[side]
                    self._widths[side] = w if prev is None else (
                        (1.0 - WIDTH_ALPHA) * prev + WIDTH_ALPHA * w
                    )

    def _place_unseen_outer(self) -> None:
        """Outer slots never seen (or long gone) sit one remembered width out."""
        s = self._slots
        stale = 4 * int(cfg.LW_ADOPT_MISS)
        for k, side, sign in ((-2, "left", -1.0), (2, "right", 1.0)):
            slot = s[k]
            if slot.ever and (slot.age == 0 or (
                slot.age <= stale and self._width_ok(k, slot.offset)
            )):
                continue
            slot.offset = s[k // 2].offset + sign * self._width(side)
            slot.ever = False
            slot.age = 10 ** 6
            slot.hits = 0

    def _check_ego_width(self) -> str:
        s = self._slots
        w = s[1].offset - s[-1].offset
        if cfg.LW_WIDTH_MIN_M <= w <= cfg.LW_WIDTH_MAX_M:
            return ""
        # Two boundaries cannot both be right. Drop the one with less support
        # and let the other side plus the remembered width carry the lane.
        weak = -1 if s[-1].hits <= s[1].hits else 1
        keep = -weak
        s[weak].age = max(s[weak].age, int(cfg.LW_ADOPT_MISS))
        s[weak].hits = 0
        ego_w = self._width("ego")
        s[weak].offset = s[keep].offset + (ego_w if weak == 1 else -ego_w)
        side = "left" if weak == -1 else "right"
        return f"ego width {w:.2f} m implausible: released {side} boundary"

    def _update_candidates(self, unmatched, feats, delta) -> None:
        """Lines no boundary claims, tracked so a persistent one can be adopted.

        A candidate survives short detection gaps (dashed paint, a missed
        frame); its `rate` is how often it has actually been detected.
        """
        a = cfg.LW_RATE_ALPHA
        for c in self._cands:
            c["offset"] += cfg.LW_COMMON_GAIN * delta
            c["seen"] = False
        for i, ob in sorted(unmatched, key=lambda t: t[1]):
            best, best_d = None, cfg.LW_CAND_MATCH_M
            for c in self._cands:
                d = abs(c["offset"] - ob)
                if not c["seen"] and d <= best_d:
                    best, best_d = c, d
            if best is None:
                self._cands.append({"offset": float(ob), "hits": 1, "seen": True,
                                    "miss": 0, "rate": a, "line": feats[i][0]})
            else:
                best["offset"] = 0.5 * (best["offset"] + float(ob))
                best["hits"] += 1
                best["seen"] = True
                best["miss"] = 0
                best["line"] = feats[i][0]
        for c in self._cands:
            if c["seen"]:
                c["rate"] = (1.0 - a) * c["rate"] + a
            else:
                c["rate"] *= 1.0 - a
                c["miss"] += 1
        self._cands = [c for c in self._cands if c["miss"] <= int(cfg.LW_ADOPT_MISS)]

    def _replaceable(self, slot: _Slot, cand: dict) -> bool:
        """A boundary gone unmeasured, or one held by a far less reliable line."""
        if not slot.ever or slot.age >= int(cfg.LW_ADOPT_MISS):
            return True
        return slot.rate <= cfg.LW_SWAP_SLOT_RATE and cand["rate"] >= cfg.LW_SWAP_CAND_RATE

    def _adopt(self) -> str:
        s = self._slots
        wmin, wmax = cfg.LW_WIDTH_MIN_M, cfg.LW_WIDTH_MAX_M
        events = []
        for k in (-1, 1, -2, 2):
            slot = s[k]
            pool = []
            for c in self._cands:
                if not c["seen"] or c["hits"] < int(cfg.LW_ADOPT_HITS):
                    continue
                if not self._replaceable(slot, c):
                    continue
                x = c["offset"]
                if k == -1:
                    ok = wmin <= s[1].offset - x <= wmax and (
                        not s[-2].seen or x - s[-2].offset >= 0.5 * wmin)
                elif k == 1:
                    ok = wmin <= x - s[-1].offset <= wmax and (
                        not s[2].seen or s[2].offset - x >= 0.5 * wmin)
                elif k == -2:
                    ok = wmin <= s[-1].offset - x <= wmax
                else:
                    ok = wmin <= x - s[1].offset <= wmax
                if ok:
                    pool.append(c)
            if not pool:
                continue
            c = min(pool, key=lambda c: abs(c["offset"] - slot.offset))
            old, old_rate, stale = slot.offset, slot.rate, slot.age >= int(cfg.LW_ADOPT_MISS)
            slot.offset = float(c["offset"])
            slot.age = 0
            slot.hits = c["hits"]
            slot.rate = float(c["rate"])
            slot.ever = True
            self.matched_lines[k] = c["line"]
            self._cands.remove(c)
            why = ("persistent line" if stale or old_rate <= 0.0 else
                   f"detected {c['rate']:.0%} of frames vs {old_rate:.0%}")
            events.append(
                f"slot {k:+d} re-found at {slot.offset:+.2f} m (was {old:+.2f}, {why})"
            )
        return "; ".join(events)

    def _shift(self, side: int) -> None:
        """Ego crossed into the neighbouring lane on `side`. Relabel only."""
        s = self._slots
        if side > 0:
            new_right = s[2].offset + self._width("right")
            self._slots = {-2: s[-1], -1: s[1], 1: s[2], 2: _Slot(new_right)}
            self._widths = {"left": self._widths["ego"], "ego": self._widths["right"],
                            "right": self._widths["right"]}
        else:
            new_left = s[-2].offset - self._width("left")
            self._slots = {-2: _Slot(new_left), -1: s[-2], 1: s[-1], 2: s[1]}
            self._widths = {"left": self._widths["left"], "ego": self._widths["left"],
                            "right": self._widths["ego"]}
        self._lane_id += int(side)
        self._changes += 1
        self._lc_t = {-1: 0.0, 1: 0.0}
        self._cands = []

    def _check_lane_change(self, dt: float) -> str:
        if self._conf < cfg.LW_MIN_CONF_FOR_LANE_CHANGE:
            self._lc_t = {-1: 0.0, 1: 0.0}
            return ""
        s = self._slots
        for side in (1, -1):
            o = s[side].offset
            crossed = o < -cfg.LW_LC_MARGIN_M if side > 0 else o > cfg.LW_LC_MARGIN_M
            self._lc_t[side] = self._lc_t[side] + dt if crossed else 0.0
            if self._lc_t[side] >= cfg.LW_LC_DWELL_S:
                t = self._lc_t[side]
                name = "right" if side > 0 else "left"
                self._shift(side)
                return (
                    f"ego lane change {name}: car centre {abs(o):.2f} m past the {name} "
                    f"boundary for {t:.2f} s -> lane id {self._lane_id}"
                )
        return ""

    # -------------------------------------------------------------- update
    def update(
        self,
        lines: Optional[Sequence[np.ndarray]],
        fresh: Optional[Sequence[bool]] = None,
        dt: float = 1.0 / 30.0,
        speed_mps: Optional[float] = None,
    ) -> LaneWorldState:
        """Fold one frame of tracked lane lines in.

        `fresh[i]` False marks a line that only coasted this frame (no detection);
        such lines are ignored, the world coasts on its own prediction instead.
        """
        self._frame += 1
        dt = float(np.clip(dt, 1e-3, 0.5))
        ds = max(0.0, float(speed_mps or 0.0)) * dt
        lines = list(lines or [])
        if fresh is not None and len(fresh) == len(lines):
            lines = [l for l, f in zip(lines, fresh) if f]
        feats = []
        for line in lines:
            smp = self._samples(line)
            if smp is not None:
                feats.append((line, smp[0], smp[1]))

        events = []
        self.matched_lines = {}
        if self._valid:
            for s in self._slots.values():
                s.offset += self._c[0] * ds
            for c in self._cands:
                c["offset"] += self._c[0] * ds
        else:
            how = self._acquire(feats) if feats else None
            if how is None:
                self._conf = 0.0
                return self._publish("", feats, 0)
            events.append(f"acquired ego lane ({how})")

        matched, unmatched = self._associate(feats)
        delta = 0.0
        if matched:
            self._coast_s = 0.0
            self._lost_match_s = 0.0
            delta = self._apply(matched, feats)
            self._fit_shape(matched, feats)
            self._update_widths()
        else:
            for s in self._slots.values():
                s.tick(False)
                s.age += 1
                s.hits = 0
            if feats:
                self._lost_match_s += dt
                if self._lost_match_s >= cfg.LW_RELOCK_S:
                    how = self._acquire(feats)
                    if how is not None:
                        events.append(
                            f"re-acquired ego lane ({how}): {len(feats)} lines matched no "
                            f"boundary for {cfg.LW_RELOCK_S:.1f} s"
                        )
                        matched, unmatched = self._associate(feats)
                        if matched:
                            delta = self._apply(matched, feats)
            else:
                self._coast_s += dt
                if self._coast_s >= cfg.LW_HOLD_S:
                    self._clear()
                    return self._publish(
                        f"lost: no lane lines for {cfg.LW_HOLD_S:.1f} s", feats, 0
                    )

        self._update_candidates(unmatched, feats, delta)
        ev = self._adopt()
        if ev:
            events.append(ev)
        ev = self._check_ego_width()
        if ev:
            events.append(ev)
        self._place_unseen_outer()

        s = self._slots
        if s[-1].age == 0 and s[1].age == 0:
            score = 1.0
        elif s[-1].age == 0 or s[1].age == 0:
            score = 0.6
        elif s[-2].age == 0 or s[2].age == 0:
            score = 0.35
        else:
            score = 0.0
        self._conf = (1.0 - CONF_ALPHA) * self._conf + CONF_ALPHA * score

        ev = self._check_lane_change(dt)
        if ev:
            events.append(ev)
        return self._publish("; ".join(events), feats, len(matched))

    def _publish(self, event: str, feats, n_matched: int) -> LaneWorldState:
        if not self._valid:
            self.state = LaneWorldState(
                valid=False, ego_lane_id=self._lane_id, lane_changes=self._changes,
                event=event, frame=self._frame, n_lines=len(feats),
            )
            return self.state
        s = self._slots
        measured = tuple(s[k].age == 0 for k in SLOTS)
        seen = tuple(s[k].seen for k in SLOTS)
        if measured[1] and measured[2]:
            source = "both"
        elif measured[1]:
            source = "left_only"
        elif measured[2]:
            source = "right_only"
        elif measured[0] or measured[3]:
            source = "outer_only"
        else:
            source = "coast"
        self.state = LaneWorldState(
            valid=True,
            held=not any(measured),
            confidence=float(self._conf),
            c1=float(self._c[0]),
            c2=float(self._c[1]),
            c3=float(self._c[2]),
            offsets=tuple(float(s[k].offset) for k in SLOTS),
            measured=measured,
            seen=seen,
            ego_lane_id=self._lane_id,
            lane_changes=self._changes,
            event=event,
            source=source,
            frame=self._frame,
            n_lines=len(feats),
            n_matched=int(n_matched),
        )
        return self.state


class ObjectLaneAssigner:
    """Per-track lane label from the object's physical position at its range.

    A label changes only after the object has been inside the new lane, by a
    range-dependent margin, for OBJ_LANE_SWITCH_FRAMES consecutive updates. A
    change of ego lane relabels every object by the same amount automatically,
    because labels are always relative to the current ego lane.
    """

    def __init__(self) -> None:
        self._st: Dict[int, dict] = {}
        self._lane_id: Optional[int] = None

    def reset(self) -> None:
        self._st.clear()
        self._lane_id = None

    def drop(self, live_ids) -> None:
        live = set(int(t) for t in live_ids)
        self._st = {k: v for k, v in self._st.items() if k in live}

    def update(self, track_id: int, x_m: float, y_m: float,
               world: Optional[LaneWorldState]) -> Tuple[Optional[int], float, str]:
        """(lane index, offset from that lane's centre, reason)."""
        tid = int(track_id)
        st = self._st.get(tid)
        if world is None or not world.valid:
            if st is None:
                return None, 0.0, "no lane model"
            return st["idx"], st["off"], "no lane model: holding last lane"
        if self._lane_id is not None and world.ego_lane_id != self._lane_id:
            # Ego changed lane: every stored label is relative to the old ego lane.
            shift = world.ego_lane_id - self._lane_id
            for v in self._st.values():
                v["idx"] -= shift
                if v["cand"] is not None:
                    v["cand"] -= shift
        self._lane_id = world.ego_lane_id

        hit = world.lane_of(x_m, y_m)
        if st is None:
            st = {"idx": hit.index, "cand": None, "n": 0, "off": hit.offset_m}
            self._st[tid] = st
            return hit.index, hit.offset_m, f"new: x {x_m:+.2f} in lane {hit.index:+d}"
        tol = cfg.OBJ_LANE_MARGIN_M + cfg.OBJ_LANE_MARGIN_PER_M * max(0.0, float(y_m))
        reason = "geometry"
        if hit.index == st["idx"]:
            st["cand"], st["n"] = None, 0
        elif hit.margin_m < tol:
            st["cand"], st["n"] = None, 0
            reason = f"on boundary ({hit.margin_m:.2f} < {tol:.2f} m): holding lane {st['idx']:+d}"
        else:
            if st["cand"] == hit.index:
                st["n"] += 1
            else:
                st["cand"], st["n"] = hit.index, 1
            need = int(cfg.OBJ_LANE_SWITCH_FRAMES)
            if st["n"] >= need:
                reason = f"moved lane {st['idx']:+d} -> {hit.index:+d} ({hit.margin_m:.2f} m inside)"
                st["idx"], st["cand"], st["n"] = hit.index, None, 0
            else:
                reason = f"pending lane {hit.index:+d} ({st['n']}/{need})"
        lo, hi = world.lane_bounds(st["idx"], y_m)
        st["off"] = float(x_m) - 0.5 * (lo + hi)
        return st["idx"], st["off"], reason
