"""Threat assessment and alert arbitration for the cluster.

Turns one frame of perception (tracked objects, CIPO, LDW/FCW snapshot, ego
speed) into what the driver display needs:

* a threat level per tracked object (0 neutral, 1 green, 2 yellow, 3 red),
  with hysteresis so colours never flicker;
* the lead-vehicle summary for the gap band and distance tag;
* left/right lane-line states for lane departure;
* the single banner and audio cue that win arbitration this frame.

Thresholds follow the UI/UX spec (section 5). They are calibration values:
change them here, nowhere else. Nothing in this module issues warnings on its
own — FCW and LDW decisions still come from ``src.inference.ldw_fcw``; this
module only decides how strongly each object and lane line is drawn.

Conventions (same as ``CIPOTracker``):
    Z_3d  range ahead of the camera, m
    X_3d  lateral, m (negative = left)
    vy    relative longitudinal velocity, m/s (negative = closing)
    vx    relative lateral velocity, m/s
    lane_index  0 ego lane, -1 left, +1 right, 99/None unknown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

NEUTRAL, SAFE, CAUTION, WARNING = 0, 1, 2, 3
LEVEL_NAMES = ("neutral", "safe", "caution", "warning")

# ---------------------------------------------------------------- thresholds
# Longitudinal (in-path / lead vehicle)
THW_SAFE_S = 2.0           # time headway at or above this is green
THW_WARN_S = 1.0           # below this is red
TTC_CAUTION_S = 4.0        # closing with TTC at or below this is yellow
TTC_WARN_S = 2.6           # at or below this is red
CLOSING_FAST_MPS = 20.0 / 3.6   # "approaching fast" when closing faster than 20 km/h
CLOSING_FAST_THW_S = 3.0        # ... and the gap is under this time headway
MIN_CLOSING_MPS = 0.5      # slower closing than this is treated as not closing
MIN_EGO_MPS = 2.0          # below this time headway is meaningless

# Lateral (adjacent lanes, front camera only)
ADJ_CLOSE_M = 12.0         # adjacent vehicle this close is yellow (matches OBJ alert)
CUT_IN_VX_MPS = 0.6        # lateral speed toward the ego lane that reads as a cut-in
CUT_IN_RANGE_M = 30.0
LANE_CHANGE_CONFLICT_M = 15.0   # LDW toward an occupied lane inside this range is red

# Display range: objects farther than this are never tinted.
MAX_TINT_RANGE_M = 80.0

# Hysteresis
DOWNGRADE_HOLD_S = 0.5     # a lower level must persist this long before it shows
RED_MIN_HOLD_S = 1.0       # once red, stay red at least this long

# Speed-limit compliance (display units)
OVER_LIMIT_CAUTION = 2.0   # more than this over: plate outline yellow (speedo rounding tolerance)
OVER_LIMIT_WARN = 10.0     # this far over: plate outline red + single tone

# Banner kinds, highest priority first (spec section 7).
BANNER_PRIORITY = (
    "fcw_plus",     # imminent collision: solid red "BRAKE"
    "fcw",          # collision warning
    "lane_change",  # departing toward an occupied adjacent lane
    "ldw",          # lane departure
)


@dataclass
class _TrackState:
    shown: int = NEUTRAL
    pending: int = NEUTRAL
    pending_since: float = 0.0
    red_since: float = -1e9
    last_seen: float = 0.0


@dataclass
class LeadSummary:
    track_id: int
    level: int
    range_m: float
    thw_s: Optional[float]
    ttc_s: Optional[float]
    closing_mps: float
    closing_fast: bool

    def as_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "level": self.level,
            "range_m": self.range_m,
            "thw_s": self.thw_s,
            "ttc_s": self.ttc_s,
            "closing_mps": self.closing_mps,
            "closing_fast": self.closing_fast,
        }


@dataclass
class ThreatFrame:
    """Everything the cluster draws for one frame."""

    levels: Dict[int, int] = field(default_factory=dict)
    primary_tid: Optional[int] = None
    lead: Optional[LeadSummary] = None
    lane_left: str = "normal"     # normal | caution | warning | off
    lane_right: str = "normal"
    banner: Optional[str] = None  # one of BANNER_PRIORITY or None
    banner_side: Optional[str] = None
    red_active: bool = False      # any red alert: dims non-driving content
    vignette: bool = False        # red screen edge (forward collision only)
    audio: Optional[str] = None   # earcon to (re)start this frame
    audio_loop: Optional[str] = None  # earcon that should keep repeating
    over_limit: int = NEUTRAL     # speed-limit plate outline level

    def as_dict(self) -> dict:
        return {
            "levels": dict(self.levels),
            "primary_tid": self.primary_tid,
            "lead": self.lead.as_dict() if self.lead else None,
            "lane_left": self.lane_left,
            "lane_right": self.lane_right,
            "banner": self.banner,
            "banner_side": self.banner_side,
            "red_active": self.red_active,
            "vignette": self.vignette,
            "audio": self.audio,
            "audio_loop": self.audio_loop,
            "over_limit": self.over_limit,
        }


def _f(obj: dict, key: str, default: float = 0.0) -> float:
    try:
        v = obj.get(key, default)
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def _lane_index(obj: dict) -> Optional[int]:
    li = obj.get("lane_index")
    try:
        li = int(li)
    except (TypeError, ValueError):
        return None
    return None if abs(li) >= 99 else li


def time_headway(range_m: float, ego_mps: Optional[float]) -> Optional[float]:
    if ego_mps is None or ego_mps < MIN_EGO_MPS:
        return None
    return float(range_m) / float(ego_mps)


def time_to_collision(range_m: float, closing_mps: float) -> Optional[float]:
    if closing_mps < MIN_CLOSING_MPS:
        return None
    return float(range_m) / float(closing_mps)


def longitudinal_level(range_m: float, closing_mps: float, ego_mps: Optional[float],
                       fcw_active: bool = False, band: str = "SAFE") -> int:
    """Green / yellow / red for an in-path object (spec table, section 5)."""
    ttc = time_to_collision(range_m, closing_mps)
    thw = time_headway(range_m, ego_mps)
    if fcw_active:
        return WARNING
    if ttc is not None and ttc <= TTC_WARN_S:
        return WARNING
    if thw is not None and thw < THW_WARN_S:
        return WARNING
    if ttc is not None and ttc <= TTC_CAUTION_S:
        return CAUTION
    if thw is not None and thw < THW_SAFE_S:
        return CAUTION
    if closing_mps > CLOSING_FAST_MPS and (thw is None or thw < CLOSING_FAST_THW_S):
        return CAUTION
    # No ego speed (no HUD log / OBD): fall back to the tracker's range band so
    # a close lead still reads as caution rather than silently green.
    if thw is None and band in ("WARNING", "DANGER"):
        return CAUTION
    return SAFE


class ThreatAssessor:
    """Stateful per-frame threat classifier. Call ``update`` once per frame."""

    def __init__(self, units: str = "mph"):
        self.units = "kmh" if str(units).lower() in ("kmh", "km/h", "kph") else "mph"
        self._tracks: Dict[int, _TrackState] = {}
        self._t = 0.0
        self._prev_banner = None  # (kind, side)
        self._prev_lead_caution = False
        self._prev_over: int = NEUTRAL
        self._over_tone_limit = None

    # ------------------------------------------------------------- helpers
    def _hysteresis(self, tid: int, raw: int) -> int:
        st = self._tracks.get(tid)
        now = self._t
        if st is None:
            st = _TrackState(shown=raw, pending=raw, pending_since=now,
                             red_since=now if raw == WARNING else -1e9)
            self._tracks[tid] = st
        st.last_seen = now
        if raw >= st.shown:
            if raw == WARNING and st.shown != WARNING:
                st.red_since = now
            st.shown = raw
            st.pending = raw
            st.pending_since = now
            return st.shown
        # Lower level requested.
        if st.shown == WARNING and (now - st.red_since) < RED_MIN_HOLD_S:
            return st.shown
        if raw != st.pending:
            st.pending = raw
            st.pending_since = now
        if (now - st.pending_since) >= DOWNGRADE_HOLD_S:
            st.shown = raw
        return st.shown

    def _speed_display(self, ego_mps: Optional[float]) -> Optional[float]:
        if ego_mps is None:
            return None
        return ego_mps * (3.6 if self.units == "kmh" else 2.2369362920544)

    def _limit_display(self, isa: dict) -> Optional[float]:
        posted = isa.get("posted_mph") if isa else None
        if posted is None:
            return None
        # Sign OCR reads US mph plates. Convert only for km/h display.
        return float(posted) * (1.609344 if self.units == "kmh" else 1.0)

    # ------------------------------------------------------------- main
    def update(self, objects: Iterable[dict], cipo_obj: Optional[dict], cipo_status: str,
               alerts: Optional[dict], ego_mps: Optional[float], dt: float) -> ThreatFrame:
        self._t += max(0.0, float(dt or 0.0))
        alerts = alerts or {}
        objects = list(objects or [])
        out = ThreatFrame()

        fcw = str(alerts.get("fcw") or "OFF")
        fcw_on = fcw in ("FCW", "FCW+")
        ldw = str(alerts.get("ldw") or "OFF")
        ldw_side = alerts.get("ldw_side") or (ldw if ldw in ("LEFT", "RIGHT") else None)
        cipo_tid = int(cipo_obj.get("track_id", -1)) if cipo_obj else -1

        live = set()
        lane_change_conflict = None
        best = None  # (level, -ttc_or_range) for primary threat

        for obj in objects:
            tid = int(obj.get("track_id", -1) or -1)
            if tid <= 0:
                continue
            z = _f(obj, "Z_3d", 99.0)
            if z <= 0.0:
                continue
            live.add(tid)
            x = _f(obj, "X_3d")
            closing = -_f(obj, "vy")
            li = _lane_index(obj)
            is_cipo = tid == cipo_tid or bool(obj.get("is_cipo"))
            in_path = is_cipo or (bool(obj.get("in_path")) and (li in (0, None)))

            if z > MAX_TINT_RANGE_M:
                raw = NEUTRAL
            elif in_path:
                raw = longitudinal_level(
                    z, closing, ego_mps,
                    fcw_active=(fcw_on and is_cipo),
                    band=str(cipo_status or "SAFE") if is_cipo else "SAFE",
                )
            elif li is not None and abs(li) == 1:
                side = "LEFT" if li < 0 else "RIGHT"
                toward_ego = (li < 0 and _f(obj, "vx") > CUT_IN_VX_MPS) or \
                             (li > 0 and _f(obj, "vx") < -CUT_IN_VX_MPS)
                if ldw_side == side and z < LANE_CHANGE_CONFLICT_M:
                    raw = WARNING
                    lane_change_conflict = side
                elif z < ADJ_CLOSE_M or (toward_ego and z < CUT_IN_RANGE_M):
                    raw = CAUTION
                else:
                    raw = SAFE
            else:
                raw = NEUTRAL

            level = self._hysteresis(tid, raw)
            out.levels[tid] = level
            if level > NEUTRAL:
                ttc = time_to_collision(z, closing)
                key = (level, -(ttc if ttc is not None else 99.0), -z)
                if best is None or key > best[0]:
                    best = (key, tid)

            if is_cipo:
                thw = time_headway(z, ego_mps)
                out.lead = LeadSummary(
                    track_id=tid,
                    level=level,
                    range_m=z,
                    thw_s=thw,
                    ttc_s=time_to_collision(z, closing),
                    closing_mps=max(0.0, closing),
                    closing_fast=closing > CLOSING_FAST_MPS,
                )

        # Forget tracks that left the scene.
        for tid in list(self._tracks):
            if tid not in live and (self._t - self._tracks[tid].last_seen) > 2.0:
                del self._tracks[tid]

        out.primary_tid = best[1] if best else None

        # ----------------------------------------------------- lane lines
        caps = alerts.get("caps") or {}
        road_status = str(alerts.get("road_status") or "")
        if caps and not caps.get("lanes", True):
            out.lane_left = out.lane_right = "off"
        elif ldw in ("LEFT", "RIGHT"):
            if ldw == "LEFT":
                out.lane_left = "warning"
            else:
                out.lane_right = "warning"
        elif ldw == "WATCH":
            lg = alerts.get("left_gap")
            rg = alerts.get("right_gap")
            if lg is not None and rg is not None:
                if float(lg) <= float(rg):
                    out.lane_left = "caution"
                else:
                    out.lane_right = "caution"
        if road_status and road_status not in ("CONFIRMED", "PREDICTED") \
                and out.lane_left == "normal" and out.lane_right == "normal":
            out.lane_left = out.lane_right = "off"

        # ----------------------------------------------------- arbitration
        candidates = []
        if fcw == "FCW+":
            candidates.append(("fcw_plus", None))
        if fcw == "FCW":
            candidates.append(("fcw", None))
        if lane_change_conflict:
            candidates.append(("lane_change", lane_change_conflict))
        if ldw in ("LEFT", "RIGHT"):
            candidates.append(("ldw", ldw))
        if candidates:
            candidates.sort(key=lambda c: BANNER_PRIORITY.index(c[0]))
            out.banner, out.banner_side = candidates[0]

        out.red_active = out.banner is not None
        out.vignette = out.banner in ("fcw", "fcw_plus")

        # ----------------------------------------------------- speed limit
        isa = alerts.get("isa") or {}
        limit = self._limit_display(isa)
        speed = self._speed_display(ego_mps)
        if limit is not None and speed is not None:
            over = speed - limit
            if over >= OVER_LIMIT_WARN:
                out.over_limit = WARNING
            elif over > OVER_LIMIT_CAUTION:
                out.over_limit = CAUTION
            else:
                out.over_limit = SAFE
        else:
            out.over_limit = NEUTRAL

        # ----------------------------------------------------- audio cues
        # Repeating tones while forward-collision banners hold; one-shot cues
        # on the rising edge of everything else. One sound at a time.
        if out.banner == "fcw_plus":
            out.audio_loop = "fcw_plus"
        elif out.banner == "fcw":
            out.audio_loop = "fcw"
        if (out.banner, out.banner_side) != self._prev_banner and out.banner in ("ldw", "lane_change"):
            out.audio = "ldw_left" if out.banner_side == "LEFT" else "ldw_right"
        lead_caution = bool(out.lead and out.lead.level == CAUTION and out.lead.closing_fast)
        if out.banner is None and lead_caution and not self._prev_lead_caution:
            out.audio = "caution"
        if out.over_limit == WARNING and self._prev_over != WARNING \
                and self._over_tone_limit != limit and out.banner is None:
            out.audio = "overspeed"
            self._over_tone_limit = limit

        self._prev_banner = (out.banner, out.banner_side)
        self._prev_lead_caution = lead_caution
        self._prev_over = out.over_limit
        return out

    def reset(self):
        self._tracks.clear()
        self._prev_banner = None
        self._prev_lead_caution = False
        self._prev_over = NEUTRAL
        self._over_tone_limit = None


def format_lead_tag(lead: Optional[LeadSummary], units: str = "mph") -> Optional[dict]:
    """Text for the distance tag above the lead vehicle (spec UX-GAP-02/03)."""
    if lead is None:
        return None
    kmh = str(units).lower() in ("kmh", "km/h", "kph")
    if kmh:
        dist = f"{lead.range_m:.0f} m"
    else:
        dist = f"{lead.range_m * 3.28084:.0f} ft"
    text = dist
    if lead.level >= CAUTION and lead.thw_s is not None:
        text += f" · {lead.thw_s:.1f} s"
    elif lead.level >= CAUTION and lead.ttc_s is not None:
        text += f" · TTC {lead.ttc_s:.1f} s"
    sub = ""
    if lead.closing_fast or lead.level == WARNING:
        if lead.closing_mps >= MIN_CLOSING_MPS:
            v = lead.closing_mps * (3.6 if kmh else 2.2369362920544)
            sub = f"▼ {v:.0f} {'km/h' if kmh else 'mph'}"
    return {"text": text, "sub": sub, "level": int(lead.level)}


def level_name(level: int) -> str:
    try:
        return LEVEL_NAMES[int(level)]
    except (IndexError, TypeError, ValueError):
        return LEVEL_NAMES[0]


def clamp_level(v) -> int:
    try:
        return max(NEUTRAL, min(WARNING, int(v)))
    except (TypeError, ValueError):
        return NEUTRAL


__all__ = [
    "NEUTRAL", "SAFE", "CAUTION", "WARNING",
    "ThreatAssessor", "ThreatFrame", "LeadSummary",
    "longitudinal_level", "time_headway", "time_to_collision",
    "format_lead_tag", "level_name", "clamp_level",
]
