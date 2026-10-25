"""Ego-speed sources for the ADAS pipeline.

LDW only arms above 25 mph and FCW above 15 mph, and the cluster shows the
speed figure, so the pipeline needs the vehicle's speed from somewhere:

    auto            per-frame JSON log next to the video, else the Garmin HUD
                    on the frame itself. Never another clip's log.
    json            the JSON log only
    none            unknown speed (LDW stays off, FCW still runs)
    fixed:<v><unit> a constant, e.g. fixed:65mph or fixed:100kmh (bench demos)
    obd[:<port>]    live OBD-II PID 0x0D via an ELM327 adapter (pip install obd)

``get_mps(frame_index)`` returns metres per second or ``None``.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Optional

MPH_TO_MPS = 0.44704
KMH_TO_MPS = 1.0 / 3.6


class SpeedSource:
    name = "none"
    label = ""          # short text for the cluster (e.g. "OBD-II", "log")

    def observe(self, frame, frame_index: int):
        """Optional per-frame update. JSON and fixed sources ignore the image."""
        return None

    def get_mps(self, frame_index: int) -> Optional[float]:
        return None

    def close(self):
        pass


class JsonSpeedSource(SpeedSource):
    name = "json"
    label = "speed log"

    def __init__(self, log):
        self._log = log

    def get_mps(self, frame_index):
        return self._log.get_mps(frame_index, min_mps=0.0)


class HudSpeedSource(SpeedSource):
    """Read ``NN MPH`` from a Garmin bottom bar on the current frame.

    Used when the open video has no speed JSON. Another clip's log is never
    substituted. A failed read holds the last good value; a jump bigger than
    ``max_jump`` mph is dropped as an OCR spike.
    """

    name = "hud"
    label = "Garmin HUD"

    def __init__(self, max_jump: int = 8):
        self._mph = None
        self._max_jump = int(max_jump)
        self._tmpl = None
        self._pending = None
        self._pending_hits = 0

    def observe(self, frame, frame_index: int):
        if frame is None:
            return
        from src.utils.ego_speed import read_hud_speed_mph
        if self._tmpl is None:
            import os
            from src.utils.ego_speed import _MPH_TEMPLATE_PATH
            import cv2
            if os.path.isfile(_MPH_TEMPLATE_PATH):
                self._tmpl = cv2.imread(_MPH_TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
            else:
                self._tmpl = False
        tmpl = self._tmpl if self._tmpl is not False else None
        mph = read_hud_speed_mph(frame, tmpl)
        if mph is None:
            return
        mph = int(mph)
        if self._mph is None or abs(mph - int(self._mph)) <= self._max_jump:
            self._mph = mph
            self._pending = None
            self._pending_hits = 0
            return
        # Garmin steps the HUD, and a stop can become 30 in one update.
        # A one-frame misread must not stick; the same new number twice does.
        if self._pending == mph:
            self._pending_hits += 1
        else:
            self._pending = mph
            self._pending_hits = 1
        if self._pending_hits >= 2:
            self._mph = mph
            self._pending = None
            self._pending_hits = 0

    def get_mps(self, frame_index):
        if self._mph is None:
            return None
        return float(self._mph) * MPH_TO_MPS


class FixedSpeedSource(SpeedSource):
    name = "fixed"
    label = "fixed speed"

    def __init__(self, mps: float):
        self._mps = max(0.0, float(mps))

    def get_mps(self, frame_index):
        return self._mps


class ObdSpeedSource(SpeedSource):
    """Polls vehicle speed over OBD-II on a background thread."""

    name = "obd"
    label = "OBD-II"
    STALE_S = 1.5

    def __init__(self, port: Optional[str] = None):
        try:
            import obd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OBD speed needs the 'obd' package: pip install obd") from exc
        self._obd = obd
        self._conn = obd.OBD(port) if port else obd.OBD()
        if not self._conn.is_connected():
            raise RuntimeError(f"OBD adapter not connected ({port or 'auto-detect'})")
        self._mps = None
        self._t = 0.0
        self._run = True
        self._th = threading.Thread(target=self._poll, name="obd-speed", daemon=True)
        self._th.start()

    def _poll(self):
        cmd = self._obd.commands.SPEED
        while self._run:
            try:
                r = self._conn.query(cmd)
                if not r.is_null():
                    self._mps = float(r.value.to("km/h").magnitude) * KMH_TO_MPS
                    self._t = time.monotonic()
            except Exception:
                pass
            time.sleep(0.1)

    def get_mps(self, frame_index):
        if self._mps is None or (time.monotonic() - self._t) > self.STALE_S:
            return None
        return self._mps

    def close(self):
        self._run = False
        try:
            self._conn.close()
        except Exception:
            pass


_FIXED_RE = re.compile(r"^fixed:\s*([0-9.]+)\s*(mph|kmh|km/h|kph|mps|m/s)?$", re.IGNORECASE)


def make_speed_source(spec: Optional[str], video_path: Optional[str] = None) -> SpeedSource:
    spec = (spec or "auto").strip()
    low = spec.lower()
    if low in ("auto", "json"):
        if video_path:
            try:
                from src.utils.ego_speed import EgoSpeedLog
                log = EgoSpeedLog.auto_load(video_path)
            except Exception:
                log = None
            if log is not None:
                return JsonSpeedSource(log)
        if low == "json":
            print(f"[ADAS] No speed JSON for {video_path}; speed unknown", flush=True)
            return SpeedSource()
        # auto: the frame may carry a Garmin HUD. No other video's log.
        return HudSpeedSource()
    if low == "none":
        return SpeedSource()
    m = _FIXED_RE.match(spec)
    if m:
        v = float(m.group(1))
        unit = (m.group(2) or "mph").lower()
        if unit == "mph":
            v *= MPH_TO_MPS
        elif unit in ("kmh", "km/h", "kph"):
            v *= KMH_TO_MPS
        return FixedSpeedSource(v)
    if low.startswith("obd"):
        port = spec.split(":", 1)[1] if ":" in spec else None
        return ObdSpeedSource(port)
    raise ValueError(
        f"Unknown speed source '{spec}'. Use auto, json, none, fixed:65mph, fixed:100kmh, obd or obd:/dev/ttyUSB0"
    )
