"""Ego-speed sources for the ADAS pipeline.

LDW only arms above 25 mph and FCW above 15 mph, and the cluster shows the
speed figure, so the pipeline needs the vehicle's speed from somewhere:

    auto            per-frame JSON log next to the video (Garmin HUD OCR), else none
    json            the JSON log only
    none            unknown speed (LDW stays off, FCW still runs)
    fixed:<v><unit> a constant, e.g. fixed:65mph or fixed:100kmh (bench demos)
    obd[:<port>]    live OBD-II PID 0x0D via an ELM327 adapter (pip install obd)

``get_mps(frame_index)`` returns metres per second or ``None``.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Optional

# Temporary stand-in when the open video has no *_speed.json beside it.
# Garmin OCR log: 39 mph at the start, one sample per frame.
_FALLBACK_SPEED_JSON = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "GRMN6693_540_speed.json",
))

MPH_TO_MPS = 0.44704
KMH_TO_MPS = 1.0 / 3.6


class SpeedSource:
    name = "none"
    label = ""          # short text for the cluster (e.g. "OBD-II", "log")

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
        if os.path.isfile(_FALLBACK_SPEED_JSON):
            from src.utils.ego_speed import EgoSpeedLog
            print(
                f"[ADAS] No speed log for {video_path}; using {_FALLBACK_SPEED_JSON}",
                flush=True,
            )
            return JsonSpeedSource(EgoSpeedLog.from_json(_FALLBACK_SPEED_JSON))
        if low == "json":
            print(f"[ADAS] No speed JSON for {video_path}; speed unknown", flush=True)
        return SpeedSource()
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
