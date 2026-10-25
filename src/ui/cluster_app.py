"""Instrument-cluster front end: QML cluster + perception pipeline glue.

    perception worker ──frame_processed──▶ ClusterController
                                             ├─ ThreatAssessor   (green/yellow/red, banner, audio)
                                             ├─ BevSceneDriver   (3D scene inside Cluster.qml)
                                             ├─ CameraProvider   (front-camera tile)
                                             ├─ AlertAudio       (earcons)
                                             └─ hud map ──▶ Cluster.qml

The worker is either ``InferenceWorker`` (TensorRT on video / camera) or
``SimulationWorker`` (scripted perception); both emit the same signal.
"""

from __future__ import annotations

import glob
import os
import time
import wave
from datetime import datetime
from typing import Optional

import numpy as np
from PySide6.QtCore import (
    Property, QObject, QSize, QTimer, QUrl, Signal, Slot, Qt,
)
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtQuick import QQuickImageProvider, QQuickView

from src.adas.threat import CAUTION, WARNING, ThreatAssessor
from src.ui.bev_quick3d import BevSceneDriver

_QML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qml")
CLUSTER_QML = os.path.join(_QML_DIR, "Cluster.qml")
MPS_TO_MPH = 2.2369362920544
MPS_TO_KMH = 3.6


# --------------------------------------------------------------- camera tile
class CameraProvider(QQuickImageProvider):
    """Serves the latest annotated camera frame as image://camera/<n>."""

    def __init__(self):
        super().__init__(QQuickImageProvider.Image)
        self._img = QImage()

    def set_frame(self, rgb: np.ndarray):
        if rgb is None or rgb.ndim != 3:
            return
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        self._img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

    def requestImage(self, _id, size, requested_size):  # noqa: N802 (Qt API)
        return self._img


# ------------------------------------------------------------------- earcons
def _tone(freq, dur, rate=44100, amp=0.35):
    n = int(rate * dur)
    t = np.arange(n) / rate
    x = np.sin(2 * np.pi * freq * t) + 0.18 * np.sin(4 * np.pi * freq * t)
    ramp = max(1, int(0.006 * rate))
    env = np.ones(n)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return amp * x * env / 1.18


def _silence(dur, rate=44100):
    return np.zeros(int(rate * dur))


def _write_wav(path, mono, pan=0.0, rate=44100):
    """pan: -1 left, 0 centre, +1 right (constant power)."""
    a = (pan + 1.0) * np.pi / 4.0
    left, right = np.cos(a) * mono, np.sin(a) * mono
    st = np.stack([left, right], axis=1)
    pcm = np.clip(st * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def build_earcons(cache_dir):
    """Synthesize the cluster earcons (spec §7, UX-AUD-01) once per machine."""
    os.makedirs(cache_dir, exist_ok=True)
    pulse = lambda f, on, off: np.concatenate([_tone(f, on, amp=0.5), _silence(off)])
    sounds = {
        # Repeating urgent tones: 4 and 6 pulses per second, looped.
        "fcw": (np.concatenate([pulse(1800, 0.09, 0.16)] * 4), 0.0),
        "fcw_plus": (np.concatenate([pulse(2200, 0.07, 0.097)] * 6), 0.0),
        "ldw_left": (np.concatenate([_tone(1000, 0.09), _silence(0.07), _tone(1000, 0.09)]), -0.9),
        "ldw_right": (np.concatenate([_tone(1000, 0.09), _silence(0.07), _tone(1000, 0.09)]), 0.9),
        "caution": (_tone(880, 0.18, amp=0.25), 0.0),
        "overspeed": (np.concatenate([_tone(660, 0.12, amp=0.3), _tone(880, 0.14, amp=0.3)]), 0.0),
    }
    paths = {}
    for name, (mono, pan) in sounds.items():
        p = os.path.join(cache_dir, f"{name}.wav")
        if not os.path.isfile(p):
            _write_wav(p, mono, pan)
        paths[name] = p
    return paths


class AlertAudio(QObject):
    """One alert sound at a time; the highest priority (caller decides) wins."""

    def __init__(self, enabled=True, parent=None):
        super().__init__(parent)
        self.muted = not enabled
        self._fx = {}
        self._loop_name = None
        try:
            from PySide6.QtMultimedia import QSoundEffect
            cache = os.path.join(os.path.expanduser("~"), ".cache", "adas_cluster", "earcons")
            for name, path in build_earcons(cache).items():
                fx = QSoundEffect(self)
                fx.setSource(QUrl.fromLocalFile(path))
                fx.setVolume(0.9)
                self._fx[name] = fx
            # setLoopCount takes an int; Loop.Infinite == -2.
            self._infinite = -2
        except Exception as exc:  # no QtMultimedia / no audio device
            print(f"[Cluster] Audio disabled: {exc}", flush=True)
            self._fx = {}

    def _stop_all(self):
        for fx in self._fx.values():
            if fx.isPlaying():
                fx.stop()

    def play(self, name):
        if self.muted or name not in self._fx or self._loop_name is not None:
            return
        self._stop_all()
        fx = self._fx[name]
        fx.setLoopCount(1)
        fx.play()

    def loop(self, name: Optional[str]):
        if name == self._loop_name:
            return
        if self._loop_name is not None and self._loop_name in self._fx:
            self._fx[self._loop_name].stop()
        self._loop_name = name
        if name is None or self.muted or name not in self._fx:
            return
        self._stop_all()
        fx = self._fx[name]
        fx.setLoopCount(self._infinite)
        fx.play()

    def set_muted(self, muted):
        self.muted = bool(muted)
        if self.muted:
            self._stop_all()
        elif self._loop_name:
            name, self._loop_name = self._loop_name, None
            self.loop(name)


# ------------------------------------------------------------ SoC telemetry
def read_soc_temp_c() -> Optional[float]:
    """Hottest CPU/GPU/SoC thermal zone (Jetson exposes cpu-thermal, gpu-thermal, tj-thermal)."""
    best = None
    for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            with open(os.path.join(zone, "type")) as f:
                kind = f.read().strip().lower()
            if not any(k in kind for k in ("cpu", "gpu", "soc", "tj", "x86_pkg")):
                continue
            with open(os.path.join(zone, "temp")) as f:
                t = int(f.read().strip()) / 1000.0
            if 0.0 < t < 150.0:
                best = t if best is None else max(best, t)
        except (OSError, ValueError):
            continue
    return best


# --------------------------------------------------------------- controller
class ClusterController(QObject):
    hudChanged = Signal()

    def __init__(self, view: QQuickView, make_worker, source: str, units="mph",
                 audio=True, parent=None):
        super().__init__(parent)
        self.view = view
        self._make_worker = make_worker
        self.source = source
        self.units = "kmh" if str(units).lower() in ("kmh", "km/h", "kph") else "mph"
        self.assessor = ThreatAssessor(units=self.units)
        self.audio = AlertAudio(enabled=audio, parent=self)
        self.camera = CameraProvider()
        self.bev: Optional[BevSceneDriver] = None
        self.worker = None

        self._hud = {}
        self._frame_id = 0
        self._has_frame = False
        self._show_camera = False
        self._lane_draw = True
        self._paused = False
        self._trip_m = 0.0
        self._soc = read_soc_temp_c()
        self._last_frame_t = 0.0
        self._last = {}

        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick_slow)
        self._clock.start()
        self._soc_timer = QTimer(self)
        self._soc_timer.setInterval(2000)
        self._soc_timer.timeout.connect(self._read_soc)
        self._soc_timer.start()

    # ------------------------------------------------------------ wiring
    def attach_scene(self):
        root = self.view.rootObject()
        item = root.findChild(QObject, "bevScene") if root is not None else None
        if item is None:
            raise RuntimeError("Cluster.qml has no bevScene item")
        self.bev = BevSceneDriver(item, parent=self)
        self.bev.set_units(self.units)
        self._push_hud({})

    def start(self):
        self.worker = self._make_worker(self.source)
        self.worker.frame_processed.connect(self.on_frame)
        self.worker.status_message.connect(self._on_status)
        if hasattr(self.worker, "set_lane_draw"):
            self.worker.set_lane_draw(self._lane_draw)
        self.worker.start()

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.audio.loop(None)

    # ------------------------------------------------------------ QML API
    @Property("QVariantMap", notify=hudChanged)
    def hud(self):
        return self._hud

    @Slot()
    def togglePause(self):
        if self.worker is not None:
            self._paused = bool(self.worker.toggle_pause())
            if self.bev is not None:
                self.bev.set_paused(self._paused)
            if self._paused:
                self.audio.loop(None)
            self._refresh()

    @Slot()
    def toggleCamera(self):
        self._show_camera = not self._show_camera
        self._refresh()

    @Slot()
    def toggleLaneDraw(self):
        self._lane_draw = not self._lane_draw
        if self.worker is not None and hasattr(self.worker, "set_lane_draw"):
            self.worker.set_lane_draw(self._lane_draw)
        self._refresh()

    @Slot()
    def toggleMute(self):
        self.audio.set_muted(not self.audio.muted)
        self._refresh()

    @Slot()
    def toggleFullscreen(self):
        if self.view.windowState() & Qt.WindowFullScreen:
            self.view.showNormal()
        else:
            self.view.showFullScreen()

    @Slot()
    def exitFullscreen(self):
        if self.view.windowState() & Qt.WindowFullScreen:
            self.view.showNormal()

    @Slot()
    def quit(self):
        self.stop()
        QGuiApplication.quit()

    @Slot(str)
    def openVideo(self, url):
        path = QUrl(url).toLocalFile() if "://" in url else url
        if not path:
            return
        self.stop()
        self.source = path
        self.assessor.reset()
        self._trip_m = 0.0
        self._has_frame = False
        self._paused = False
        if self.bev is not None:
            self.bev.lane_frame.reset()
            self.bev.set_paused(False)
        self.start()

    # ------------------------------------------------------------ frames
    @Slot(str)
    def _on_status(self, msg):
        print(f"[Cluster] {msg}", flush=True)

    @Slot(np.ndarray, list, list, object, str, object, object, float, float, object, float, object)
    def on_frame(self, frame_rgb, proposals, objs, cipo_obj, cipo_status,
                 left_3d, right_3d, fps, latency_ms, speed_mps=None,
                 source_dt=1.0 / 30.0, alerts=None):
        alerts = alerts or {}
        dt = float(source_dt or 1.0 / 15.0)
        threat = self.assessor.update(objs, cipo_obj, cipo_status, alerts, speed_mps, dt)

        if self.bev is not None:
            self.bev.update_bev_data(
                proposals, objs, cipo_status, left_3d, right_3d,
                speed_mps=speed_mps, dt=dt, alerts=alerts, threat=threat,
            )

        if not self._paused:
            # Loop first: a one-shot cue on the frame a loop ends must not be
            # swallowed by the loop that is just stopping.
            self.audio.loop(threat.audio_loop)
            if threat.audio:
                self.audio.play(threat.audio)

        if frame_rgb is not None and frame_rgb.size and frame_rgb.any():
            self.camera.set_frame(frame_rgb)
            self._has_frame = True
            self._frame_id += 1

        if speed_mps is not None:
            self._trip_m += max(0.0, float(speed_mps)) * dt

        self._last = {
            "threat": threat, "alerts": alerts, "speed": speed_mps,
            "fps": fps, "latency": latency_ms, "cipo_status": cipo_status,
            "n_objs": len(objs or []),
        }
        self._refresh()

    # ------------------------------------------------------------ hud map
    def _tick_slow(self):
        self._refresh()

    def _read_soc(self):
        self._soc = read_soc_temp_c()

    def _source_label(self):
        src = str(self.source or "")
        if src.upper() == "SIMULATION":
            return "sim", "Simulation"
        low = src.lower()
        if low.startswith("csi"):
            return "camera", "CSI camera " + (src.split(":", 1)[1] if ":" in src else "0")
        if low.startswith(("camera:", "/dev/video", "v4l2:", "gst:")):
            return "camera", "Camera " + src.split(":", 1)[-1]
        return "video", os.path.basename(src) or "No source"

    def _refresh(self):
        self._push_hud(self._last)

    def _push_hud(self, last):
        threat = last.get("threat")
        alerts = last.get("alerts") or {}
        speed = last.get("speed")
        caps = alerts.get("caps") or {}
        kmh = self.units == "kmh"
        conv = MPS_TO_KMH if kmh else MPS_TO_MPH

        hud = {"units": self.units}
        kind, label = self._source_label()
        hud.update(sourceKind=kind, sourceLabel=label, paused=self._paused)

        # Speed
        if speed is None:
            hud["speedText"] = "--"
            hud["speedSourceText"] = "No speed source" if last else ""
        else:
            hud["speedText"] = str(int(round(max(0.0, float(speed)) * conv)))
            src_label = str(alerts.get("speed_source") or "")
            hud["speedSourceText"] = src_label if src_label in ("fixed speed", "OBD-II") else ""
        hud["speedUnit"] = "KM/H" if kmh else "MPH"
        hud["miniBev"] = self.bev.mini_bev_json() if self.bev is not None else "{}"

        # Speed limit (sign OCR reads US mph plates)
        isa = alerts.get("isa") or {}
        posted = isa.get("posted_mph")
        # Candidate flips every class change (65 → 50 → 30). The plate shows
        # only a limit that has already been confirmed.
        lim = posted
        if lim is not None and kmh:
            lim = int(round(float(lim) * 1.609344 / 5.0) * 5)
        hud["limit"] = None if lim is None else int(lim)
        hud["limitConfirmed"] = posted is not None
        hud["overLimit"] = int(threat.over_limit) if threat is not None else 0

        # Alerts
        if threat is not None:
            hud.update(
                banner=threat.banner or "",
                bannerSide=threat.banner_side or "",
                redActive=bool(threat.red_active),
                vignette=bool(threat.vignette),
            )
        else:
            hud.update(banner="", bannerSide="", redActive=False, vignette=False)

        # Telltales + function rows
        mph = None if speed is None else float(speed) * MPS_TO_MPH
        fcw = str(alerts.get("fcw") or "OFF")
        ldw = str(alerts.get("ldw") or "OFF")
        road = str(alerts.get("road_status") or "")
        if not last:
            tt_fcw = tt_ldw = "off"
            fcw_text = ldw_text = "Starting"
        else:
            if caps and not caps.get("vehicles", False):
                tt_fcw, fcw_text = "off", "No model"
            elif fcw in ("FCW", "FCW+"):
                tt_fcw, fcw_text = "active", "Warning"
            elif mph is not None and mph < 15.0:
                tt_fcw, fcw_text = "standby", ("Below 24 km/h" if kmh else "Below 15 mph")
            else:
                tt_fcw, fcw_text = "armed", "Armed"
            if caps and not caps.get("lanes", False):
                tt_ldw, ldw_text = "off", "No model"
            elif ldw in ("LEFT", "RIGHT"):
                tt_ldw, ldw_text = "active", "Warning"
            elif road and road not in ("CONFIRMED", "PREDICTED"):
                tt_ldw, ldw_text = "fault", "No lanes"
            elif mph is None:
                tt_ldw, ldw_text = "standby", "No speed"
            elif mph < 25.0:
                tt_ldw, ldw_text = "standby", ("Below 40 km/h" if kmh else "Below 25 mph")
            else:
                tt_ldw, ldw_text = "armed", "Armed"
        hud.update(ttFcw=tt_fcw, fcwText=fcw_text, ttLdw=tt_ldw, ldwText=ldw_text,
                   ttIsa="armed" if caps.get("signs") else "off")

        # Perception status pill
        if not last:
            lane_track, lane_text = "off", "Starting"
        elif caps and not caps.get("lanes", False):
            lane_track, lane_text = "off", "Lane model off"
        elif road == "CONFIRMED":
            lane_track, lane_text = "locked", "Lanes locked"
        elif road == "PREDICTED":
            lane_track, lane_text = "held", "Lanes held"
        else:
            lane_track, lane_text = "searching", "Searching lanes"
        n = int(alerts.get("n_objects", last.get("n_objs", 0)) or 0)
        hud.update(
            laneTrack=lane_track, laneTrackText=lane_text,
            objectsText=f"{n} tracked",
            statusCaption="Perception · paused" if self._paused else "Perception",
        )

        # Camera
        hud.update(hasFrame=self._has_frame, frameId=self._frame_id,
                   showCamera=self._show_camera, laneDraw=self._lane_draw,
                   muted=self.audio.muted)

        # Bottom bar
        fps = last.get("fps")
        lat = last.get("latency")
        hud["fpsText"] = "—" if not fps else f"{fps:.0f}"
        hud["latencyText"] = "—" if lat is None else f"{lat:.0f}"
        if kmh:
            hud["tripText"], hud["tripUnit"] = f"{self._trip_m / 1000.0:.1f}", "km"
        else:
            hud["tripText"], hud["tripUnit"] = f"{self._trip_m / 1609.344:.1f}", "mi"
        hud.update(engLanes=bool(caps.get("lanes")), engVehicles=bool(caps.get("vehicles")),
                   engSigns=bool(caps.get("signs")))

        # Clock + SoC
        now = datetime.now()
        hud["clock"] = now.strftime("%I:%M").lstrip("0")
        hud["ampm"] = now.strftime("%p")
        if self._soc is not None:
            hud["socTemp"] = int(round(self._soc))
            hud["socHot"] = self._soc >= 85.0
        else:
            hud["socTemp"] = None

        self._hud = hud
        self.hudChanged.emit()


# ------------------------------------------------------------------ window
def create_cluster_window(make_worker, source, units="mph", audio=True,
                          size=(1440, 732), fullscreen=False, title="ADAS Cluster"):
    view = QQuickView()
    view.setTitle(title)
    view.setColor(Qt.black)
    view.setResizeMode(QQuickView.SizeRootObjectToView)
    controller = ClusterController(view, make_worker, source, units=units, audio=audio)
    engine = view.engine()
    engine.addImageProvider("camera", controller.camera)
    engine.rootContext().setContextProperty("cluster", controller)
    view.setSource(QUrl.fromLocalFile(CLUSTER_QML))
    if view.status() == QQuickView.Error:
        msgs = "; ".join(e.toString() for e in view.errors())
        raise RuntimeError(f"Failed to load Cluster.qml: {msgs}")
    view.resize(QSize(int(size[0]), int(size[1])))
    controller.attach_scene()
    if fullscreen:
        view.showFullScreen()
    else:
        view.show()
    return view, controller


def capture_scenario(view, controller, sim, times, out_dir, settle_ms=350, on_done=None):
    """Step the simulation headless-fast and grab the cluster at given times (s)."""
    os.makedirs(out_dir, exist_ok=True)
    pending = sorted(float(t) for t in times)
    state = {"grab_at": None}
    timer = QTimer(view)
    timer.setInterval(4)

    def tick():
        if not pending:
            timer.stop()
            if on_done:
                on_done()
            return
        if state["grab_at"] is not None:
            if time.perf_counter() >= state["grab_at"]:
                t = pending.pop(0)
                img = view.grabWindow()
                path = os.path.join(out_dir, f"cluster_t{t:05.1f}s.png")
                img.save(path)
                print(f"[Cluster] saved {path}", flush=True)
                state["grab_at"] = None
            return
        sim.step()
        if sim.t >= pending[0]:
            state["grab_at"] = time.perf_counter() + settle_ms / 1000.0

    timer.timeout.connect(tick)
    timer.start()
    return timer
