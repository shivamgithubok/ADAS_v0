#!/usr/bin/env python3
"""
ADAS instrument cluster launcher (PySide6 + Qt Quick 3D, TensorRT perception).

Usage:
    python scripts/run_pyside6_app.py                                # demo clip, cluster UI
    python scripts/run_pyside6_app.py --video path/to/drive.mp4
    python scripts/run_pyside6_app.py --camera 0                     # USB camera (/dev/video0)
    python scripts/run_pyside6_app.py --camera csi                   # Jetson CSI camera (Argus)
    python scripts/run_pyside6_app.py --sim                          # scripted perception, no engines
    python scripts/run_pyside6_app.py --fullscreen --units kmh
    python scripts/run_pyside6_app.py --ui classic                   # previous widget cockpit
    python scripts/run_pyside6_app.py --test-mode                    # headless smoke test

Keys in the cluster: Space pause · C camera · L lane paint · M mute · O open video ·
F full screen · Esc leave full screen · Q quit
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
LAUNCH_DIR = os.getcwd()  # user paths are resolved against this, then we chdir

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtQuick import QQuickWindow, QSGRendererInterface  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

DEFAULT_VIDEO = "data/demo/clip_8s.mp4"
SHOT_TIMES = (6.0, 12.6, 16.6, 17.55, 28.0, 36.8, 46.2, 52.5)


def _configure_graphics():
    """OpenGL RHI on Jetson; must run before the QApplication exists."""
    os.environ.setdefault("QSG_RHI_BACKEND", "opengl")
    try:
        QQuickWindow.setGraphicsApi(QSGRendererInterface.OpenGL)
    except Exception:
        pass


def _camera_source(spec: str) -> str:
    s = str(spec).strip()
    low = s.lower()
    if low.startswith(("csi", "gst:")):
        return s
    if low.startswith("/dev/video"):
        return s
    if s.isdigit():
        return f"camera:{s}"
    return s if low.startswith("camera:") else f"camera:{s}"


def _parse_size(text):
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("size must look like 1440x732")


def main():
    p = argparse.ArgumentParser(description="ADAS instrument cluster (Jetson Orin)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video", type=str, default=None, help="Input video file")
    src.add_argument("--camera", type=str, default=None,
                     help="Live camera: 0 / /dev/video0 (USB), csi or csi:1 (Jetson CSI), gst:<pipeline>")
    src.add_argument("--sim", action="store_true", help="Scripted perception (no TensorRT engines needed)")
    p.add_argument("--model", type=str, default="models/anchor3dlane_raw.engine", help="Lane TensorRT engine")
    p.add_argument("--speed", type=str, default="auto",
                   help="Ego speed: auto | json | none | fixed:65mph | fixed:100kmh | obd | obd:/dev/ttyUSB0")
    p.add_argument("--units", choices=("mph", "kmh"), default="mph", help="Display units")
    p.add_argument("--ui", choices=("cluster", "classic"), default="cluster", help="Front end")
    p.add_argument("--bev", choices=("quick3d", "painter"), default="quick3d",
                   help="Classic UI only: BEV backend")
    p.add_argument("--fullscreen", action="store_true", help="Start full screen (kiosk)")
    p.add_argument("--size", type=_parse_size, default=(1440, 732), help="Window size, e.g. 1920x976")
    p.add_argument("--mute", action="store_true", help="Start with alert sounds off")
    p.add_argument("--screenshots", type=str, default=None,
                   help="With --sim: save cluster images at key alert moments to this folder, then exit")
    p.add_argument("--test-mode", action="store_true", help="Headless smoke test; exits after a few seconds")
    args = p.parse_args()

    # Resolve user-supplied paths against where the command was run, then move
    # to the repo root (model, asset and QML paths are repo-relative).
    def _abs(path):
        return path if path is None or os.path.isabs(path) else os.path.join(LAUNCH_DIR, path)
    if args.video:
        args.video = _abs(args.video)
    if args.screenshots:
        args.screenshots = _abs(args.screenshots)
    if args.model and os.path.exists(_abs(args.model)):
        args.model = _abs(args.model)
    os.chdir(ROOT)

    if args.test_mode:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        if args.ui == "classic" and "--bev" not in sys.argv:
            args.bev = "painter"

    if args.screenshots and not args.sim:
        p.error("--screenshots needs --sim")

    if args.sim:
        source = "SIMULATION"
    elif args.camera is not None:
        source = _camera_source(args.camera)
    else:
        source = args.video or DEFAULT_VIDEO
        if not os.path.isfile(source):
            print(f"[ADAS] Video not found: {source}. Use --video, --camera or --sim.", flush=True)

    _configure_graphics()
    app = QApplication(sys.argv)
    app.setApplicationName("ADAS Cluster")

    if args.ui == "classic":
        if args.sim:
            p.error("--sim needs the cluster UI")
        from src.ui.main_window import ADASMainWindow
        window = ADASMainWindow(video_path=source, model_path=args.model, bev_backend=args.bev)
        if args.test_mode:
            QTimer.singleShot(3000, lambda: (print("[ADAS Test] Classic UI OK"), app.quit()))
        window.show()
        sys.exit(app.exec())

    from src.ui.cluster_app import capture_scenario, create_cluster_window

    def make_worker(source_path):
        if source_path == "SIMULATION":
            from src.ui.sim_source import SimulationWorker
            return SimulationWorker()
        from src.ui.worker import InferenceWorker
        return InferenceWorker(video_path=source_path, model_path=args.model, speed_spec=args.speed)

    view, controller = create_cluster_window(
        make_worker, source, units=args.units, audio=not args.mute,
        size=args.size, fullscreen=args.fullscreen,
    )

    if args.screenshots:
        # Drive the simulation frame by frame (no thread) and grab key moments.
        from src.ui.sim_source import SimulationWorker
        sim = SimulationWorker()
        sim.frame_processed.connect(controller.on_frame)
        controller.worker = sim
        controller.audio.set_muted(True)
        capture_scenario(view, controller, sim, SHOT_TIMES, args.screenshots, on_done=app.quit)
        sys.exit(app.exec())

    controller.start()
    app.aboutToQuit.connect(controller.stop)
    if args.test_mode:
        print("[ADAS Test] Cluster UI started", flush=True)
        QTimer.singleShot(4000, lambda: (print("[ADAS Test] Cluster UI OK"), app.quit()))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
