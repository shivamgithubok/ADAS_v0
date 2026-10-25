"""
PySide6 Async TensorRT ADAS Pipeline Worker Thread
Runs:
 1. 3D Anchor Lane TensorRT Engine
 2. YOLOv8 Vehicle Detector TensorRT Engine
 3. CIPO Tracker (ground-plane range + constant-velocity coast)
"""

import cv2
import numpy as np
import os
import time
from PySide6.QtCore import QThread, Signal
from src.inference.postprocess import postprocess_onnx_output
from src.inference.lane_preprocess import prepare_lane_input
from src.inference.cipo_tracker import CIPOTracker
from src.inference.object_detector import OfflineYOLOVehicleDetector
from src.inference.speed_limit_tracker import SpeedLimitTracker
from src.inference.traffic_sign_detector import (
    TrafficSignDetector,
    draw_isa_overlay,
    filter_sign_dets,
    resolve_model_path,
)
from src.inference.lprnet import LPRNetRecognizer, annotate_speed_dets, default_paths as lpr_paths
from src.inference.ldw_fcw import AdasWarningTracker, draw_adas_alerts
from src.utils.paint_snap import LaneDrawStabilizer
from src.utils.split_visualization import draw_front_view_cipo
from src.utils import lane_debug_overlay
from src.utils.camera_transform import CameraTransform
from src.tracking.road_state import RoadStateEstimator
from src.utils.speed_source import make_speed_source
_LIVE_TRT_RUNTIMES = []


def is_live_source(path) -> bool:
    """camera:N, /dev/videoN or csi[:N] are live cameras; anything else a file."""
    if not path:
        return False
    p = str(path).strip().lower()
    return p.startswith(("camera:", "csi", "/dev/video", "v4l2:", "gst:"))


def csi_pipeline(sensor_id=0, width=1920, height=1080, fps=30, out_w=1280, out_h=720):
    """GStreamer pipeline for a Jetson CSI camera (Argus)."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw,width={out_w},height={out_h},format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1 sync=false"
    )


def open_capture(path):
    """Open a video file or live camera. Returns (cap, is_live)."""
    live = is_live_source(path)
    if not live:
        if path and os.path.exists(path):
            return cv2.VideoCapture(path), False
        return None, False
    p = str(path).strip()
    low = p.lower()
    if low.startswith("gst:"):
        return cv2.VideoCapture(p[4:], cv2.CAP_GSTREAMER), True
    if low.startswith("csi"):
        sid = int(p.split(":", 1)[1]) if ":" in p else 0
        cap = cv2.VideoCapture(csi_pipeline(sid), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print(
                "[ADAS] CSI camera failed to open. OpenCV needs GStreamer support: "
                "use JetPack's python3-opencv (see MANUAL.md, 'Live camera').",
                flush=True,
            )
        return cap, True
    if low.startswith(("camera:", "v4l2:")):
        dev = p.split(":", 1)[1]
    else:
        dev = p
    dev = int(dev) if str(dev).isdigit() else dev
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap, True


def _retain_tensorrt_runtimes():
    """Keep every TensorRT runtime alive for the process.

    ``with trt.Runtime()`` destroys the runtime on exit. On this Jetson the
    next ``createInferRuntime`` then blocks forever, so the video loop never
    starts. Ultralytics and LPRNet both use that pattern.
    """
    import tensorrt as trt

    runtime_cls = trt.Runtime
    if getattr(runtime_cls, "_adas_retain", False):
        return

    def _exit(self, exc_type, exc, tb):
        _LIVE_TRT_RUNTIMES.append(self)
        return False

    runtime_cls.__exit__ = _exit
    runtime_cls._adas_retain = True


from src.utils.calibration import (
    make_P_matrix,
    preset_for_video,
    OPENLANE_CAM_PITCH_DEG,
    OPENLANE_CAM_HEIGHT,
)
from src.utils.ground_calib import GroundCalibration
from src.utils.openlane_view import OpenLaneView

INPUT_H, INPUT_W = 360, 480

def preprocess_frame(frame, lane_view=None):
    """Lane input. `lane_view` is the OpenLane road warp; otherwise sky-crop."""
    if lane_view is not None:
        resized = lane_view.apply(frame)
        img, mask, meta = prepare_lane_input(resized)
        return img, mask, resized, lane_view, meta
    frame_transform = CameraTransform.for_frame(frame, (INPUT_W, INPUT_H))
    resized = frame_transform.apply(frame)
    img, mask, meta = prepare_lane_input(resized)
    return img, mask, resized, frame_transform, meta

class InferenceWorker(QThread):
    # Signal emitted to UI: frame_rgb, proposals, processed_objs, cipo_obj, cipo_status,
    # left_3d, right_3d, avg_fps, latency_ms, speed_mps, source_dt, alerts
    # speed_mps / source_dt drive the lane-anchored BEV: the road scrolls by v*dt.
    frame_processed = Signal(np.ndarray, list, list, object, str, object, object, float, float, object, float, object)
    status_message = Signal(str)

    def __init__(self, video_path=None, model_path="models/anchor3dlane_raw.engine", parent=None,
                 speed_spec="auto"):
        super().__init__(parent)
        self.video_path = video_path
        self.live = is_live_source(video_path)
        self.speed_spec = speed_spec
        self.speed_source = None
        self.engine_path = model_path if model_path.endswith('.engine') else "models/anchor3dlane_raw.engine"
        self.yolo_engine_path = "models/yolov8n.engine"
        self.sign_engine_path = "models/traffic_sign_yolo11n.engine"
        self.lpr_engine_path = "models/us_lprnet_baseline18.engine"
        self.running = False
        self.paused = False
        self.show_lane_draw = False
        # YOLO is created on the worker thread after CUDA is ready (avoids empty/missed frames)
        self.detector = None
        self.sign_detector = None
        self.lpr = None
        self.isa = SpeedLimitTracker(confirm_hits=3, act_conf=0.55)
        self._sign_every = 4
        self._last_sign_dets = []
        self._isa_use_class = True
        self.alerts = AdasWarningTracker()
        # P is LOCKED to OpenLane training extrinsics. Retuning pitch (e.g. Garmin -6°)
        # shears the front corridor vs cyan lanes — model 3D assumes this camera.
        pitch, height = preset_for_video(video_path)
        self.calib_pitch = float(pitch)
        self.calib_height = float(height)
        self.P_matrix = make_P_matrix(OPENLANE_CAM_PITCH_DEG, OPENLANE_CAM_HEIGHT)
        # A live camera has no per-clip calibration JSON: start generic and
        # adapt to the real frame size on the first frame.
        self._calib_base = GroundCalibration.for_video(None if self.live else video_path)
        self.ground_calib = self._calib_base
        self._ui_pitch = None
        self._ui_height = None
        self._lane_view = None
        self.lane_draw = LaneDrawStabilizer()

    def set_calibration(self, pitch_deg, height_m):
        """OpenLane P stays locked. Use set_object_calib for ranging sliders."""
        self.calib_pitch = OPENLANE_CAM_PITCH_DEG
        self.calib_height = OPENLANE_CAM_HEIGHT
        self.P_matrix = make_P_matrix(OPENLANE_CAM_PITCH_DEG, OPENLANE_CAM_HEIGHT)

    def set_object_calib(self, pitch_deg, height_m):
        """Live Pitch/H sliders retune object ranging only (this camera)."""
        self._ui_pitch = float(pitch_deg)
        self._ui_height = float(height_m)
        self._apply_object_calib()

    def _apply_object_calib(self, width=None, height=None):
        gc = self._calib_base
        if gc is None:
            return
        if width and height:
            gc = gc.adapted_to(int(width), int(height))
        if self._ui_pitch is not None and self._ui_height is not None:
            gc = gc.with_pitch_height(self._ui_pitch, self._ui_height)
        self.ground_calib = gc

    def run(self):
        self.running = True
        use_trt = False

        cuda_ctx = None
        trt_engine = None
        trt_context = None

        tracker = None
        detector = None
        road_state_estimator = RoadStateEstimator()
        try:
            self.speed_source = make_speed_source(
                self.speed_spec, None if self.live else self.video_path
            )
        except Exception as se:
            print(f"[ADAS] Speed source unavailable: {se}", flush=True)
            self.status_message.emit(f"Speed source unavailable: {se}")
            self.speed_source = make_speed_source("none")
        speed_src = self.speed_source
        print(f"[ADAS] Speed source: {speed_src.name}", flush=True)

        # Initialize CUDA Context & Triple TensorRT Engines
        if os.path.exists(self.engine_path):
            try:
                import tensorrt as trt
                import pycuda.driver as cuda

                _retain_tensorrt_runtimes()
                cuda.init()
                dev = cuda.Device(0)
                cuda_ctx = dev.make_context()

                TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                
                # 1. Load 3D Lane TensorRT Engine
                self.status_message.emit("Loading 3D Lane TensorRT Engine...")
                with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                    trt_engine = runtime.deserialize_cuda_engine(f.read())
                trt_context = trt_engine.create_execution_context()

                reg_proposals_shape = (1, 4431, 86)
                anchors_shape = (1, 4431, 65)

                dummy_img = np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
                dummy_mask = np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)

                d_img = cuda.mem_alloc(dummy_img.nbytes)
                d_mask = cuda.mem_alloc(dummy_mask.nbytes)

                h_reg_proposals = np.empty(reg_proposals_shape, dtype=np.float32)
                h_anchors = np.empty(anchors_shape, dtype=np.float32)

                d_reg_proposals = cuda.mem_alloc(h_reg_proposals.nbytes)
                d_anchors = cuda.mem_alloc(h_anchors.nbytes)

                stream = cuda.Stream()

                trt_context.set_tensor_address("img", int(d_img))
                trt_context.set_tensor_address("mask", int(d_mask))
                trt_context.set_tensor_address("reg_proposals", int(d_reg_proposals))
                trt_context.set_tensor_address("anchors", int(d_anchors))

                # 2. YOLO nano + signs + LPR. The lane pycuda context must be
                # popped first; a second TensorRT runtime deadlocks if it is current.
                detector = None
                self.lpr = None
                self._isa_use_class = True
                try:
                    cuda_ctx.pop()
                except Exception:
                    pass
                try:
                    print("[ADAS] Loading vehicle detector...", flush=True)
                    yolo_path = self.yolo_engine_path
                    if not os.path.isfile(yolo_path):
                        yolo_path = "models/yolov8n.pt"
                    self.detector = OfflineYOLOVehicleDetector(
                        model_path=yolo_path, conf_thresh=0.22, imgsz=640
                    )
                    detector = self.detector
                except Exception as ye:
                    self.detector = None
                    detector = None
                    self.status_message.emit(f"Vehicle detector warning: {ye}")
                    print(f"[ADAS] Vehicle detector skipped: {ye}", flush=True)
                try:
                    print("[ADAS] Loading traffic-sign detector...", flush=True)
                    sign_path = resolve_model_path(self.sign_engine_path)
                    if sign_path:
                        self.sign_detector = TrafficSignDetector(
                            model_path=sign_path, conf_thresh=0.35
                        )
                        self.status_message.emit(f"Traffic-sign engine: {sign_path}")
                        print(f"[ADAS] Traffic-sign engine ready: {sign_path}", flush=True)
                    else:
                        self.sign_detector = None
                        print("[ADAS] Traffic-sign weights missing", flush=True)
                except Exception as se:
                    self.sign_detector = None
                    self.status_message.emit(f"Traffic-sign warning: {se}")
                    print(f"[ADAS] Traffic-sign skipped: {se}", flush=True)
                finally:
                    try:
                        cuda_ctx.push()
                    except Exception:
                        pass

                lpr_path = self.lpr_engine_path
                if not os.path.isfile(lpr_path):
                    lpr_path = lpr_paths()["engine"]
                if os.path.isfile(lpr_path):
                    print("[ADAS] Loading speed-limit reader...", flush=True)
                    try:
                        self.lpr = LPRNetRecognizer(engine_path=lpr_path)
                        # Keep YOLO class digits as the fallback. LPRNet mph
                        # still wins when OCR returns a posted limit.
                        self._isa_use_class = True
                        self.status_message.emit(f"LPRNet engine: {lpr_path}")
                        print(f"[ADAS] LPRNet ready: {lpr_path}", flush=True)
                    except Exception as le:
                        self.lpr = None
                        self.status_message.emit(f"LPRNet init warning: {le}")
                        print(f"[ADAS] LPRNet skipped: {le}", flush=True)

                # 3. CIPO tracker: OpenLane P for lanes, measured calib for objects
                tracker = CIPOTracker(
                    P_matrix=self.P_matrix,
                    danger_dist=15.0,
                    warning_dist=30.0,
                    ground_calib=self.ground_calib,
                )
                tracker._video_path = self.video_path
                self.status_message.emit(
                    f"Lanes P locked OpenLane; objects "
                    f"f={self.ground_calib.f_px:.0f} h={self.ground_calib.cam_height_m:.2f}m "
                    f"({self.ground_calib.source})"
                )

                use_trt = True
                print("[ADAS] Engines ready, starting video inference", flush=True)
                self.status_message.emit("Engines ready: lanes + YOLO (ground-plane range)")
            except Exception as e:
                self.status_message.emit(f"TensorRT Init Warning: {e}")

        cap = None
        live = self.live
        if self.video_path:
            cap, live = open_capture(self.video_path)
            opened = bool(cap is not None and cap.isOpened())
            kind = "Camera" if live else "Video"
            if cap is None and not live:
                print(f"[ADAS] Video file not found: {self.video_path}", flush=True)
            else:
                print(
                    f"[ADAS] {kind} {'opened' if opened else 'FAILED to open'}: {self.video_path}",
                    flush=True,
                )
            self.status_message.emit(f"{kind}: {os.path.basename(str(self.video_path))}")
        last_wall = None

        fps_history = []
        frame_i = 0
        start_frame = int(os.environ.get("ADAS_START_FRAME", "0") or 0)
        if start_frame > 0 and cap is not None and not live and cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_i = start_frame
            print(f"[ADAS] Seeking to frame {start_frame}", flush=True)
        last_source_ms = None

        try:
            while self.running:
                if self.paused:
                    self.msleep(50)
                    continue

                if live and (cap is None or not cap.isOpened()):
                    # Camera missing or unplugged: retry once a second instead
                    # of flooding the GUI with blank frames.
                    self.msleep(1000)
                    if cap is not None:
                        cap.release()
                    cap, _ = open_capture(self.video_path)
                    if cap is not None and cap.isOpened():
                        print(f"[ADAS] Camera reopened: {self.video_path}", flush=True)
                        last_wall = None
                    continue

                t0 = time.perf_counter()

                frame = None
                frame_transform = None
                source_dt = 1.0 / 30.0
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        if live:
                            # Dropped camera frame: retry rather than rewind.
                            self.msleep(20)
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        last_source_ms = None
                        frame_i = 0
                        self.lane_draw.reset()
                        continue
                    if live:
                        now = time.perf_counter()
                        if last_wall is not None:
                            source_dt = float(np.clip(now - last_wall, 0.01, 0.25))
                        last_wall = now
                    else:
                        source_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        if last_source_ms is not None and source_ms > last_source_ms:
                            source_dt = (source_ms - last_source_ms) / 1000.0
                        last_source_ms = source_ms

                proposals = []
                processed_objs = []
                cipo_obj = None
                cipo_status = "SAFE"
                ego_left, ego_right = None, None
                left_3d, right_3d = None, None
                road_state = None
                # Speed comes from its own source, engines or not.
                # A Garmin overlay is read from this frame when no JSON log exists.
                if frame is not None:
                    speed_src.observe(frame, frame_i)
                speed_mps = speed_src.get_mps(frame_i) if frame is not None else None
                alerts_snap = {
                    "ldw": "OFF",
                    "fcw": "OFF",
                    "priority": "none",
                    "ldw_side": None,
                    "isa": self.isa.snapshot(),
                }

                if frame is not None and use_trt:
                    if frame_i == 0:
                        print("[ADAS] Frame 0: running lane inference...", flush=True)
                    # Step A: 3D Lane TensorRT Inference
                    if self._lane_view is None:
                        fh, fw = frame.shape[:2]
                        base = self._calib_base or self.ground_calib
                        lane_calib = base.adapted_to(fw, fh) if base is not None else None
                        if lane_calib is not None:
                            self._lane_view = OpenLaneView(lane_calib, self.P_matrix)
                            print(
                                f"[ADAS] Lane view: OpenLane road warp "
                                f"({lane_calib.source})",
                                flush=True,
                            )
                    img_tensor, mask_tensor, _, frame_transform, prep_meta = preprocess_frame(
                        frame, self._lane_view
                    )
                    cuda.memcpy_htod_async(d_img, img_tensor, stream)
                    cuda.memcpy_htod_async(d_mask, mask_tensor, stream)

                    trt_context.execute_async_v3(stream.handle)

                    cuda.memcpy_dtoh_async(h_reg_proposals, d_reg_proposals, stream)
                    cuda.memcpy_dtoh_async(h_anchors, d_anchors, stream)
                    stream.synchronize()
                    if frame_i == 0:
                        print("[ADAS] Frame 0: lane inference returned", flush=True)

                    raw_proposals, scores = postprocess_onnx_output(
                        h_reg_proposals, conf_threshold=prep_meta["conf"]
                    )
                    road_state = road_state_estimator.update(
                        raw_proposals, dt=source_dt, speed_mps=speed_mps
                    )
                    # Render immediate measurements while tracks acquire.
                    # CONFIRMED and PREDICTED corridors both feed CIPO.
                    proposals = road_state.visual_lanes
                    safety_lanes = road_state.lanes

                    # Step B: YOLO Vehicle Detection
                    # Pop pycuda context so Ultralytics/TensorRT YOLO can use the GPU (P0)
                    if cuda_ctx is not None:
                        try:
                            cuda_ctx.pop()
                        except Exception:
                            pass
                    ran_sign = False
                    try:
                        raw_detections = detector.detect(frame) if detector else []
                        if self.sign_detector is not None and (frame_i % self._sign_every) == 0:
                            sign_dets, _ = self.sign_detector.detect(frame)
                            sign_dets = filter_sign_dets(sign_dets, frame.shape)
                            self._last_sign_dets = sign_dets
                            ran_sign = True
                    finally:
                        if cuda_ctx is not None:
                            try:
                                cuda_ctx.push()
                            except Exception:
                                pass

                    if self.sign_detector is not None:
                        if ran_sign:
                            if self.lpr is not None and self._last_sign_dets:
                                annotate_speed_dets(frame, self._last_sign_dets, self.lpr)
                            self.isa.update(
                                self._last_sign_dets,
                                ran_infer=True,
                                use_class=self._isa_use_class,
                                frame_shape=frame.shape,
                            )
                        else:
                            self.isa.update(
                                [], ran_infer=False, use_class=self._isa_use_class,
                                frame_shape=frame.shape,
                            )

                    # Step C: CIPO Tracker & 3D In-Path Association
                    h_frame, w_frame = frame.shape[:2]
                    ego_left, ego_right = road_state.ego_left, road_state.ego_right
                    left_3d = road_state.left_corridor_3d
                    right_3d = road_state.right_corridor_3d
                    if tracker is not None:
                        tracker.P = np.asarray(self.P_matrix, dtype=np.float64)
                        self._apply_object_calib(w_frame, h_frame)
                        tracker.ground_calib = self.ground_calib
                    if tracker is not None:
                        processed_objs, cipo_obj = tracker.process_detections(
                            raw_detections,
                            safety_lanes,
                            frame_size=(w_frame, h_frame),
                            ego_left=ego_left,
                            ego_right=ego_right,
                            frame_transform=frame_transform,
                            road_state_confirmed=road_state.is_confirmed,
                            road_status=road_state.status,
                            left_corridor_3d=left_3d,
                            right_corridor_3d=right_3d,
                            dt=source_dt,
                            ego_speed_mps=speed_mps,
                            lane_world=getattr(road_state, "lane_world", None),
                        )
                        cipo_status = tracker.last_cipo_status

                road_status = road_state.status if road_state is not None else "UNKNOWN"
                alerts_snap = self.alerts.update(
                    ego_left,
                    ego_right,
                    road_status,
                    cipo_obj,
                    cipo_status,
                    speed_mps,
                    dt=source_dt,
                    objects=processed_objs,
                )
                alerts_snap["isa"] = self.isa.snapshot()
                alerts_snap["road_status"] = road_status
                alerts_snap["caps"] = {
                    "lanes": bool(use_trt),
                    "vehicles": self.detector is not None,
                    "signs": self.sign_detector is not None,
                }
                alerts_snap["cipo_status"] = cipo_status
                alerts_snap["n_objects"] = len(processed_objs)
                alerts_snap["lane_held"] = bool(
                    road_state is not None and getattr(road_state, "status", "") == "PREDICTED"
                )
                alerts_snap["speed_source"] = speed_src.label
                lane_world = getattr(road_state, "lane_world", None)
                alerts_snap["lane_world"] = lane_world
                if lane_world is not None and lane_world.event:
                    print(f"[LaneWorld] frame {frame_i}: {lane_world.event}")

                # Step D: All rendering reads the same validated temporal road state.
                if frame is None or not use_trt:
                    left_3d, right_3d = None, None

                # Step E: Render Front View Overlay (Lanes + Drivable Corridor + 3D Bboxes)
                if frame is not None:
                    annotated_frame = draw_front_view_cipo(
                        frame,
                        proposals,
                        processed_objs,
                        cipo_obj,
                        np.asarray(self.P_matrix, dtype=np.float64),
                        show_drivable=True,
                        ego_left=ego_left,
                        ego_right=ego_right,
                        frame_transform=frame_transform,
                        road_state_valid=(left_3d is not None and right_3d is not None),
                        left_corridor_3d=left_3d,
                        right_corridor_3d=right_3d,
                        ground_calib=self.ground_calib,
                        lane_stabilizer=self.lane_draw,
                        reset_draw=bool(
                            road_state is not None and getattr(road_state, "reset_draw", False)
                        ),
                        ego_speed_mps=speed_mps,
                        show_lanes=bool(self.show_lane_draw),
                    )
                    if lane_debug_overlay.enabled() and annotated_frame.shape[:2] == frame.shape[:2]:
                        if lane_world is not None and lane_world.event:
                            self._lane_debug_event = f"f{frame_i}: {lane_world.event}"
                        annotated_frame = lane_debug_overlay.draw_lane_world_debug(
                            annotated_frame,
                            lane_world,
                            self.ground_calib,
                            objects=processed_objs,
                            lines=proposals,
                            last_event=getattr(self, "_lane_debug_event", ""),
                        )
                    if self.sign_detector is not None:
                        ego_mph = None
                        if speed_mps is not None:
                            ego_mph = int(round(speed_mps / 0.44704))
                        annotated_frame = draw_isa_overlay(
                            annotated_frame,
                            self.isa.snapshot(),
                            ego_mph=ego_mph,
                            detections=self._last_sign_dets,
                            badges=False,
                        )
                    annotated_frame = draw_adas_alerts(
                        annotated_frame,
                        alerts_snap,
                        ego_left=ego_left,
                        ego_right=ego_right,
                        P_matrix=np.asarray(self.P_matrix, dtype=np.float64),
                        frame_transform=frame_transform,
                        cipo_obj=cipo_obj,
                    )
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    # Downscale for UI transfer/paint (keeps HUD readable, cuts Qt cost)
                    max_w = 800
                    if frame_rgb.shape[1] > max_w:
                        scale = max_w / float(frame_rgb.shape[1])
                        frame_rgb = cv2.resize(
                            frame_rgb,
                            (max_w, int(frame_rgb.shape[0] * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                else:
                    frame_rgb = np.zeros((405, 720, 3), dtype=np.uint8)

                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0
                fps = 1.0 / max(0.001, (t1 - t0))
                fps_history.append(fps)
                avg_fps = float(np.mean(fps_history[-30:]))

                # Emit signal to GUI
                self.frame_processed.emit(
                    frame_rgb, proposals, processed_objs, cipo_obj, cipo_status, left_3d, right_3d,
                    avg_fps, latency_ms, speed_mps, float(source_dt), alerts_snap
                )

                frame_i += 1
                # Files: pace to ~15 FPS when the pipeline is fast enough.
                # Cameras pace themselves; just yield.
                sleep_ms = 1 if live else max(1, int(66 - latency_ms))
                self.msleep(sleep_ms)

        finally:
            if cap:
                cap.release()
            if self.speed_source is not None:
                try:
                    self.speed_source.close()
                except Exception:
                    pass
            
            # Explicitly release CUDA memory buffers & engine contexts before popping context
            try:
                if 'trt_context' in locals() and trt_context is not None:
                    del trt_context
                if 'trt_engine' in locals() and trt_engine is not None:
                    del trt_engine
                if 'stream' in locals() and stream is not None:
                    del stream
                if 'd_img' in locals() and d_img is not None:
                    del d_img
                if 'd_mask' in locals() and d_mask is not None:
                    del d_mask
                if 'd_reg_proposals' in locals() and d_reg_proposals is not None:
                    del d_reg_proposals
                if 'd_anchors' in locals() and d_anchors is not None:
                    del d_anchors
                if hasattr(self, "lpr") and self.lpr is not None:
                    try:
                        self.lpr.close()
                    except Exception:
                        pass
                    self.lpr = None
                if hasattr(self, 'detector') and self.detector is not None:
                    # Clean up the YOLO model and its associated CUDA states if possible
                    if hasattr(self.detector, 'model') and self.detector.model is not None:
                        self.detector.model = None
            except Exception as ce:
                print(f"[Worker Cleanup Warning] {ce}")

            if cuda_ctx:
                try:
                    cuda_ctx.pop()
                except Exception:
                    pass

    def stop(self):
        self.running = False
        self.wait()

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused

    def toggle_lane_draw(self):
        self.show_lane_draw = not bool(self.show_lane_draw)
        return self.show_lane_draw

    def set_lane_draw(self, on):
        self.show_lane_draw = bool(on)
        return self.show_lane_draw
