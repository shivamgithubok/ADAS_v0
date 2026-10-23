"""Standalone LISA traffic-sign detector (YOLOv11n, not wired into CIPO)."""

from __future__ import annotations

import os

import cv2
import numpy as np

# Hugging Face cvtechniques/TrafficSignDetection class order (data.yaml).
# Order from the published YOLO11n checkpoint (13 classes, includes 55).
TRAFFIC_SIGN_CLASSES = [
    "doNotEnter",
    "pedestrianCrossing",
    "speedLimit15",
    "speedLimit25",
    "speedLimit30",
    "speedLimit35",
    "speedLimit40",
    "speedLimit45",
    "speedLimit50",
    "speedLimit55",
    "speedLimit65",
    "stop",
    "yield",
]

# BGR colors for overlay (one per class).
CLASS_COLORS = {
    "doNotEnter": (40, 40, 220),
    "pedestrianCrossing": (220, 160, 40),
    "speedLimit15": (80, 200, 255),
    "speedLimit25": (40, 180, 255),
    "speedLimit30": (0, 200, 220),
    "speedLimit35": (0, 165, 255),
    "speedLimit40": (0, 140, 255),
    "speedLimit45": (0, 110, 230),
    "speedLimit50": (20, 90, 210),
    "speedLimit55": (30, 75, 205),
    "speedLimit65": (40, 60, 200),
    "stop": (0, 0, 220),
    "yield": (0, 215, 255),
}

HF_REPO = "cvtechniques/TrafficSignDetection"
HF_WEIGHTS = "best.pt"
DEFAULT_PT = "models/traffic_sign_yolo11n.pt"
DEFAULT_ONNX = "models/traffic_sign_yolo11n.onnx"
DEFAULT_ENGINE = "models/traffic_sign_yolo11n.engine"


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_weight_paths():
    root = repo_root()
    return {
        "pt": os.path.join(root, DEFAULT_PT),
        "onnx": os.path.join(root, DEFAULT_ONNX),
        "engine": os.path.join(root, DEFAULT_ENGINE),
    }


def resolve_model_path(preferred=None):
    """Prefer TensorRT engine, then PT, then ONNX."""
    paths = default_weight_paths()
    candidates = []
    if preferred:
        candidates.append(os.path.abspath(preferred))
    candidates.extend([paths["engine"], paths["pt"], paths["onnx"]])
    seen = set()
    for path in candidates:
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        return path
    return None


class TrafficSignDetector:
    """Ultralytics YOLO wrapper for the 12-class LISA traffic-sign model."""

    def __init__(self, model_path=None, conf_thresh=0.35, iou_thresh=0.45, imgsz=640):
        from ultralytics import YOLO

        path = resolve_model_path(model_path)
        if path is None:
            raise FileNotFoundError(
                "No traffic-sign weights found. Run scripts/export_traffic_sign_orin.py first."
            )
        self.model_path = path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.model = YOLO(path, task="detect")
        raw_names = dict(getattr(self.model, "names", {}) or {})
        # TensorRT engines often ship generic class0..classN — use the LISA labels.
        if (
            not raw_names
            or all(str(v).startswith("class") for v in raw_names.values())
            or set(raw_names.values()) != set(TRAFFIC_SIGN_CLASSES)
        ):
            self.names = {i: n for i, n in enumerate(TRAFFIC_SIGN_CLASSES)}
        else:
            self.names = raw_names

    def detect(self, frame_bgr):
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )
        return self._parse(results[0] if results else None), (
            results[0].speed if results else {}
        )

    def _parse(self, result):
        detections = []
        if result is None or result.boxes is None:
            return detections
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            name = self.names.get(cls_id, TRAFFIC_SIGN_CLASSES[cls_id] if cls_id < 12 else str(cls_id))
            detections.append(
                {
                    "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "conf": float(box.conf[0]),
                    "class": name,
                    "class_id": cls_id,
                }
            )
        return detections


def filter_sign_dets(detections, frame_shape, hud_frac=0.10, min_h_frac=0.028):
    """Drop HUD / hood / tiny boxes. Infer is full-frame; this is post only."""
    h, w = frame_shape[:2]
    y_cut = h * (1.0 - float(hud_frac))
    min_h = h * float(min_h_frac)
    kept = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if y2 >= y_cut:
            continue
        if (y2 - y1) < min_h:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        kept.append(det)
    return kept


def draw_isa_overlay(frame_bgr, isa, ego_mph=None, detections=None, badges=True):
    """Sign boxes, plus optional LIMIT / ego speed badges."""
    out = draw_detections(frame_bgr, detections or [])
    if not badges:
        return out
    h, w = out.shape[:2]
    posted = isa.get("posted_mph") if isa else None
    status = (isa or {}).get("status", "NONE")
    cand = (isa or {}).get("candidate_mph")
    if posted is not None:
        text = f"LIMIT {posted}"
        color = (40, 200, 80) if status == "CONFIRMED" else (40, 180, 220)
    elif cand is not None:
        text = f"CAND {cand}"
        color = (40, 180, 220)
    else:
        text = "LIMIT --"
        color = (90, 90, 90)
    if status == "STALE" and posted is not None:
        text = f"LIMIT {posted}?"
        color = (80, 160, 220)

    cv2.rectangle(out, (12, 12), (220, 58), (16, 16, 20), -1)
    cv2.rectangle(out, (12, 12), (220, 58), color, 2)
    cv2.putText(out, text, (22, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

    if ego_mph is not None:
        ego = f"EGO {int(ego_mph)} MPH"
        cv2.rectangle(out, (w - 200, 12), (w - 12, 58), (16, 16, 20), -1)
        cv2.putText(out, ego, (w - 188, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
    return out


def draw_detections(frame_bgr, detections):
    out = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["class"]
        color = CLASS_COLORS.get(label, (0, 220, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        mph = det.get("mph")
        lpr = det.get("lpr_text")
        if mph is not None:
            text = f"{mph}"
            if lpr and lpr != str(mph):
                text += f" ({lpr})"
            text += f" {det['conf']:.2f}"
        else:
            text = f"{label} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1 - 6)
        cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 2), color, -1)
        cv2.putText(
            out,
            text,
            (x1 + 2, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def letterbox_stack(images, cols=3, cell=420, pad=8):
    """Build a simple contact sheet from BGR images."""
    if not images:
        return np.zeros((cell, cell, 3), dtype=np.uint8)
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows * cell + (rows + 1) * pad, cols * cell + (cols + 1) * pad, 3), 24, np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        h, w = img.shape[:2]
        scale = min(cell / float(h), cell / float(w))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((cell, cell, 3), 18, np.uint8)
        y0, x0 = (cell - nh) // 2, (cell - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        y = pad + r * (cell + pad)
        x = pad + c * (cell + pad)
        sheet[y : y + cell, x : x + cell] = canvas
    return sheet
