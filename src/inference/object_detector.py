import os

import numpy as np

# COCO vehicle classes used for ADAS
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

_TRACKER_YAML = os.path.join(os.path.dirname(__file__), "bytetrack_lowfps.yaml")

# YOLO-only: hide Garmin HUD + ego hood. Lanes still use the full frame.
YOLO_BOTTOM_DROP_FRAC = 0.12


def _yolo_view(frame, drop_frac=YOLO_BOTTOM_DROP_FRAC):
    """Top crop so YOLO never sees the bonnet. Box xyxy stays in full-frame coords."""
    h, w = frame.shape[:2]
    drop = int(round(h * drop_frac))
    keep_h = max(1, h - drop)
    if keep_h >= h:
        return frame, h, w
    return frame[:keep_h], h, w


def _is_ego_hood_box(bbox, frame_w, frame_h):
    """Wide, short, bottom-glued box = own hood, not traffic."""
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = 0.5 * (x1 + x2)
    glued_bottom = y2 >= 0.88 * frame_h
    starts_low = y1 >= 0.55 * frame_h
    very_wide = bw >= 0.40 * frame_w
    flat = bh <= 0.35 * frame_h
    centered = abs(cx - 0.5 * frame_w) <= 0.35 * frame_w
    return glued_bottom and starts_low and very_wide and flat and centered


def _drop_ego_hood(detections, frame_w, frame_h):
    return [d for d in detections if not _is_ego_hood_box(d["bbox"], frame_w, frame_h)]


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter_area(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _boxes_same_object(a, b, iou_thr=0.45, cover_thr=0.62):
    """True when two boxes are the same vehicle (incl. car-inside-truck)."""
    iou = _iou(a, b)
    if iou >= iou_thr:
        return True
    inter = _inter_area(a, b)
    if inter <= 0:
        return False
    smaller = min(_box_area(a), _box_area(b))
    return smaller > 0 and (inter / smaller) >= cover_thr


_TRUCKISH = frozenset({"truck", "bus", "lorry"})
_CARISH = frozenset({"car", "motorcycle", "bike"})


def _dedupe_vehicle_boxes(detections):
    """One detection per physical vehicle.

    YOLOv8 NMS is class-aware, so the same car often survives as both `car` and
    `truck`. On overlap keep the car unless the truck/bus box is clearly bigger
    and more confident.
    """
    if len(detections) < 2:
        return detections
    n = len(detections)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_same_object(detections[i]["bbox"], detections[j]["bbox"]):
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    kept = []
    for idxs in groups.values():
        if len(idxs) == 1:
            kept.append(detections[idxs[0]])
            continue
        cars = [detections[i] for i in idxs if detections[i].get("class", "car") in _CARISH]
        trucks = [detections[i] for i in idxs if detections[i].get("class", "car") in _TRUCKISH]
        others = [detections[i] for i in idxs]
        best = max(others, key=lambda d: float(d.get("conf", 0.0)))
        if cars and trucks:
            c = max(cars, key=lambda d: float(d.get("conf", 0.0)))
            t = max(trucks, key=lambda d: float(d.get("conf", 0.0)))
            t_area = _box_area(t["bbox"])
            c_area = _box_area(c["bbox"])
            truck_wins = (
                float(t["conf"]) >= float(c["conf"]) + 0.18
                and t_area >= 1.25 * max(c_area, 1.0)
            )
            kept.append(t if truck_wins else c)
        else:
            kept.append(best)
    return kept


class OfflineYOLOVehicleDetector:
    """
    YOLOv8-nano + ByteTrack for vehicles.

    P0: never emit a hard-empty frame on a miss/error — coast last tracks.
    P1: imgsz=640, car/moto/bus/truck, low-FPS ByteTrack, IoU ID recovery.
    YOLO runs on a bottom-cropped view (HUD + ego hood hidden). Lane crop is unchanged.
    """

    def __init__(
        self,
        model_path="models/yolov8n.engine",
        conf_thresh=0.22,
        imgsz=640,
        max_coast_frames=10,
    ):
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.max_coast_frames = max_coast_frames
        self.model = None
        self.model_path = None
        self._last_dets = []
        self._coast_frames = 0
        self._next_fallback_id = 10000

        self._load_model(model_path)

    def _candidate_paths(self, preferred):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        models = os.path.join(root, "models")
        names = []
        if preferred:
            names.append(preferred)
        # Prefer Orin-compiled engine, then PT (Ultralytics export/track), then ONNX
        names.extend(
            [
                os.path.join(models, "yolov8n.engine"),
                os.path.join(models, "yolov8n.pt"),
                os.path.join(models, "yolov8n.onnx"),
            ]
        )
        seen = set()
        out = []
        for p in names:
            ap = os.path.abspath(p)
            if ap in seen or not os.path.isfile(ap):
                continue
            seen.add(ap)
            out.append(ap)
        return out

    def _load_model(self, preferred):
        try:
            from ultralytics import YOLO
        except Exception as e:
            print(f"[ObjectDetector] Ultralytics import failed: {e}")
            return

        last_err = None
        for path in self._candidate_paths(preferred):
            try:
                self.model = YOLO(path, task="detect")
                self.model_path = path
                print(f"[ObjectDetector] Loaded YOLOv8n ByteTrack from {path} (imgsz={self.imgsz})")
                return
            except Exception as e:
                last_err = e
                print(f"[ObjectDetector] Failed {path}: {e}")
        if last_err is not None:
            print(f"[ObjectDetector] No YOLO weights loaded: {last_err}")

    def _parse_results(self, results):
        detections = []
        if results is None or results.boxes is None:
            return detections
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = VEHICLE_CLASSES.get(cls_id, "car")
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1
            detections.append(
                {
                    "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "conf": conf,
                    "class": label,
                    "track_id": track_id,
                }
            )
        return detections

    def _recover_ids(self, detections):
        """Assign previous IDs by IoU when ByteTrack omitted id this frame."""
        if not detections:
            return detections
        prev = [d for d in self._last_dets if d.get("track_id", -1) > 0]
        used = set()
        for det in detections:
            if det["track_id"] > 0:
                used.add(det["track_id"])
                continue
            best_iou, best_id = 0.20, None
            for p in prev:
                pid = p["track_id"]
                if pid in used:
                    continue
                iou = _iou(det["bbox"], p["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, pid
            if best_id is not None:
                det["track_id"] = best_id
                used.add(best_id)
            else:
                # Do not mint a new id: occlusion / ByteTrack miss should
                # coast the old 3D track instead of spawning a ghost car.
                det["track_id"] = -1
        return detections

    def detect(self, frame):
        if self.model is None:
            if self._last_dets and self._coast_frames < self.max_coast_frames:
                self._coast_frames += 1
                return list(self._last_dets)
            return []

        try:
            yolo_frame, full_h, full_w = _yolo_view(frame)
            tracker = _TRACKER_YAML if os.path.isfile(_TRACKER_YAML) else "bytetrack.yaml"
            try:
                results = self.model.track(
                    yolo_frame,
                    persist=True,
                    imgsz=self.imgsz,
                    tracker=tracker,
                    classes=list(VEHICLE_CLASSES.keys()),
                    verbose=False,
                    conf=self.conf_thresh,
                    iou=0.50,
                    agnostic_nms=True,
                )[0]
            except TypeError:
                results = self.model.track(
                    yolo_frame,
                    persist=True,
                    imgsz=self.imgsz,
                    tracker=tracker,
                    classes=list(VEHICLE_CLASSES.keys()),
                    verbose=False,
                    conf=self.conf_thresh,
                    iou=0.50,
                )[0]
            detections = _drop_ego_hood(
                self._recover_ids(
                    _dedupe_vehicle_boxes(self._parse_results(results))
                ),
                full_w,
                full_h,
            )
            if detections:
                self._last_dets = detections
                self._coast_frames = 0
                return detections

            # Empty YOLO output: coast last tracks instead of wiping the scene
            if self._last_dets and self._coast_frames < self.max_coast_frames:
                self._coast_frames += 1
                return list(self._last_dets)
            return []
        except Exception as e:
            print(f"[ObjectDetector] Detection error: {e}")
            if self._last_dets and self._coast_frames < self.max_coast_frames:
                self._coast_frames += 1
                return list(self._last_dets)
            return []
