"""NVIDIA TAO US LPRNet (CTC) — read digits from a YOLO speed-sign crop.

Off-the-shelf weights are license-plate OCR. Feed the lower digit band of a
MUTCD SPEED LIMIT plate, not the full “SPEED LIMIT” legend.
"""

from __future__ import annotations

import os
import re
import time

import cv2
import numpy as np

DEFAULT_ONNX = "models/us_lprnet_baseline18.onnx"
DEFAULT_ENGINE = "models/us_lprnet_baseline18.engine"
DEFAULT_DICT = "models/lprnet_dict_us.txt"

# NGC US dict: 0-9 then A-Z without O. Blank index = len(charset).
US_CHARSET = list("0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ")
BLANK_IDX = len(US_CHARSET)  # 35
IN_H, IN_W = 48, 96
SEQ_LEN = IN_W // 4  # 24

# Posted US limits we will accept from a 2-digit read.
US_SPEED_MPH = frozenset({15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75})

CROP_MODES = ("digits", "digits_wide", "full")


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_paths():
    root = repo_root()
    return {
        "onnx": os.path.join(root, DEFAULT_ONNX),
        "engine": os.path.join(root, DEFAULT_ENGINE),
        "dict": os.path.join(root, DEFAULT_DICT),
    }


def load_charset(path=None):
    if path and os.path.isfile(path):
        chars = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    chars.append(s[0])
        if chars:
            return chars
    return list(US_CHARSET)


def crop_sign_band(frame_bgr, bbox, mode="digits"):
    """Crop a speed-sign box. `digits` keeps the lower number band."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    bw, bh = x2 - x1, y2 - y1

    if mode == "full":
        return frame_bgr[y1:y2, x1:x2].copy()

    if mode == "digits_wide":
        top, bot, side = 0.40, 0.02, 0.04
    else:
        # Default: skip “SPEED LIMIT” text, keep the two digits.
        top, bot, side = 0.48, 0.04, 0.08

    y1b = y1 + int(bh * top)
    y2b = y2 - int(bh * bot)
    x1b = x1 + int(bw * side)
    x2b = x2 - int(bw * side)
    if y2b - y1b < 8 or x2b - x1b < 8:
        return frame_bgr[y1:y2, x1:x2].copy()
    return frame_bgr[y1b:y2b, x1b:x2b].copy()


def letterbox_48x96(crop_bgr, pad_value=255):
    """Keep aspect ratio on the 48×96 plate canvas. Stretching a square
    speed-sign crop to 2:1 is what makes CTC drop the second digit."""
    h, w = crop_bgr.shape[:2]
    scale = min(IN_W / float(max(w, 1)), IN_H / float(max(h, 1)))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((IN_H, IN_W, 3), int(pad_value), dtype=np.uint8)
    y0, x0 = (IN_H - nh) // 2, (IN_W - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def preprocess_bgr(crop_bgr, letterbox=True, pad_value=255):
    """RGB, 48×96, /255, NCHW float32 — DeepStream net-scale-factor=1/255."""
    if crop_bgr is None or crop_bgr.size == 0:
        crop_bgr = np.full((IN_H, IN_W, 3), 255, dtype=np.uint8)
    if letterbox:
        rgb = cv2.cvtColor(letterbox_48x96(crop_bgr, pad_value=pad_value), cv2.COLOR_BGR2RGB)
    else:
        resized = cv2.resize(crop_bgr, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(chw)


def ctc_greedy_decode(ids, probs=None, charset=None):
    """Collapse repeats and drop blank. Matches DeepStream LPR greedy CTC."""
    charset = charset or US_CHARSET
    blank = len(charset)
    ids = np.asarray(ids).reshape(-1)
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    chars = []
    kept_p = []
    prev = -1
    for t, idx in enumerate(ids):
        idx = int(idx)
        if idx != blank and idx != prev:
            if 0 <= idx < len(charset):
                chars.append(charset[idx])
                if probs is not None and t < probs.size:
                    kept_p.append(float(probs[t]))
        prev = idx
    text = "".join(chars)
    conf = float(np.mean(kept_p)) if kept_p else 0.0
    return text, conf


def parse_speed_mph(text):
    """Pull a 2-digit US posted limit from an LPRNet string."""
    digits = re.sub(r"\D+", "", text or "")
    if len(digits) >= 2:
        for i in range(len(digits) - 1):
            mph = int(digits[i : i + 2])
            if mph in US_SPEED_MPH:
                return mph
    return None


def _np_dtype(trt_dtype, trt_mod):
    mapping = {
        trt_mod.DataType.FLOAT: np.float32,
        trt_mod.DataType.HALF: np.float16,
        trt_mod.DataType.INT32: np.int32,
        trt_mod.DataType.INT8: np.int8,
        trt_mod.DataType.BOOL: np.bool_,
    }
    if hasattr(trt_mod.DataType, "INT64"):
        mapping[trt_mod.DataType.INT64] = np.int64
    return mapping.get(trt_dtype, np.float32)


def is_speed_limit_class(name):
    return bool(name) and str(name).lower().startswith("speedlimit")


def annotate_speed_dets(frame_bgr, detections, recognizer, mode="full"):
    """OCR the YOLO box. Keep YOLO class only if CTC does not yield a limit."""
    t0 = time.perf_counter()
    n = 0
    for det in detections:
        if not is_speed_limit_class(det.get("class", "")):
            continue
        out = recognizer.read_bbox(frame_bgr, det["bbox"], mode=mode)
        n += 1
        det["yolo_class"] = det.get("class")
        det["yolo_conf"] = det.get("conf")
        det["lpr_text"] = out["text"]
        det["lpr_conf"] = out["conf"]
        det["mph"] = out["mph"]
        if out["mph"] is not None:
            det["class"] = f"speedLimit{out['mph']}"
            det["conf"] = float(out["conf"])
    return detections, (time.perf_counter() - t0) * 1000.0, n


class LPRNetRecognizer:
    """TensorRT runner for NVIDIA US LPRNet (batch 1, FP16 engine).

    Uses the current pycuda context (ADAS worker) or creates one if standalone.
    Do not import pycuda.autoinit here — that would steal the lane-engine context.
    """

    def __init__(self, engine_path=None, dict_path=None):
        import tensorrt as trt
        import pycuda.driver as cuda

        paths = default_paths()
        engine_path = engine_path or paths["engine"]
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(
                f"LPRNet engine missing: {engine_path}. Run scripts/export_lprnet_orin.py"
            )
        self.charset = load_charset(dict_path or paths["dict"])
        self.blank = len(self.charset)
        self.engine_path = engine_path
        self._own_ctx = None
        try:
            cuda.init()
        except Exception:
            pass
        current = None
        try:
            current = cuda.Context.get_current()
        except Exception:
            current = None
        if current is None:
            self._own_ctx = cuda.Device(0).make_context()

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self._cuda = cuda
        self._trt = trt

        self.in_name = "image_input"
        self.argmax_name = "tf_op_layer_ArgMax"
        self.max_name = "tf_op_layer_Max"
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        for n in names:
            if "argmax" in n.lower() or n.endswith("ArgMax"):
                self.argmax_name = n
            elif n.lower().endswith("max") and "argmax" not in n.lower():
                self.max_name = n
            elif "image" in n.lower() or "input" in n.lower():
                self.in_name = n

        self.context.set_input_shape(self.in_name, (1, 3, IN_H, IN_W))
        in_shape = tuple(int(x) for x in self.context.get_tensor_shape(self.in_name))
        arg_shape = tuple(int(x) for x in self.context.get_tensor_shape(self.argmax_name))
        max_shape = tuple(int(x) for x in self.context.get_tensor_shape(self.max_name))

        self.h_input = np.empty(in_shape, dtype=np.float32)
        self.h_argmax = np.empty(arg_shape, dtype=_np_dtype(self.engine.get_tensor_dtype(self.argmax_name), trt))
        self.h_max = np.empty(max_shape, dtype=_np_dtype(self.engine.get_tensor_dtype(self.max_name), trt))
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_argmax = cuda.mem_alloc(self.h_argmax.nbytes)
        self.d_max = cuda.mem_alloc(self.h_max.nbytes)
        self.context.set_tensor_address(self.in_name, int(self.d_input))
        self.context.set_tensor_address(self.argmax_name, int(self.d_argmax))
        self.context.set_tensor_address(self.max_name, int(self.d_max))

    def close(self):
        if self._own_ctx is not None:
            try:
                self._own_ctx.pop()
            except Exception:
                pass
            self._own_ctx = None

    def infer_tensor(self, tensor):
        cuda = self._cuda
        np.copyto(self.h_input, tensor)
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_argmax, self.d_argmax, self.stream)
        cuda.memcpy_dtoh_async(self.h_max, self.d_max, self.stream)
        self.stream.synchronize()
        text, conf = ctc_greedy_decode(self.h_argmax, self.h_max, self.charset)
        return text, conf

    def read_crop(self, crop_bgr):
        tensor = preprocess_bgr(crop_bgr)
        text, conf = self.infer_tensor(tensor)
        return {
            "text": text,
            "conf": conf,
            "mph": parse_speed_mph(text),
        }

    def read_bbox(self, frame_bgr, bbox, mode="full"):
        crop = crop_sign_band(frame_bgr, bbox, mode=mode)
        out = self.read_crop(crop)
        out["crop"] = crop
        out["mode"] = mode
        return out

    def benchmark(self, crop_bgr, warmup=20, iters=100):
        tensor = preprocess_bgr(crop_bgr)
        for _ in range(warmup):
            self.infer_tensor(tensor)
        gpu = []
        wall = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = preprocess_bgr(crop_bgr)
            t1 = time.perf_counter()
            self.infer_tensor(tensor)
            t2 = time.perf_counter()
            gpu.append((t2 - t1) * 1000.0)
            wall.append((t2 - t0) * 1000.0)
        gpu.sort()
        wall.sort()
        def _summ(xs):
            return {
                "mean_ms": float(np.mean(xs)),
                "p50_ms": float(xs[len(xs) // 2]),
                "p95_ms": float(xs[int(len(xs) * 0.95)]),
                "min_ms": float(xs[0]),
                "max_ms": float(xs[-1]),
            }
        return {
            "warmup": warmup,
            "iters": iters,
            "infer": _summ(gpu),
            "preprocess_plus_infer": _summ(wall),
            "qps": 1000.0 / max(float(np.mean(gpu)), 1e-6),
        }
