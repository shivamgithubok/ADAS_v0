#!/usr/bin/env python3
"""
Export YOLOv8-nano and compile a TensorRT FP16 engine for this Jetson Orin.

Usage (from repo root, venv active):
    python scripts/export_yolo_orin.py
    python scripts/export_yolo_orin.py --imgsz 640
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PT_PATH = os.path.join(ROOT, "models", "yolov8n.pt")
ONNX_PATH = os.path.join(ROOT, "models", "yolov8n.onnx")
ENGINE_PATH = os.path.join(ROOT, "models", "yolov8n.engine")


def _find_trtexec():
    found = shutil.which("trtexec")
    if found:
        return found
    for p in (
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/lib/aarch64-linux-gnu/bin/trtexec",
    ):
        if os.path.isfile(p):
            return p
    return None


def export_onnx(imgsz: int) -> str:
    from ultralytics import YOLO

    if not os.path.isfile(PT_PATH):
        print(f"[export] Downloading yolov8n.pt into {PT_PATH}")
        model = YOLO("yolov8n.pt")
        os.makedirs(os.path.dirname(PT_PATH), exist_ok=True)
        # Ultralytics caches weights; copy if needed
        if os.path.isfile("yolov8n.pt") and not os.path.isfile(PT_PATH):
            shutil.copy2("yolov8n.pt", PT_PATH)
        model = YOLO(PT_PATH if os.path.isfile(PT_PATH) else "yolov8n.pt")
    else:
        model = YOLO(PT_PATH)

    print(f"[export] ONNX imgsz={imgsz} from {PT_PATH}")
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=12,
        simplify=True,
        dynamic=False,
        half=False,
    )
    out = os.path.abspath(str(out))
    if out != ONNX_PATH:
        os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
        shutil.copy2(out, ONNX_PATH)
        print(f"[export] Copied ONNX → {ONNX_PATH}")
    return ONNX_PATH


def build_engine(onnx_path: str, imgsz: int) -> str:
    trtexec = _find_trtexec()
    if trtexec:
        backup = ENGINE_PATH + ".bak"
        if os.path.isfile(ENGINE_PATH) and os.path.getsize(ENGINE_PATH) > 0:
            shutil.copy2(ENGINE_PATH, backup)
            print(f"[trt] Backed up old engine → {backup}", flush=True)
        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={ENGINE_PATH}",
            "--fp16",
        ]
        print("[trt]", " ".join(cmd))
        subprocess.check_call(cmd)
        print(f"[trt] Wrote {ENGINE_PATH}")
        return ENGINE_PATH

    print("[trt] trtexec not found — Ultralytics TensorRT export")
    from ultralytics import YOLO

    model = YOLO(PT_PATH if os.path.isfile(PT_PATH) else onnx_path)
    out = model.export(format="engine", imgsz=imgsz, half=True, device=0)
    out = os.path.abspath(str(out))
    if out != ENGINE_PATH:
        shutil.copy2(out, ENGINE_PATH)
    print(f"[trt] Wrote {ENGINE_PATH}")
    return ENGINE_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--skip-engine", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    onnx_path = export_onnx(args.imgsz)
    if not args.skip_engine:
        build_engine(onnx_path, args.imgsz)
    print("[done] YOLOv8n nano ready for this Orin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
