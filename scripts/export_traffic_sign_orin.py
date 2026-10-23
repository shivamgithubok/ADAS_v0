#!/usr/bin/env python3
"""
Download cvtechniques/TrafficSignDetection (YOLOv11n) and compile TensorRT FP16.

Usage (from repo root, venv active):
    python scripts/export_traffic_sign_orin.py
    python scripts/export_traffic_sign_orin.py --skip-engine
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.traffic_sign_detector import DEFAULT_ENGINE, DEFAULT_ONNX, DEFAULT_PT, HF_REPO, HF_WEIGHTS

PT_PATH = os.path.join(ROOT, DEFAULT_PT)
ONNX_PATH = os.path.join(ROOT, DEFAULT_ONNX)
ENGINE_PATH = os.path.join(ROOT, DEFAULT_ENGINE)


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


def download_weights() -> str:
    os.makedirs(os.path.dirname(PT_PATH), exist_ok=True)
    if os.path.isfile(PT_PATH) and os.path.getsize(PT_PATH) > 1_000_000:
        print(f"[export] Using existing {PT_PATH}")
        return PT_PATH

    from huggingface_hub import hf_hub_download

    print(f"[export] Downloading {HF_REPO}/{HF_WEIGHTS}")
    cached = hf_hub_download(repo_id=HF_REPO, filename=HF_WEIGHTS)
    shutil.copy2(cached, PT_PATH)
    print(f"[export] Wrote {PT_PATH} ({os.path.getsize(PT_PATH)} bytes)")
    return PT_PATH


def export_onnx(imgsz: int) -> str:
    from ultralytics import YOLO

    model = YOLO(PT_PATH)
    print(f"[export] ONNX imgsz={imgsz} from {PT_PATH}")
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=17,
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


def build_engine(onnx_path: str) -> str:
    trtexec = _find_trtexec()
    if trtexec:
        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={ENGINE_PATH}",
            "--fp16",
        ]
        print("[trt]", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)
        print(f"[trt] Wrote {ENGINE_PATH}")
        return ENGINE_PATH

    print("[trt] trtexec not found — Ultralytics TensorRT export")
    from ultralytics import YOLO

    model = YOLO(PT_PATH)
    out = model.export(format="engine", imgsz=640, half=True, device=0)
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
    download_weights()
    export_onnx(args.imgsz)
    if not args.skip_engine:
        build_engine(ONNX_PATH)
    print("[done] traffic-sign YOLOv11n ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
