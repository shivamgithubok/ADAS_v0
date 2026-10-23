#!/usr/bin/env python3
"""
Download NVIDIA TAO US LPRNet ONNX and compile TensorRT FP16 (batch 1).

Usage (from repo root, venv active):
    python scripts/export_lprnet_orin.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.lprnet import DEFAULT_ENGINE, DEFAULT_ONNX

ONNX_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/tao/lprnet/versions/"
    "deployable_onnx_v1.1/files/us_lprnet_baseline18_deployable.onnx"
)
ONNX_PATH = os.path.join(ROOT, DEFAULT_ONNX)
ENGINE_PATH = os.path.join(ROOT, DEFAULT_ENGINE)


def _find_trtexec():
    found = shutil.which("trtexec")
    if found:
        return found
    for p in ("/usr/src/tensorrt/bin/trtexec", "/usr/lib/aarch64-linux-gnu/bin/trtexec"):
        if os.path.isfile(p):
            return p
    return None


def download_onnx():
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
    if os.path.isfile(ONNX_PATH) and os.path.getsize(ONNX_PATH) > 1_000_000:
        print(f"[export] Using existing {ONNX_PATH}")
        return ONNX_PATH
    print(f"[export] Downloading {ONNX_URL}")
    urllib.request.urlretrieve(ONNX_URL, ONNX_PATH)
    print(f"[export] Wrote {ONNX_PATH} ({os.path.getsize(ONNX_PATH)} bytes)")
    return ONNX_PATH


def build_engine(onnx_path):
    trtexec = _find_trtexec()
    if not trtexec:
        raise FileNotFoundError("trtexec not found")
    shape = "image_input:1x3x48x96"
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={ENGINE_PATH}",
        "--fp16",
        f"--minShapes={shape}",
        f"--optShapes={shape}",
        f"--maxShapes={shape}",
    ]
    print("[trt]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    print(f"[trt] Wrote {ENGINE_PATH} ({os.path.getsize(ENGINE_PATH)} bytes)")
    return ENGINE_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-engine", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    download_onnx()
    if not args.skip_engine:
        build_engine(ONNX_PATH)
    print("[done] US LPRNet ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
