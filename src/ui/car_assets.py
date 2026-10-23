"""GLB car assets for the Qt Quick 3D BEV (transforms in meters, Y-up).

World: +X right, +Y up, -Z forward (down the road).
Files live in data/3d_objects/ (renamed: audi.glb, skoda_fabia.glb, tesla.glb, …).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_OBJ_DIR = os.path.join(_ROOT, "data", "3d_objects")

KIND_SKODA = 0
KIND_TESLA = 1
KIND_SHC = 2
KIND_DODGE = 3


@dataclass(frozen=True)
class CarAsset:
    name: str
    filename: str
    scale: float = 1.0
    rot_x: float = 0.0
    rot_y: float = 180.0
    rot_z: float = 0.0
    y: float = 0.0
    directory: str = ""

    @property
    def path(self) -> str:
        root = self.directory or _OBJ_DIR
        return os.path.join(root, self.filename)

    def exists(self) -> bool:
        return os.path.isfile(self.path)


# Ego — Sketchfab FBX scale 0.01
EGO_AUDI = CarAsset(
    name="audi",
    filename="audi.glb",
    scale=100.0,
    rot_y=180.0,
    y=0.0,
)

TRAFFIC_SKODA = CarAsset(
    name="skoda_fabia",
    filename="skoda_fabia.glb",
    scale=100.0,
    rot_y=180.0,
    y=0.0,
)

TRAFFIC_TESLA = CarAsset(
    name="tesla",
    filename="tesla.glb",
    # Nested Sketchfab X-90 — applied on teslaFix in QML, not composed with yaw.
    # Outer node uses the same yaw as ego (180). Scale/Y come from bounds fit.
    scale=1.0,
    rot_x=-90.0,
    rot_y=0.0,
    y=0.0,
)

TRAFFIC_SHC = CarAsset(
    name="shc_mc",
    filename="shc_mc.glb",
    # Mesh is X-forward; +90 yaw lines it up with Skoda/Audi (-Z forward).
    scale=0.01,
    rot_y=90.0,
    y=0.0,
)

TRAFFIC_DODGE = CarAsset(
    name="truck",
    filename="truck.glb",
    directory=os.path.join(_ROOT, "data", "3d_truck"),
    scale=1.0,
    rot_y=180.0,
    y=0.0,
)

# Exact YOLO/COCO names. Do not substring-match ("van" is not a class here).
TRUCK_LABELS = ("truck", "bus", "lorry")

KIND_ASSETS = {
    KIND_SKODA: TRAFFIC_SKODA,
    KIND_TESLA: TRAFFIC_TESLA,
    KIND_SHC: TRAFFIC_SHC,
    KIND_DODGE: TRAFFIC_DODGE,
}

# Max simultaneous instances per mesh (Tesla is ~684k tris — one copy only).
# SHC native units are meters; the old 0.01 scale made in-path cars ~5 cm tall.
KIND_MAX = {
    KIND_SKODA: 8,
    KIND_TESLA: 0,  # skipped until GLB rest pose is fixed
    KIND_SHC: 0,    # skipped: 0.01 scale rendered a speck at the CIPO marker
    KIND_DODGE: 2,
}


def asset_url(asset: CarAsset) -> str:
    from PySide6.QtCore import QUrl
    return QUrl.fromLocalFile(os.path.abspath(asset.path)).toString()
