"""Metric ground-plane ranging for the real (Garmin) camera.

Why this exists
---------------
`src/utils/calibration.py` describes the *OpenLane training* camera: fx=fy=2015
on a 1920x1280 sensor, pitch -3 deg, height 1.5 m.  That P matrix is correct for
feeding Anchor3DLane and for projecting its 3D lane output back onto the model
image, and it must stay locked for that job.

It is *not* correct for ranging objects in a Garmin clip.  Measured on
GRMN6694_540_nohud.mp4 (960x502) by scripts/calibrate_from_video.py:

    quantity                 OpenLane P implies      measured on the clip
    horizon row v_vp         302.8                   316.0
    optical centre u_vp      480.0                   527.6
    effective cam height     0.9422 m                1.345 m
    h * f (range scale)      950.5 px*m              873 px*m  (95% CI 861-885)
    focal length             1007.5 / 632.9 px       649 px (square)

The two focal lengths in that last row are not a typo: `homography_crop_resize`
scales x by 480/1920 and y by 360/1280, so P carries a 1.125x anisotropy that a
real square-pixel sensor does not have.

Consequences of using P for ranging, and the reason this module exists:
  * the 13.2 px horizon error compresses range without bound.  Reported range
    saturates at h*f/(v_vp_true - v_vp_P) = 72 m, so a car at 91 m reads 42 m
    and a group at 28/32/45/91 m reads 21/24/29/42 m.
  * anything whose base sits above row 302.8 (overpasses, signs, gantries)
    yields a negative range that the old code clamped to 1.0 m, which is why a
    truck on an overpass was labelled "1.0 m".
  * lateral offset is h*(u - u_vp)/(v - v_vp), so the 0.70x height error and the
    47.6 px optical-centre error together place a car centred in the adjacent
    lane at -0.7 m at 30 m, and at -0.01 m at 50 m: dead on the ego centreline.

Note that lateral offset contains no focal length at all.  Only forward range
depends on f, and only through the product h*f.  That is why the lane-width and
vanishing-point measurements (which need no f) are trustworthy on their own, and
why an f error cannot cause lateral collapse -- a point worth remembering when
reading the old `project_2d_to_3d_ground`.

Frame convention (matches drivable_area.py and the Anchor3DLane output)
----------------------------------------------------------------------
    x  lateral, +right, 0 = camera optical axis projected onto the road
    y  forward along the optical axis, 0 = camera
    z  up, 0 = road surface

Origin is the *camera*, not the rear axle.  `lateral_mount_offset_m` shifts x to
the vehicle centreline when you have measured it; `longitudinal_offset_m` is
recorded for downstream consumers but deliberately not applied here, because the
lane geometry this is compared against is also camera-origin.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

# Bbox bottom edges land within a pixel or two of the true tyre contact line;
# 1.5 px is the 1-sigma we measured from YOLO box jitter on parked cars.
DEFAULT_SIGMA_V_PX = 1.5
DEFAULT_SIGMA_U_PX = 2.0

# A ray this close to the horizon has unusable range (see range_sigma_m), and
# below it the intersection is behind the camera.
MIN_BELOW_HORIZON_PX = 8.0

# Beyond this, one pixel of jitter moves the estimate by >8 m. Report nothing.
DEFAULT_MAX_RANGE_M = 110.0

_CALIB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "calib",
)


@dataclass(frozen=True)
class GroundCalibration:
    """Square-pixel pinhole + flat-road geometry for one camera/clip.

    The vanishing point (u_vp, v_vp) is stored instead of (cx, cy, pitch, yaw)
    because that is what the estimator actually measures and what the ranging
    equations actually use:  v_vp = cy + f*tan(pitch), u_vp = cx + f*tan(yaw).
    Carrying the VP directly avoids having to split it into a principal point
    and a mounting angle, a split that is unobservable from road geometry alone.
    """

    source_width: int
    source_height: int
    f_px: float
    u_vp: float
    v_vp: float
    cam_height_m: float
    lateral_mount_offset_m: float = 0.0
    longitudinal_offset_m: float = 0.0
    roll_rad: float = 0.0
    max_range_m: float = DEFAULT_MAX_RANGE_M
    sigma_v_px: float = DEFAULT_SIGMA_V_PX
    sigma_u_px: float = DEFAULT_SIGMA_U_PX
    source: str = "measured"

    # ------------------------------------------------------------- derived
    @property
    def hf(self) -> float:
        """Range scale h*f in px*m. The single number absolute range needs."""
        return float(self.cam_height_m) * float(self.f_px)

    @property
    def pitch_rad(self) -> float:
        """Downward pitch implied by the horizon, assuming cy = height/2."""
        return math.atan2(self.v_vp - 0.5 * self.source_height, self.f_px)

    @property
    def yaw_rad(self) -> float:
        """Camera yaw implied by the VP, assuming cx = width/2."""
        return math.atan2(self.u_vp - 0.5 * self.source_width, self.f_px)

    @property
    def hfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan(0.5 * self.source_width / self.f_px))

    @property
    def K(self) -> np.ndarray:
        """Square-pixel intrinsics with the principal point at the VP.

        Putting the principal point at the VP folds pitch and yaw into K, which
        makes the camera->road rotation identity and the back-projection below a
        two-line expression. Use `K_centred` if you need true optical intrinsics.
        """
        return np.array(
            [[self.f_px, 0.0, self.u_vp],
             [0.0, self.f_px, self.v_vp],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def K_centred(self) -> np.ndarray:
        """Intrinsics with the principal point at the image centre."""
        return np.array(
            [[self.f_px, 0.0, 0.5 * self.source_width],
             [0.0, self.f_px, 0.5 * self.source_height],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    # ------------------------------------------------------- back-projection
    def backproject_ground(
        self, u: float, v: float
    ) -> Optional[Tuple[float, float]]:
        """Intersect the ray through source pixel (u, v) with the road z=0.

        Returns (x_m, y_m) in the vehicle frame, or None when the measurement is
        invalid: at/above the horizon, or beyond `max_range_m`.

        This is exactly `K^-1 [u,v,1]` rotated into the road frame and scaled
        until z=0; because K puts the principal point on the VP, that rotation
        is the identity and the whole thing collapses to:

            y = h*f / (v - v_vp)          # forward range
            x = h  * (u - u_vp) / (v - v_vp)

        `backproject_ground_explicit` shows the long-hand matrix form and is
        unit-tested against this one.
        """
        dv = float(v) - self.v_vp
        if dv < MIN_BELOW_HORIZON_PX:
            return None
        y = self.hf / dv
        if not np.isfinite(y) or y <= 0.0 or y > self.max_range_m:
            return None
        x = self.cam_height_m * (float(u) - self.u_vp) / dv
        if self.roll_rad:
            # Small-angle roll tilts the horizon; correct x by the row offset.
            x -= self.cam_height_m * math.tan(self.roll_rad)
        return float(x - self.lateral_mount_offset_m), float(y)

    def backproject_ground_explicit(
        self, u: float, v: float
    ) -> Optional[Tuple[float, float]]:
        """Long-hand K^-1 / rotate / intersect form. Same result, slower.

        Kept because it is the version that is easy to check by eye and easy to
        extend if the road ever stops being a plane through the camera nadir.
        """
        dv = float(v) - self.v_vp
        if dv < MIN_BELOW_HORIZON_PX:
            return None
        # Ray in camera coords, optical axis +z_cam, y_cam down.
        d_cam = np.linalg.inv(self.K) @ np.array([float(u), float(v), 1.0])
        # Camera -> road frame. With the VP as principal point the camera's
        # optical axis is parallel to the road, so this is a pure axis relabel:
        #   x_road = x_cam,  y_road = z_cam (forward),  z_road = -y_cam (up)
        d_road = np.array([d_cam[0], d_cam[2], -d_cam[1]], dtype=np.float64)
        if d_road[2] >= -1e-9:                     # not pointing down
            return None
        t = self.cam_height_m / (-d_road[2])       # camera at z = +h
        x, y = t * d_road[0], t * d_road[1]
        if y <= 0.0 or y > self.max_range_m:
            return None
        return float(x - self.lateral_mount_offset_m), float(y)

    def project_ground(self, x: float, y: float) -> Tuple[float, float]:
        """Forward projection: road point (x, y, 0) -> source pixel (u, v)."""
        y = max(1e-6, float(y))
        u = self.u_vp + (float(x) + self.lateral_mount_offset_m) * self.f_px / y
        v = self.v_vp + self.hf / y
        return float(u), float(v)

    def horizon_row(self) -> float:
        return float(self.v_vp)

    # ------------------------------------------------------------ noise model
    def range_sigma_m(self, y: float, sigma_v_px: Optional[float] = None) -> float:
        """1-sigma range error from bbox bottom-edge jitter.

        dy/dv = -h*f/(v-v_vp)^2 = -y^2/(h*f), the d^2/h law. On this camera
        h*f = 873, so one pixel is 0.26 m at 15 m but 5.6 m at 70 m.
        """
        s = self.sigma_v_px if sigma_v_px is None else float(sigma_v_px)
        return float(y) * float(y) / self.hf * s

    def lateral_sigma_m(
        self,
        x: float,
        y: float,
        sigma_u_px: Optional[float] = None,
        sigma_v_px: Optional[float] = None,
    ) -> float:
        """1-sigma lateral error, combining column jitter and row jitter."""
        su = self.sigma_u_px if sigma_u_px is None else float(sigma_u_px)
        sv = self.sigma_v_px if sigma_v_px is None else float(sigma_v_px)
        d_du = float(y) / self.f_px                       # dx/du
        d_dv = float(x) * float(y) / self.hf              # |dx/dv|
        return float(math.hypot(d_du * su, d_dv * sv))

    def measurement_cov(self, x: float, y: float) -> np.ndarray:
        """2x2 R for a BEV filter measuring (x, y) in metres."""
        sx = self.lateral_sigma_m(x, y)
        sy = self.range_sigma_m(y)
        # YOLO box centre is a lot noisier than 2 px; a tight R makes the
        # χ² gate reject real cars and the filter coasts into nonsense.
        return np.diag([max(sx, 0.55) ** 2, max(sy, 1.00) ** 2])

    # ------------------------------------------------------------- rescaling
    def scaled_to(self, width: int, height: int) -> "GroundCalibration":
        """Rescale to another resolution of the *same* framing.

        Use this and never hand-edit intrinsics after a resize. Anisotropic
        resizes are rejected: a non-square-pixel calibration cannot be expressed
        by this class, and silently accepting one is how the current P matrix
        ended up with fx=1007.5 alongside fy=632.9.
        """
        sx = float(width) / float(self.source_width)
        sy = float(height) / float(self.source_height)
        if abs(sx - sy) > 1e-3:
            raise ValueError(
                f"anisotropic resize {sx:.4f} vs {sy:.4f} would break the "
                "square-pixel model; crop to the target aspect first"
            )
        return GroundCalibration(
            source_width=int(width),
            source_height=int(height),
            f_px=self.f_px * sx,
            u_vp=self.u_vp * sx,
            v_vp=self.v_vp * sy,
            cam_height_m=self.cam_height_m,
            lateral_mount_offset_m=self.lateral_mount_offset_m,
            longitudinal_offset_m=self.longitudinal_offset_m,
            roll_rad=self.roll_rad,
            max_range_m=self.max_range_m,
            sigma_v_px=self.sigma_v_px * sy,
            sigma_u_px=self.sigma_u_px * sx,
            source=self.source,
        )

    def adapted_to(self, width: int, height: int) -> "GroundCalibration":
        """Fit this calib to a frame size without ever raising.

        Same aspect → uniform `scaled_to`. Same width, different height (Garmin
        HUD vs nohud) → keep f and the VP, change only `source_height`.
        Different aspect of the *same* camera (letterbox / HUD strip) → scale
        u_vp with width, v_vp with height, f with width (HFOV held). That is a
        guess, not a calibration; `source` is tagged so the UI can show it.
        A different camera must get its own JSON or `self_calib.bootstrap_calib`,
        never this path.
        """
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            return self
        if (width, height) == (self.source_width, self.source_height):
            return self
        sx = float(width) / float(self.source_width)
        sy = float(height) / float(self.source_height)
        if abs(sx - sy) <= 1e-3:
            return self.scaled_to(width, height)
        if width == self.source_width:
            return GroundCalibration(
                **{**asdict(self), "source_height": height,
                   "source": self.source + "+height_adjusted"}
            )
        return GroundCalibration(
            source_width=width,
            source_height=height,
            f_px=self.f_px * sx,
            u_vp=self.u_vp * sx,
            v_vp=self.v_vp * sy,
            cam_height_m=self.cam_height_m,
            lateral_mount_offset_m=self.lateral_mount_offset_m,
            longitudinal_offset_m=self.longitudinal_offset_m,
            roll_rad=self.roll_rad,
            max_range_m=self.max_range_m,
            sigma_v_px=self.sigma_v_px * sy,
            sigma_u_px=self.sigma_u_px * sx,
            source=f"{self.source}+adapted_{width}x{height}",
        )

    def cropped(self, crop_x: float, crop_y: float,
                width: int, height: int) -> "GroundCalibration":
        """Shift the VP for a pure crop. f and h are unchanged by cropping."""
        return GroundCalibration(
            source_width=int(width),
            source_height=int(height),
            f_px=self.f_px,
            u_vp=self.u_vp - float(crop_x),
            v_vp=self.v_vp - float(crop_y),
            cam_height_m=self.cam_height_m,
            lateral_mount_offset_m=self.lateral_mount_offset_m,
            longitudinal_offset_m=self.longitudinal_offset_m,
            roll_rad=self.roll_rad,
            max_range_m=self.max_range_m,
            sigma_v_px=self.sigma_v_px,
            sigma_u_px=self.sigma_u_px,
            source=self.source,
        )

    def with_horizon(self, v_vp: float) -> "GroundCalibration":
        """Replace only the horizon row, for online pitch tracking."""
        return GroundCalibration(**{**asdict(self), "v_vp": float(v_vp)})

    def with_pitch_height(self, pitch_deg: float, cam_height_m: float
                          ) -> "GroundCalibration":
        """UI retune of *this* camera. Never writes OpenLane P.

        Pitch moves v_vp about the image centre: v_vp = cy + f tan(pitch).
        Height scales range and lateral together. u_vp and f stay put.
        """
        pitch = math.radians(float(pitch_deg))
        cy = 0.5 * float(self.source_height)
        v_vp = cy + float(self.f_px) * math.tan(pitch)
        lo, hi = 0.25 * self.source_height, 0.90 * self.source_height
        return GroundCalibration(**{
            **asdict(self),
            "v_vp": float(np.clip(v_vp, lo, hi)),
            "cam_height_m": float(np.clip(cam_height_m, 0.6, 3.0)),
            "source": f"{self.source}+ui",
        })

    # ------------------------------------------------------------------- io
    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path: str) -> "GroundCalibration":
        with open(path) as fh:
            return cls(**json.load(fh))

    @classmethod
    def _json_for_stem(cls, stem: str) -> Optional[str]:
        """Find models/calib JSON for a video stem, including *_360 resizes.

        GRMN6694_360.mp4 must pick up GRMN6694_540_nohud.json, not the generic
        fallback. The lane net always sees 480x360 via CameraTransform; that is
        independent of this file. Object ranging uses the decoded frame size.
        """
        if not os.path.isdir(_CALIB_DIR):
            return None
        tokens = stem.replace("-", "_").split("_")
        names = [
            stem,
            stem.replace("_nohud", ""),
            stem.replace("_360", ""),
            stem.replace("_540p30", "").replace("_540", ""),
            tokens[0] if tokens else stem,
        ]
        for name in names:
            p = os.path.join(_CALIB_DIR, f"{name}.json")
            if name and os.path.isfile(p):
                return p
        # Same camera id, different suffix (GRMN6694_360 → GRMN6694_540_nohud.json).
        # Require a long prefix so ADAS2 / webcam / front.mp4 never inherit a
        # Garmin JSON just because something in this folder starts with "GRMN".
        prefix = tokens[0] if tokens else stem
        if not prefix or len(prefix) < 6:
            return None
        hits = sorted(
            f for f in os.listdir(_CALIB_DIR)
            if f.startswith(prefix) and f.endswith(".json")
        )
        if not hits:
            return None
        nohud = [f for f in hits if "nohud" in f]
        return os.path.join(_CALIB_DIR, (nohud or hits)[0])

    @classmethod
    def for_video(cls, video_path: Optional[str],
                  frame_shape: Optional[Tuple[int, int]] = None
                  ) -> "GroundCalibration":
        """Calib for this clip: JSON if present, else self-estimate, else generic.

        Never copies Garmin pixel numbers onto an unrelated camera. A measured
        JSON is still preferred; `scripts/calibrate_from_video.py --write` is
        how you replace the HFOV prior with a metric f.
        """
        from src.utils.self_calib import bootstrap_calib, generic_calib

        w = h = None
        if frame_shape is not None:
            h, w = int(frame_shape[0]), int(frame_shape[1])

        if video_path:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            path = cls._json_for_stem(stem)
            if path:
                calib = cls.from_json(path)
                if w and h:
                    calib = calib.adapted_to(w, h)
                return calib
            if w is None or h is None:
                try:
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    cap.release()
                except Exception:
                    w = h = 0
            if w and h:
                try:
                    return bootstrap_calib(w, h, video_path=video_path)
                except Exception:
                    return generic_calib(w, h)

        if w and h:
            return generic_calib(w, h)
        return generic_calib(960, 540)


# Measured on testing_new_videos/GRMN6694_540_nohud.mp4 by
# scripts/calibrate_from_video.py. Cross-checked three independent ways:
#   * Hough vanishing point of lane markings          -> v_vp 316.6 (IQR 313.8-318.4)
#   * lane-marking separation vs row, W=3.7 m         -> v_vp 315.5, h 1.345 m
#     (residuals 0.7/-1.3/0.7 px over rows 380/400/420)
#   * optical flow of static road texture vs HUD mph  -> h*f 873 px*m (CI 861-885)
# The first two agree on the horizon to ~1 px without sharing any assumption,
# which is the main reason to trust this over the OpenLane P matrix.
GARMIN_960x502 = GroundCalibration(
    source_width=960,
    source_height=502,
    f_px=649.0,
    u_vp=527.6,
    v_vp=316.0,
    cam_height_m=1.345,
    lateral_mount_offset_m=0.0,
    max_range_m=DEFAULT_MAX_RANGE_M,
    source="measured:GRMN6694_540_nohud",
)

# What src/utils/calibration.py's P matrix actually implies for a 960x502 frame
# after the 20% sky crop. Kept so the diagnostics harness can plot old vs new on
# the same axes, and so the numbers in this module's docstring are checkable.
OPENLANE_P_EQUIVALENT = GroundCalibration(
    source_width=960,
    source_height=502,
    f_px=632.9,           # vertical effective f; the horizontal one is 1007.5
    u_vp=480.0,
    v_vp=302.8,
    cam_height_m=0.9422,  # h * (ratio_y/ratio_x) = 1.5 * 1.125, then /1.125 in x
    source="derived:OpenLane_P_matrix",
)


def bbox_ground_point(bbox) -> Tuple[float, float]:
    """Bbox bottom-centre: the tyre contact line, in source pixels."""
    x1, y1, x2, y2 = (float(c) for c in bbox)
    return 0.5 * (x1 + x2), y2
