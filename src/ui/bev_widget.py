"""
PySide6 Interactive 3D Rotatable Telemetry BEV Widget
Renders dynamic 3D perspective Bird's Eye View (BEV) map with:
 - 360° Mouse Yaw & Pitch Camera Rotation
 - Top-down car sprites (ego + detected) perspective-mapped onto the ground
 - 3D Perspective Distance Arcs & Gridlines
 - Translucent 3D Drivable Corridor Mesh & Glowing Lane Ribbons
 - 3D Positioned Vehicle Target Markers & ID Badges
"""

import os

import numpy as np
from PySide6.QtCore import Qt, QPointF, QElapsedTimer
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPolygonF, QFont, QPainterPath,
    QLinearGradient, QRadialGradient, QTransform, QPixmap
)

from src.inference.postprocess import ANCHOR_Y_STEPS
from src.utils.drivable_area import parse_lane_components
from src.ui.bev_sprites import (
    load_car_pixmap, load_ready_png, load_car_sprite_pool,
    find_white_sprite_index, hw_from_pixmap, EGO_ONLY_NAMES,
)

# BEV car overlay rules
BEV_NEAR_M = 22.0          # near cars → white skin
BEV_MAX_DIST_M = 55.0      # hide cars beyond this range
BEV_MAX_LATERAL_M = 6.2    # ~ego + one adjacent lane; backup 3rd-lane cut

# Preferred BEV camera (matches tuned screenshot)
DEFAULT_VIEW_PITCH = 31.0
DEFAULT_VIEW_YAW = 0.0
DEFAULT_ZOOM = 1.05
DEFAULT_CALIB_PITCH = -7.0
DEFAULT_CALIB_H = 1.0
# Elevated rear/side car body height (meters) — gives volume, not flat stickers
CAR_ROOF_H = 1.55
EGO_ROOF_H = 1.65


class BEVWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(450, 600)
        self.setMouseTracking(True)

        # ── 3D Interactive Camera State (matches preferred angle / zoom) ─────
        self.pitch_deg = DEFAULT_VIEW_PITCH
        self.yaw_deg = DEFAULT_VIEW_YAW
        self.cam_h = 10.0
        self.cam_dist = 12.0
        self.zoom_factor = DEFAULT_ZOOM
        self.pan_offset = QPointF(0, 0)

        self.is_rotating = False
        self.is_panning = False
        self.last_mouse_pos = QPointF(0, 0)

        # ── Current Render Data ──────────────────────────────────────────────
        self.proposals = []
        self.processed_objs = []
        self.cipo_status = "SAFE"
        self.left_3d = None
        self.right_3d = None

        # ── Calibration Offsets (extrinsics tuner defaults) ──────────────────
        self.calib_pitch = DEFAULT_CALIB_PITCH
        self.calib_h = DEFAULT_CALIB_H

        # ── Car sprites — ego from 1-back.png (elevated rear/side look) ──────
        _data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
        _cache = os.path.join(_data, 'cache', 'bev_cars')
        _car_dir = os.path.join(_data, 'sides_car')
        if not os.path.isdir(_car_dir):
            _car_dir = os.path.join(_data, 'side_car')
        _ready = os.path.join(_car_dir, 'ready')

        # Ego: prefer background-removed 1-back.png in sides_car/ready
        _ego_candidates = [
            os.path.join(_ready, '1-back.png'),
            os.path.join(_data, 'side_view', 'ready', '1-back.png'),
            os.path.join(_data, 'front', 'tooooo.avif'),
        ]
        self.ego_pixmap = None
        for _ego_path in _ego_candidates:
            if not os.path.isfile(_ego_path):
                continue
            if _ego_path.lower().endswith('.png'):
                self.ego_pixmap = load_ready_png(_ego_path)
            else:
                self.ego_pixmap = load_car_pixmap(_ego_path, cache_dir=_cache)
            if self.ego_pixmap is not None:
                break

        self._ego_hl = 2.15
        self._ego_hw = hw_from_pixmap(self.ego_pixmap, self._ego_hl)
        self._ego_roof = EGO_ROOF_H

        # Traffic pool excludes ego-only plates
        self.car_pixmaps, self.car_names = load_car_sprite_pool(
            _car_dir, cache_dir=_cache, exclude_names=EGO_ONLY_NAMES
        )
        self._white_idx = find_white_sprite_index(self.car_names)
        self._car_hl = 1.95
        self._car_hw = hw_from_pixmap(
            self.car_pixmaps[0] if self.car_pixmaps else None, self._car_hl
        )
        self._car_roof = CAR_ROOF_H

        # track_id → sprite index (stable per vehicle across frames; near cars override to white)
        self._id_sprite = {}

        # ── Theme Palette ────────────────────────────────────────────────────
        self.bg_color = QColor(14, 18, 24)
        self.road_color = QColor(24, 29, 38)
        self.grid_color = QColor(42, 50, 62)
        self.grid_text_color = QColor(130, 145, 165)

        # Cinematic highway look (reference chase-cam road) — default ON
        self.cinematic_road = True
        self.show_lane_lines = False  # corridor-only until LANES toggle is on


        # Running dashed-lane scroll — advances on detection frames only (no extra timer repaints)
        self._dash_phase = 0.0
        self._dash_len = 4.5
        self._dash_gap = 5.5
        self._dash_speed_mps = 15.0
        self._dash_clock = QElapsedTimer()
        self._dash_clock.start()
        self._dash_last_ms = 0

        # Cached static cinematic road (asphalt + solid edges) — rebuilt on view change
        self._road_cache = None  # QPixmap | None
        self._road_cache_key = None

    def toggle_cinematic_road(self):
        """Switch between cinematic highway and classic grid road."""
        self.cinematic_road = not self.cinematic_road
        self._invalidate_road_cache()
        self.update()
        return self.cinematic_road

    def toggle_lane_lines(self):
        """Show / hide Anchor3D detection lane lines on the BEV canvas."""
        return self.set_lane_lines(not self.show_lane_lines)

    def set_lane_lines(self, on):
        self.show_lane_lines = bool(on)
        self.update()
        return self.show_lane_lines

    def _invalidate_road_cache(self):
        self._road_cache = None
        self._road_cache_key = None

    def _road_cache_signature(self, w, h):
        return (
            int(w), int(h),
            round(self.pitch_deg, 2), round(self.yaw_deg, 2),
            round(self.zoom_factor, 3),
            round(self.pan_offset.x(), 1), round(self.pan_offset.y(), 1),
            round(self.calib_pitch, 2), round(self.calib_h, 2),
            bool(self.cinematic_road),
        )

    def _advance_dash_phase(self, has_detection: bool, speed_mps=None, dt=None):
        """Scroll dashes by real ego odometry when available, else wall-clock dt."""
        if not self.cinematic_road or not has_detection:
            return
        if speed_mps is not None and dt is not None and float(speed_mps) > 0.5:
            step = float(speed_mps) * min(0.12, max(0.0, float(dt)))
        else:
            now = self._dash_clock.elapsed()
            if self._dash_last_ms <= 0:
                self._dash_last_ms = now
                return
            wall_dt = min(0.12, max(0.0, (now - self._dash_last_ms) / 1000.0))
            self._dash_last_ms = now
            step = self._dash_speed_mps * wall_dt
        period = self._dash_len + self._dash_gap
        self._dash_phase = (self._dash_phase + step) % period

    def _bev_visible(self, obj) -> bool:
        """Distance + lane filters: ego/2nd lane only, within BEV_MAX_DIST_M."""
        z = float(obj.get('Z_3d', 0.0))
        x = float(obj.get('X_3d', 0.0))
        if not (0.5 < z <= BEV_MAX_DIST_M):
            return False
        if obj.get('show_bev') is False or int(obj.get('lane_rank', 0)) >= 2:
            return False
        if abs(x) > BEV_MAX_LATERAL_M:
            return False
        return True

    def _assign_sprites_for_frame(self, objs):
        """
        Near cars (< BEV_NEAR_M) always use the white skin.
        Farther cars use other skins by track_id (white reserved for near when possible).
        """
        if not self.car_pixmaps:
            return {}
        n = len(self.car_pixmaps)
        white = self._white_idx if self._white_idx is not None else 0
        used = set()
        assign = {}

        # Near first so they claim white, then farther cars pick other skins
        ordered = sorted(
            enumerate(objs),
            key=lambda t: (float(t[1].get('Z_3d', 99.0)), int(t[1].get('track_id', -1))),
        )
        for oi, obj in ordered:
            if not self._bev_visible(obj):
                continue
            z = float(obj.get('Z_3d', 99.0))
            tid = int(obj.get('track_id', -1))
            is_near = z <= BEV_NEAR_M

            if is_near:
                idx = white
            else:
                preferred = self._id_sprite.get(tid) if tid > 0 else None
                # Prefer non-white skins for mid/far cars
                if preferred is not None and preferred != white and preferred not in used:
                    idx = preferred
                elif tid > 0:
                    start = tid % n
                    idx = None
                    for k in range(n):
                        cand = (start + k) % n
                        if cand == white and n > 1:
                            continue
                        if cand not in used:
                            idx = cand
                            break
                    if idx is None:
                        for k in range(n):
                            cand = (start + k) % n
                            if cand not in used:
                                idx = cand
                                break
                    if idx is None:
                        idx = start if start != white or n == 1 else (start + 1) % n
                    self._id_sprite[tid] = idx
                else:
                    idx = None
                    for cand in range(n):
                        if cand == white and n > 1:
                            continue
                        if cand not in used:
                            idx = cand
                            break
                    if idx is None:
                        idx = oi % n

            used.add(idx)
            assign[oi] = self.car_pixmaps[idx]
        return assign

    def _draw_car_pixmap(self, painter, pixmap, p_fl, p_fr, p_rr, p_rl):
        """Perspective-map a car pixmap onto the 4 projected footprint corners."""
        img_w = float(pixmap.width())
        img_h = float(pixmap.height())
        src_quad = QPolygonF([
            QPointF(0.0, 0.0),
            QPointF(img_w, 0.0),
            QPointF(img_w, img_h),
            QPointF(0.0, img_h),
        ])
        dst_quad = QPolygonF([p_fl, p_fr, p_rr, p_rl])
        transform = QTransform()
        if not QTransform.quadToQuad(src_quad, dst_quad, transform):
            return False
        painter.save()
        clip = QPainterPath()
        clip.addPolygon(dst_quad)
        painter.setClipPath(clip)
        painter.setTransform(transform, combine=True)
        painter.setOpacity(1.0)
        painter.drawPixmap(0, 0, pixmap)
        painter.restore()
        return True

    def _elevated_car_quad(self, x, y, hw, hl, roof_h, width, height):
        """
        Build a rear/elevated volume quad (not a flat ground sticker).

        Sprite top  → farther + raised (roof / nose away)
        Sprite bottom → nearer on the road (rear bumper toward camera)

        Matches the chase-cam side/rear look used for ego (1-back.png).
        Returns (p_fl, p_fr, p_rr, p_rl) for _draw_car_pixmap.
        """
        roof_w = hw * 0.80
        y_front = y + hl * 0.50
        y_rear = y - hl * 0.42
        p_fl = self.world_to_canvas_3d(x - roof_w, y_front, roof_h, width, height)
        p_fr = self.world_to_canvas_3d(x + roof_w, y_front, roof_h, width, height)
        p_rr = self.world_to_canvas_3d(x + hw, y_rear, 0.04, width, height)
        p_rl = self.world_to_canvas_3d(x - hw, y_rear, 0.04, width, height)
        return p_fl, p_fr, p_rr, p_rl

    def _ego_front_y(self) -> float:
        """World-Y (meters forward) at the ego front bumper / light origin."""
        return float(self._ego_hl) * 0.58

    @staticmethod
    def _interp_lane_x_at_y(pts_3d, y_target):
        """Interpolate lateral X of a corridor polyline at forward distance y_target."""
        arr = np.asarray(pts_3d, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 1:
            return None
        ys = arr[:, 1]
        xs = arr[:, 0]
        order = np.argsort(ys)
        ys = ys[order]
        xs = xs[order]
        if y_target <= ys[0]:
            return float(xs[0])
        if y_target >= ys[-1]:
            return float(xs[-1])
        return float(np.interp(y_target, ys, xs))

    def _corridor_from_ego_front(self, left_3d, right_3d, y_start):
        """
        Rebuild ego corridor so the green light starts at the front of the car
        (no overlap on the body) while keeping the true lane width from detections.
        """
        left = np.asarray(left_3d, dtype=np.float64)
        right = np.asarray(right_3d, dtype=np.float64)
        if left.ndim != 2 or right.ndim != 2 or len(left) < 2 or len(right) < 2:
            return None, None

        x_l0 = self._interp_lane_x_at_y(left, y_start)
        x_r0 = self._interp_lane_x_at_y(right, y_start)
        if x_l0 is None or x_r0 is None:
            return None, None

        # Ensure left is left of right (width intact)
        if x_l0 > x_r0:
            x_l0, x_r0 = x_r0, x_l0

        z0 = float(left[0, 2]) if left.shape[1] > 2 else 0.0
        left_out = [(x_l0, float(y_start), z0)]
        right_out = [(x_r0, float(y_start), z0)]

        for row in left:
            if float(row[1]) > y_start + 0.05:
                left_out.append((float(row[0]), float(row[1]), float(row[2]) if len(row) > 2 else 0.0))
        for row in right:
            if float(row[1]) > y_start + 0.05:
                right_out.append((float(row[0]), float(row[1]), float(row[2]) if len(row) > 2 else 0.0))

        if len(left_out) < 2 or len(right_out) < 2:
            # Fallback short wedge with correct width at front
            left_out = [(x_l0, y_start, z0), (x_l0, y_start + 25.0, z0)]
            right_out = [(x_r0, y_start, z0), (x_r0, y_start + 25.0, z0)]

        return left_out, right_out

    def set_calibration(self, pitch_deg, height_m):
        """Updates extrinsics calibration offset."""
        self.calib_pitch = pitch_deg
        self.calib_h = height_m
        self._invalidate_road_cache()
        self.update()

    def update_bev_data(self, proposals, processed_objs=None, cipo_status="SAFE", left_3d=None,
                        right_3d=None,         speed_mps=None, dt=1.0 / 30.0, alerts=None):
        """Receives new frame BEV data and triggers UI render update."""
        self.proposals = proposals if proposals is not None else []
        # Trust depth-model X/Z as-is (no artificial lateral nudging)
        self.processed_objs = processed_objs if processed_objs is not None else []
        self.cipo_status = cipo_status
        self.left_3d = left_3d
        self.right_3d = right_3d

        has_detection = bool(self.proposals) or bool(self.processed_objs)
        self._advance_dash_phase(has_detection, speed_mps=speed_mps, dt=dt)
        self.update()

    # ─────────────────────────────────────────────────────────────────────────
    # 3D PERSPECTIVE PROJECTION MATH ENGINE
    # ─────────────────────────────────────────────────────────────────────────

    def world_to_canvas_3d(self, x, y, z=0.0, width=None, height=None):
        """
        Projects 3D world coordinate (X=lateral m, Y=forward m, Z=height m)
        onto 2D viewport pixels using a 3D perspective camera matrix with Pitch & Yaw.
        """
        w = width if width is not None else self.width()
        h = height if height is not None else self.height()

        focal = 460.0 * self.zoom_factor

        dx = x
        dy = y + self.cam_dist
        dz = z - (self.cam_h * (self.calib_h / 1.6))

        rad_y = np.radians(self.yaw_deg)
        rx = dx * np.cos(rad_y) - dy * np.sin(rad_y)
        ry = dx * np.sin(rad_y) + dy * np.cos(rad_y)
        rz = dz

        rad_p = np.radians(self.pitch_deg + self.calib_pitch)
        cy = ry * np.cos(rad_p) - rz * np.sin(rad_p)
        cz = ry * np.sin(rad_p) + rz * np.cos(rad_p)
        cx = rx

        if cy <= 0.2:
            return None

        px = (w / 2.0) + (cx / cy) * focal + self.pan_offset.x()
        py = (h * 0.72) - (cz / cy) * focal + self.pan_offset.y()

        return QPointF(px, py)

    # ─────────────────────────────────────────────────────────────────────────
    # MOUSE INTERACTION HANDLERS
    # ─────────────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_rotating = True
            self.last_mouse_pos = event.position()
        elif event.button() == Qt.RightButton:
            self.is_panning = True
            self.last_mouse_pos = event.position()

    def mouseMoveEvent(self, event):
        pos = event.position()
        delta = pos - self.last_mouse_pos
        self.last_mouse_pos = pos

        if self.is_rotating:
            self.yaw_deg = max(-85.0, min(85.0, self.yaw_deg + delta.x() * 0.35))
            self.pitch_deg = max(5.0, min(75.0, self.pitch_deg - delta.y() * 0.35))
            self._invalidate_road_cache()
            self.update()
        elif self.is_panning:
            self.pan_offset += delta
            self._invalidate_road_cache()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_rotating = False
        elif event.button() == Qt.RightButton:
            self.is_panning = False

    def wheelEvent(self, event):
        angle_delta = event.angleDelta().y()
        factor = 1.12 if angle_delta > 0 else 0.88
        new_zoom = self.zoom_factor * factor
        if 0.4 <= new_zoom <= 4.5:
            self.zoom_factor = new_zoom
            self._invalidate_road_cache()
            self.update()

    def reset_view(self):
        """Resets camera to preferred BEV angle / zoom."""
        self.set_dashboard_camera("chase")

    def set_dashboard_camera(self, mode):
        """Chase camera for ADAS View, steeper pitch for Bird's-Eye."""
        bird = str(mode) == "bird"
        self.pitch_deg = 62.0 if bird else DEFAULT_VIEW_PITCH
        self.yaw_deg = DEFAULT_VIEW_YAW
        self.cam_h = 18.0 if bird else 10.0
        self.zoom_factor = 0.82 if bird else DEFAULT_ZOOM
        self.pan_offset = QPointF(0, 0)
        self._invalidate_road_cache()
        self.update()

    # ─────────────────────────────────────────────────────────────────────────
    # ROAD / LANE DRAW HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _canvas_polyline(self, world_xy, w, h):
        pts = []
        for x, y in world_xy:
            p = self.world_to_canvas_3d(float(x), float(y), 0.0, w, h)
            if p is not None:
                pts.append(p)
        return pts

    def _stroke_glow_path(self, painter, canvas_pts, core: QColor, core_w=2.4, glow_w=7.0, glow_a=55):
        if len(canvas_pts) < 2:
            return
        path = QPainterPath()
        path.moveTo(canvas_pts[0])
        for p in canvas_pts[1:]:
            path.lineTo(p)
        painter.setBrush(Qt.NoBrush)
        # Fast path: single stroke (glow pass optional for thick solids only)
        if glow_w > core_w + 1.5 and glow_a > 0:
            glow = QColor(core.red(), core.green(), core.blue(), glow_a)
            painter.setPen(QPen(glow, glow_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(path)
        painter.setPen(QPen(core, core_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPath(path)

    def _draw_world_segment(self, painter, x0, y0, x1, y1, w, h, core, core_w=2.4, glow_w=7.0):
        pts = self._canvas_polyline([(x0, y0), (x1, y1)], w, h)
        self._stroke_glow_path(painter, pts, core, core_w=core_w, glow_w=glow_w)

    def _draw_dashed_lane_x(
        self, painter, x, y0, y1, w, h, core,
        dash=3.2, gap=4.0, core_w=2.2, phase=0.0,
    ):
        """
        Draw a dashed lane with optional scroll phase (meters).
        Increasing phase moves dashes toward the ego (forward-motion feel).
        Single-stroke only for FPS.
        """
        period = float(dash + gap)
        if period < 1e-3:
            return
        y = float(y0) - (float(phase) % period) - period
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(core, core_w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        while y < y1:
            seg0 = max(y, float(y0))
            seg1 = min(y + dash, float(y1))
            if seg1 > seg0 + 0.08:
                pts = self._canvas_polyline([(x, seg0), (x, seg1)], w, h)
                if len(pts) >= 2:
                    painter.drawLine(pts[0], pts[1])
            y += period

    def _draw_car_shadow(self, painter, x, y, hw, hl, w, h):
        """Cheap ground oval under a car (no radial polygon)."""
        p = self.world_to_canvas_3d(x, y, 0.0, w, h)
        if p is None:
            return
        # Scale shadow with perspective via nearby lateral points
        pl = self.world_to_canvas_3d(x - hw, y, 0.0, w, h)
        pr = self.world_to_canvas_3d(x + hw, y, 0.0, w, h)
        if pl is None or pr is None:
            rw, rh = 18.0, 8.0
        else:
            rw = max(10.0, abs(pr.x() - pl.x()) * 0.55)
            rh = max(5.0, rw * 0.35)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 90)))
        painter.drawEllipse(p, rw, rh)

    def _draw_classic_road(self, painter, w, h):
        """Original grid telemetry road."""
        road_pts_3d = [(-6.0, 0.0), (+6.0, 0.0), (+6.0, 80.0), (-6.0, 80.0)]
        road_pts_2d = [self.world_to_canvas_3d(x, y, 0.0, w, h) for x, y in road_pts_3d]
        if all(p is not None for p in road_pts_2d):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(self.road_color))
            painter.drawPolygon(QPolygonF(road_pts_2d))

        painter.setFont(QFont("Inter", 8))
        grid_pen = QPen(self.grid_color, 1, Qt.DotLine)

        for y_m in range(10, 85, 10):
            arc_pts = []
            for x_m in np.linspace(-7.0, 7.0, 15):
                pt = self.world_to_canvas_3d(x_m, float(y_m), 0.0, w, h)
                if pt is not None:
                    arc_pts.append(pt)
            if len(arc_pts) > 1:
                path = QPainterPath()
                path.moveTo(arc_pts[0])
                for p in arc_pts[1:]:
                    path.lineTo(p)
                painter.setPen(grid_pen)
                painter.drawPath(path)
                mid_pt = arc_pts[len(arc_pts) // 2]
                painter.setPen(QPen(self.grid_text_color))
                painter.drawText(int(mid_pt.x() - 14), int(mid_pt.y() - 3), f"{y_m}m")

        for x_m in [-4.0, -2.0, 0.0, 2.0, 4.0]:
            line_pts = []
            for y_m in np.linspace(0.0, 80.0, 10):
                pt = self.world_to_canvas_3d(x_m, y_m, 0.0, w, h)
                if pt is not None:
                    line_pts.append(pt)
            if len(line_pts) > 1:
                path = QPainterPath()
                path.moveTo(line_pts[0])
                for p in line_pts[1:]:
                    path.lineTo(p)
                line_pen = QPen(QColor(0, 200, 255, 160), 1.5, Qt.SolidLine) if x_m == 0.0 else grid_pen
                painter.setPen(line_pen)
                painter.drawPath(path)

    def _paint_cinematic_static(self, painter, w, h):
        """Static asphalt + solid edges (no animated dashes) — used for cache bake."""
        shoulder = [(-14.0, -2.0), (14.0, -2.0), (14.0, 85.0), (-14.0, 85.0)]
        sh_pts = [self.world_to_canvas_3d(x, y, 0.0, w, h) for x, y in shoulder]
        if all(p is not None for p in sh_pts):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(10, 12, 16)))
            painter.drawPolygon(QPolygonF(sh_pts))

        # Sparse shoulder grid (few lines only)
        painter.setPen(QPen(QColor(28, 34, 44, 80), 1, Qt.DotLine))
        for x_m in (-12.0, -9.0, 9.0, 12.0):
            pts = self._canvas_polyline([(x_m, 0.0), (x_m, 80.0)], w, h)
            if len(pts) > 1:
                painter.drawLine(pts[0], pts[-1])

        road_l, road_r = -5.6, 5.6
        asphalt = [(road_l, -1.0), (road_r, -1.0), (road_r, 82.0), (road_l, 82.0)]
        asp_pts = [self.world_to_canvas_3d(x, y, 0.0, w, h) for x, y in asphalt]
        if all(p is not None for p in asp_pts):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(16, 18, 22)))
            painter.drawPolygon(QPolygonF(asp_pts))
            mid_near = self.world_to_canvas_3d(0.0, 2.0, 0.0, w, h)
            mid_far = self.world_to_canvas_3d(0.0, 55.0, 0.0, w, h)
            gloss = [(-1.2, 0.0), (1.2, 0.0), (0.8, 70.0), (-0.8, 70.0)]
            g_pts = [self.world_to_canvas_3d(x, y, 0.0, w, h) for x, y in gloss]
            if mid_near is not None and mid_far is not None and all(p is not None for p in g_pts):
                g = QLinearGradient(mid_near, mid_far)
                g.setColorAt(0.0, QColor(48, 54, 64, 55))
                g.setColorAt(1.0, QColor(20, 22, 28, 0))
                painter.setBrush(QBrush(g))
                painter.drawPolygon(QPolygonF(g_pts))

        white = QColor(235, 240, 255)
        self._draw_world_segment(painter, -5.35, 0.5, -5.35, 78.0, w, h, white, core_w=2.6, glow_w=6.0)
        self._draw_world_segment(painter,  5.35, 0.5,  5.35, 78.0, w, h, white, core_w=2.6, glow_w=6.0)

        painter.setFont(QFont("Inter", 7))
        painter.setPen(QPen(QColor(160, 175, 195, 110)))
        for y_m in (20, 40, 60):
            pt = self.world_to_canvas_3d(0.0, float(y_m), 0.0, w, h)
            if pt is not None:
                painter.drawText(int(pt.x() + 8), int(pt.y() - 2), f"{y_m}m")

    def _draw_cinematic_road(self, painter, w, h):
        """
        Chase-cam highway with cached static layer + live scrolling dashes.
        """
        key = self._road_cache_signature(w, h)
        if self._road_cache is None or self._road_cache_key != key:
            cache = QPixmap(w, h)
            cache.fill(Qt.transparent)
            cp = QPainter(cache)
            cp.setRenderHint(QPainter.Antialiasing, True)
            self._paint_cinematic_static(cp, w, h)
            cp.end()
            self._road_cache = cache
            self._road_cache_key = key

        painter.drawPixmap(0, 0, self._road_cache)

        # Decorative dashed separators always on (not tied to Anchor3D toggle)
        white = QColor(244, 247, 251)
        phase = self._dash_phase
        self._draw_dashed_lane_x(
            painter, -1.85, 2.0, 78.0, w, h, white,
            dash=self._dash_len, gap=self._dash_gap, core_w=2.2, phase=phase,
        )
        self._draw_dashed_lane_x(
            painter, 1.85, 2.0, 78.0, w, h, white,
            dash=self._dash_len, gap=self._dash_gap, core_w=2.2, phase=phase,
        )

    def _draw_detected_lanes(self, painter, w, h):
        """Overlay Anchor3D model lane proposals (gated by show_lane_lines)."""
        if not self.show_lane_lines or not self.proposals:
            return
        for lane in self.proposals:
            xs, ys, zs, vis = parse_lane_components(lane)
            if vis.sum() < 2:
                continue
            pts_world = [(xs[i], ys[i]) for i in range(len(xs)) if vis[i]]
            mean_x = float(np.mean([wx for wx, wy in pts_world]))
            if abs(mean_x) > 7.5:
                continue
            pts_canvas = [self.world_to_canvas_3d(wx, wy, 0.0, w, h) for wx, wy in pts_world]
            valid_pts = [p for p in pts_canvas if p is not None]
            if len(valid_pts) < 2:
                continue

            if self.cinematic_road:
                core = QColor(255, 220, 0, 230) if abs(mean_x) < 2.0 else QColor(205, 175, 145, 200)
                self._stroke_glow_path(painter, valid_pts, core, core_w=2.2 if abs(mean_x) < 2.0 else 1.0, glow_w=0.0, glow_a=0)
            else:
                lane_color = QColor(255, 220, 0) if abs(mean_x) < 2.0 else QColor(205, 175, 145)
                painter.setPen(QPen(lane_color, 2.5 if abs(mean_x) < 2.0 else 1.0, Qt.SolidLine))
                path = QPainterPath()
                path.moveTo(valid_pts[0])
                for pt in valid_pts[1:]:
                    path.lineTo(pt)
                painter.drawPath(path)
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(Qt.NoPen)
                for pt in valid_pts[::3]:
                    painter.drawEllipse(pt, 2.0, 2.0)

    # ─────────────────────────────────────────────────────────────────────────
    # CANVAS RENDERER
    # ─────────────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter()
        if not painter.begin(self):
            return

        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

            w = self.width()
            h = self.height()

            # ── 1. Background ────────────────────────────────────────────────
            bg = QColor(6, 8, 12) if self.cinematic_road else self.bg_color
            painter.fillRect(self.rect(), bg)

            # ── 2–3. Road surface + grid / cinematic highway ─────────────────
            if self.cinematic_road:
                self._draw_cinematic_road(painter, w, h)
            else:
                self._draw_classic_road(painter, w, h)

            # ── 4. Drivable corridor (starts at ego front) ───────────────────
            if self.left_3d is not None and self.right_3d is not None and len(self.left_3d) > 0 and len(self.right_3d) > 0:
                y_light = self._ego_front_y()
                left_c, right_c = self._corridor_from_ego_front(self.left_3d, self.right_3d, y_light)
                if left_c is not None and right_c is not None:
                    pts_left  = [self.world_to_canvas_3d(x, y, 0.0, w, h) for x, y, _ in left_c]
                    pts_right = [self.world_to_canvas_3d(x, y, 0.0, w, h) for x, y, _ in right_c]
                    pts_left  = [p for p in pts_left if p is not None]
                    pts_right = [p for p in pts_right if p is not None]

                    if len(pts_left) > 1 and len(pts_right) > 1:
                        n = min(len(pts_left), len(pts_right))
                        # 0.52 of the left-to-right span, centered.
                        margin = 0.24
                        inset_l, inset_r, mids = [], [], []
                        for i in range(n):
                            left_pt, right_pt = pts_left[i], pts_right[i]
                            dx = right_pt.x() - left_pt.x()
                            dy = right_pt.y() - left_pt.y()
                            inset_l.append(QPointF(left_pt.x() + dx * margin, left_pt.y() + dy * margin))
                            inset_r.append(QPointF(right_pt.x() - dx * margin, right_pt.y() - dy * margin))
                            mids.append(QPointF(
                                (left_pt.x() + right_pt.x()) * 0.5,
                                (left_pt.y() + right_pt.y()) * 0.5,
                            ))
                        if self.cipo_status == "DANGER":
                            band = QColor(230, 45, 50, 165)
                            core = QColor(255, 210, 214)
                        elif self.cipo_status == "WARNING":
                            band = QColor(255, 195, 40, 165)
                            core = QColor(255, 240, 200)
                        else:
                            band = QColor(35, 245, 45, 160)
                            core = QColor(216, 255, 232)
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QBrush(band))
                        painter.drawPolygon(QPolygonF(inset_l + inset_r[::-1]))
                        painter.setBrush(Qt.NoBrush)
                        painter.setPen(QPen(core, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                        center_path = QPainterPath()
                        center_path.moveTo(mids[0])
                        for pt in mids[1:]:
                            center_path.lineTo(pt)
                        painter.drawPath(center_path)

            # ── 5. Detected lane overlays ────────────────────────────────────
            self._draw_detected_lanes(painter, w, h)

            # ── 7. Ego Vehicle — elevated rear/side volume (1-back.png) ──
            hw = self._ego_hw
            hl = self._ego_hl
            if self.cinematic_road:
                self._draw_car_shadow(painter, 0.0, 0.0, hw, hl, w, h)
            p_fl, p_fr, p_rr, p_rl = self._elevated_car_quad(
                0.0, 0.0, hw, hl, self._ego_roof, w, h
            )

            ego_drawn = False
            if self.ego_pixmap is not None and all(p is not None for p in [p_fl, p_fr, p_rr, p_rl]):
                ego_drawn = self._draw_car_pixmap(painter, self.ego_pixmap, p_fl, p_fr, p_rr, p_rl)

            if not ego_drawn and all(p is not None for p in [p_fl, p_fr, p_rr, p_rl]):
                fallback_poly = QPolygonF([p_fl, p_fr, p_rr, p_rl])
                painter.setPen(QPen(QColor(0, 220, 255, 200), 1.8))
                painter.setBrush(QBrush(QColor(18, 26, 38)))
                painter.drawPolygon(fallback_poly)

            # Soft red taillight markers only (no yellow head glow — looks dirty on ego)
            tail_l = self.world_to_canvas_3d(-hw * 0.7, -hl * 0.35, 0.25, w, h)
            tail_r = self.world_to_canvas_3d( hw * 0.7, -hl * 0.35, 0.25, w, h)
            if all(p is not None for p in [tail_l, tail_r]):
                for pt in [tail_l, tail_r]:
                    glow = QRadialGradient(pt, 7)
                    glow.setColorAt(0.0, QColor(255, 30, 30, 210))
                    glow.setColorAt(1.0, QColor(255, 0, 0, 0))
                    painter.setBrush(QBrush(glow))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(pt, 6.0, 6.0)

            # ── 8. Detected cars — ego/2nd lane only; distance-filtered ──
            if self.processed_objs:
                sprite_map = self._assign_sprites_for_frame(self.processed_objs)
                # Draw far → near so nearer cars occlude correctly
                draw_order = sorted(
                    range(len(self.processed_objs)),
                    key=lambda i: float(self.processed_objs[i].get('Z_3d', 0.0)),
                    reverse=True,
                )
                for oi in draw_order:
                    obj = self.processed_objs[oi]
                    if not self._bev_visible(obj):
                        continue
                    x_3d = float(obj['X_3d'])
                    z_3d = float(obj['Z_3d'])

                    pt_obj = self.world_to_canvas_3d(x_3d, z_3d, 0.0, w, h)
                    if pt_obj is None:
                        continue
                    if not (0 <= pt_obj.x() <= w and 0 <= pt_obj.y() <= h):
                        continue

                    is_cipo = obj.get('is_cipo', False)
                    in_path = obj.get('in_path', False)
                    obj_status = str(obj.get('status', self.cipo_status))

                    if is_cipo and obj_status == "DANGER":
                        marker_color = QColor(255, 40, 40)
                    elif is_cipo and obj_status == "WARNING":
                        marker_color = QColor(255, 170, 40)
                    elif is_cipo or in_path:
                        marker_color = QColor(255, 200, 0)
                    else:
                        marker_color = QColor(0, 220, 255)

                    sprite = sprite_map.get(oi)
                    thl = self._car_hl
                    thw = hw_from_pixmap(sprite, thl) if sprite is not None else self._car_hw

                    if self.cinematic_road:
                        self._draw_car_shadow(painter, x_3d, z_3d, thw, thl, w, h)

                    # Elevated rear/side volume (same language as ego / reference)
                    t_fl, t_fr, t_rr, t_rl = self._elevated_car_quad(
                        x_3d, z_3d, thw, thl, self._car_roof, w, h
                    )

                    drawn = False
                    if sprite is not None and all(p is not None for p in [t_fl, t_fr, t_rr, t_rl]):
                        drawn = self._draw_car_pixmap(painter, sprite, t_fl, t_fr, t_rr, t_rl)

                    if not drawn:
                        hw_f, hl_f = 0.70, 1.40
                        mini_body = [
                            (x_3d - 0.55, z_3d + hl_f),
                            (x_3d + 0.55, z_3d + hl_f),
                            (x_3d + hw_f, z_3d + hl_f * 0.5),
                            (x_3d + hw_f, z_3d - hl_f * 0.5),
                            (x_3d + 0.50, z_3d - hl_f),
                            (x_3d - 0.50, z_3d - hl_f),
                            (x_3d - hw_f, z_3d - hl_f * 0.5),
                            (x_3d - hw_f, z_3d + hl_f * 0.5),
                        ]
                        mini_canvas = [self.world_to_canvas_3d(px, py, 0.0, w, h) for px, py in mini_body]
                        mini_canvas = [p for p in mini_canvas if p is not None]
                        if len(mini_canvas) >= 4:
                            body_color = QColor(marker_color.red(), marker_color.green(),
                                                marker_color.blue(), 160)
                            painter.setBrush(QBrush(body_color))
                            painter.setPen(QPen(marker_color, 1.5, Qt.SolidLine))
                            mini_path = QPainterPath()
                            mini_path.moveTo(mini_canvas[0])
                            for mp in mini_canvas[1:]:
                                mini_path.lineTo(mp)
                            mini_path.closeSubpath()
                            painter.drawPath(mini_path)

                    track_id = obj.get('track_id', -1)
                    id_str = f"#{track_id:02d} " if track_id > 0 else ""
                    label_str = f"{id_str}{obj['label'].upper()} {dist_z:.1f}m"

                    badge_w = len(label_str) * 5 + 10
                    badge_x = int(pt_obj.x() + thl * 8 + 4)
                    badge_y = int(pt_obj.y() - 8)
                    painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
                    painter.setPen(Qt.NoPen)
                    painter.drawRoundedRect(badge_x, badge_y, badge_w, 16, 4, 4)

                    painter.setFont(QFont("Inter", 7, QFont.Bold))
                    painter.setPen(QPen(marker_color.lighter(130)))
                    painter.drawText(badge_x + 4, badge_y + 11, label_str)

            # ── 9. Top-Left Telemetry Overlay Badge ──────────────────────────
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont("Inter", 10, QFont.Bold))
            painter.drawText(15, 28, "3D ROTATABLE BEV TELEMETRY CANVAS")

            badge_color = QColor(220, 40, 40) if self.cipo_status == "DANGER" else (QColor(255, 180, 0) if self.cipo_status == "WARNING" else QColor(0, 200, 100))
            painter.setBrush(QBrush(badge_color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(15, 38, 110, 22, 4, 4)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont("Inter", 8, QFont.Bold))
            painter.drawText(23, 53, f"CIPO: {self.cipo_status}")

            # ── 10. Bottom-Right Controls Telemetry Legend ───────────────────
            painter.setFont(QFont("Inter", 8))
            painter.setPen(QPen(self.grid_text_color if not self.cinematic_road else QColor(150, 165, 185)))
            mode = "Cinematic" if self.cinematic_road else "Grid"
            painter.drawText(
                w - 280, h - 15,
                f"Pitch: {self.pitch_deg:.0f}° | Yaw: {self.yaw_deg:.0f}° | {mode}"
            )

        finally:
            painter.end()
