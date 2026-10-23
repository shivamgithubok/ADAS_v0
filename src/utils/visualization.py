import math

import numpy as np
import cv2
from src.utils.drivable_area import extract_ego_corridor_3d

# BEV Canvas dimensions (Height x Width)
BEV_HEIGHT = 700
BEV_WIDTH = 500
BEV_CANVAS_SIZE = (BEV_HEIGHT, BEV_WIDTH)

DISK_R_M = 70.0
RING_M = (20.0, 40.0, 70.0)
TURN_YAW_MAX_DEG = 14.0
TURN_YAW_GAIN = 0.45


def world_to_canvas(x, y, radius=DISK_R_M):
    """Real-world (x=right m, y=forward m) -> BEV pixel. Ego at canvas centre."""
    margin = 28
    usable = min(BEV_WIDTH, BEV_HEIGHT) - 2 * margin
    scale = usable / (2.0 * float(radius))
    cx = BEV_WIDTH // 2
    cy = BEV_HEIGHT // 2
    px = int(cx + float(x) * scale)
    py = int(cy - float(y) * scale)
    return px, py


def _in_forward_disk(x, y, radius=DISK_R_M):
    return float(y) >= 0.0 and (float(x) * float(x) + float(y) * float(y)) <= float(radius) * float(radius)


def _px_per_m(radius=DISK_R_M):
    margin = 28
    usable = min(BEV_WIDTH, BEV_HEIGHT) - 2 * margin
    return usable / (2.0 * float(radius))


def _rotate_pts(pts, deg, origin):
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    ox, oy = origin
    out = []
    for x, y in pts:
        dx, dy = x - ox, y - oy
        out.append((int(ox + c * dx - s * dy), int(oy + s * dx + c * dy)))
    return out


def draw_bev(
    proposals,
    anchor_y_steps,
    anchor_len=20,
    cipo_status="SAFE",
    left_corridor_3d=None,
    right_corridor_3d=None,
    allow_auto_corridor=True,
    yaw_rad=0.0,
    trail_left=None,
    trail_right=None,
    speed_mps=None,
    yaw_deg=None,
    yaw_rate=None,
):
    """
    Heading-up 70 m disk. Live lanes stay in the forward sector. `trail_*` is
    ignored (no memory paint). `yaw_rate` briefly yaws the ego sprite, then
    the caller should let it settle — this draw is stateless per frame.
    """
    canvas = np.full((BEV_HEIGHT, BEV_WIDTH, 3), (18, 18, 20), dtype=np.uint8)
    ego_px, ego_py = world_to_canvas(0.0, 0.0)
    ppm = _px_per_m()

    disk_px = int(DISK_R_M * ppm)
    cv2.circle(canvas, (ego_px, ego_py), disk_px, (106, 106, 110), -1, cv2.LINE_AA)
    for r_m, col in ((20.0, (70, 75, 82)), (40.0, (70, 75, 82)), (70.0, (118, 122, 130))):
        cv2.circle(canvas, (ego_px, ego_py), int(r_m * ppm), col, 1, cv2.LINE_AA)
    cv2.line(canvas, (ego_px, ego_py), (ego_px, ego_py - disk_px), (58, 62, 70), 1, cv2.LINE_AA)

    if left_corridor_3d is None and right_corridor_3d is None and allow_auto_corridor:
        left_3d, right_3d = extract_ego_corridor_3d(proposals, anchor_len)
    else:
        left_3d, right_3d = left_corridor_3d, right_corridor_3d
    if left_3d is not None and right_3d is not None:
        pts_left_bev = [world_to_canvas(x, y) for x, y, *_ in left_3d if _in_forward_disk(x, y)]
        pts_right_bev = [world_to_canvas(x, y) for x, y, *_ in right_3d if _in_forward_disk(x, y)]
        if len(pts_left_bev) >= 2 and len(pts_right_bev) >= 2:
            poly_bev = np.array(pts_left_bev + pts_right_bev[::-1], dtype=np.int32)
            overlay = canvas.copy()
            if cipo_status == "DANGER":
                corridor_color = (25, 40, 255)
            elif cipo_status == "WARNING":
                corridor_color = (0, 210, 255)
            else:
                corridor_color = (45, 245, 35)
            cv2.fillPoly(overlay, [poly_bev], corridor_color)
            cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0, canvas)

    if proposals is not None and len(proposals) > 0:
        for lane in proposals:
            if isinstance(lane, np.ndarray) and lane.ndim == 2 and lane.shape[1] == 3:
                pts_world = [(lane[i, 0], lane[i, 1]) for i in range(len(lane))
                             if _in_forward_disk(lane[i, 0], lane[i, 1])]
            else:
                lane_xs = lane[5:5 + anchor_len]
                lane_vis = lane[5 + 2 * anchor_len:5 + 3 * anchor_len] > 0
                pts_world = [(lane_xs[i], anchor_y_steps[i]) for i in range(anchor_len)
                             if lane_vis[i] and _in_forward_disk(lane_xs[i], anchor_y_steps[i])]

            pts_canvas = [world_to_canvas(wx, wy) for wx, wy in pts_world]
            valid_pts = [(px, py) for px, py in pts_canvas if 0 <= px < BEV_WIDTH and 0 <= py < BEV_HEIGHT]

            if len(valid_pts) > 1:
                mean_x = np.mean([wx for wx, wy in pts_world])
                if abs(mean_x) < 2.0:
                    lane_color, lane_th = (0, 220, 255), 2
                else:
                    lane_color, lane_th = (145, 175, 205), 1
                for i in range(1, len(valid_pts)):
                    cv2.line(canvas, valid_pts[i - 1], valid_pts[i], lane_color, lane_th, cv2.LINE_AA)
                for p in valid_pts[::2]:
                    cv2.circle(canvas, p, 2, (255, 255, 255), -1, cv2.LINE_AA)

    rate = 0.0 if yaw_rate is None else float(yaw_rate)
    flash = float(np.clip(math.degrees(rate) * TURN_YAW_GAIN, -TURN_YAW_MAX_DEG, TURN_YAW_MAX_DEG))
    car_w, car_h = 18, 32
    box = [
        (ego_px - car_w // 2, ego_py - car_h // 2),
        (ego_px + car_w // 2, ego_py - car_h // 2),
        (ego_px + car_w // 2, ego_py + car_h // 2),
        (ego_px - car_w // 2, ego_py + car_h // 2),
    ]
    box = np.array(_rotate_pts(box, flash, (ego_px, ego_py)), dtype=np.int32)
    cv2.fillConvexPoly(canvas, box, (235, 140, 0), cv2.LINE_AA)
    cv2.polylines(canvas, [box], True, (255, 255, 255), 2, cv2.LINE_AA)
    nose = _rotate_pts([(ego_px, ego_py + 8), (ego_px, ego_py - 16)], flash, (ego_px, ego_py))
    cv2.arrowedLine(canvas, nose[0], nose[1], (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)

    cv2.putText(canvas, "BEV  70 m disk", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    spd = "--" if speed_mps is None else f"{speed_mps * 2.23694:.0f} mph"
    cv2.putText(
        canvas,
        f"turn {flash:+.1f} deg   {spd}",
        (15, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 200, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas
