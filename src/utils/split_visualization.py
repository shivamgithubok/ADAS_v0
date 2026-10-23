import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.visualization import draw_bev, world_to_canvas
from src.inference.postprocess import ANCHOR_Y_STEPS, decode_lane_pixels
from src.inference import lane_filter_config as lane_cfg
from src.utils.drivable_area import (
    extract_ego_corridor_3d,
    force_corridor_fixed_width,
    get_ego_corridor_2d_pixels,
    get_ego_corridor_sides_2d,
    fill_missing_lane_gaps,
    find_ego_lanes,
    parse_lane_components,
    STANDARD_LANE_WIDTH,
)
from src.utils.paint_snap import LaneDrawStabilizer, lane_paint_mask
from src.utils.draw_3d_box import draw_3d_wireframe_box

# Front-view overlay (BGR): lime / amber / red by CIPO range.
CORRIDOR_SAFE_FAR = (20, 175, 0)
CORRIDOR_SAFE_NEAR = (45, 245, 35)
CORRIDOR_WARN_FAR = (0, 155, 220)
CORRIDOR_WARN_NEAR = (0, 210, 255)
CORRIDOR_DANGER_FAR = (0, 25, 170)
CORRIDOR_DANGER_NEAR = (25, 40, 255)
EGO_LANE_COLOR = (0, 220, 255)
ADJ_LANE_COLOR = (145, 175, 205)
CORRIDOR_FILL_ALPHA = 0.46
CORRIDOR_INSET_FRAC = 0.12


# ─────────────────────────────────────────────────────────────────────────────
# PROFESSIONAL 2D DETECTION BOX RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _draw_pro_detection_box(img, overlay, x1, y1, x2, y2, track_id, label, dist_m, color, is_cipo=False, lane_index=None):
    """
    Thin 2D detection box:
      • 1px outline only (no fill / glow over the vehicle)
      • Dark translucent label chip background (drawn on overlay)
      • Vector line work & crisp drop-shadowed text (drawn on img)
    """
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    w_box = x2 - x1
    h_box = y2 - y1
    c_len = max(8, min(16, w_box // 5, h_box // 5))
    c_thick = 1

    corners = [
        (x1, y1,  1,  1),
        (x2, y1, -1,  1),
        (x1, y2,  1, -1),
        (x2, y2, -1, -1),
    ]
    for (cx, cy, dx, dy) in corners:
        cv2.line(img, (cx, cy), (cx + dx * c_len, cy), color, c_thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * c_len), color, c_thick, cv2.LINE_AA)

    lbl_lower = (label or "").lower()
    if "truck" in lbl_lower or "bus" in lbl_lower:
        cls_code = "TRUCK"
        cls_icon = "▲"
    elif "motorcycle" in lbl_lower or "bike" in lbl_lower:
        cls_code = "MOTO"
        cls_icon = "◈"
    else:
        cls_code = "CAR"
        cls_icon = "●"

    id_str = f"#{track_id:02d}" if (track_id is not None and track_id > 0) else "#--"
    row1 = f" {id_str}  {cls_icon} {cls_code} "
    if lane_index is None or abs(int(lane_index)) > 2:
        row2 = f"  {dist_m:.1f} m  "
    else:
        li = int(lane_index)
        tag = "EGO" if li == 0 else f"L{li:+d}"
        row2 = f"  {dist_m:.1f} m {tag} "

    font_r1 = cv2.FONT_HERSHEY_DUPLEX
    font_r2 = cv2.FONT_HERSHEY_SIMPLEX
    fs1, th1 = 0.38, 1
    fs2, th2 = 0.42, 1

    (tw1, th1_px), _ = cv2.getTextSize(row1, font_r1, fs1, th1)
    (tw2, th2_px), _ = cv2.getTextSize(row2, font_r2, fs2, th2)

    chip_w = max(tw1, tw2) + 12
    chip_h = th1_px + th2_px + 18

    chip_x = x1
    chip_y = y1 - chip_h - 3
    if chip_y < 2:
        chip_y = y2 + 3

    cv2.rectangle(overlay, (chip_x, chip_y),
                  (chip_x + chip_w, chip_y + chip_h), (10, 10, 16), -1)

    cv2.rectangle(img, (chip_x, chip_y),
                  (chip_x + chip_w, chip_y + 3), color, -1)

    y_r1 = chip_y + 3 + th1_px + 3
    cv2.putText(img, row1, (chip_x + 6 + 1, y_r1 + 1), font_r1, fs1, (0, 0, 0), th1, cv2.LINE_AA)
    cv2.putText(img, row1, (chip_x + 6, y_r1), font_r1, fs1, (220, 220, 220), th1, cv2.LINE_AA)

    y_r2 = y_r1 + th2_px + 5
    cv2.putText(img, row2, (chip_x + 6 + 1, y_r2 + 1), font_r2, fs2, (0, 0, 0), th2, cv2.LINE_AA)
    cv2.putText(img, row2, (chip_x + 6, y_r2), font_r2, fs2, color, th2, cv2.LINE_AA)



# ─────────────────────────────────────────────────────────────────────────────

def draw_futuristic_corner_bbox(img, pt1, pt2, color, thickness=1, corner_len=14):
    """Renders futuristic cybernetic corner brackets around detected vehicle bounding boxes."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    c_len = min(corner_len, w // 4, h // 4)

    cv2.line(img, (x1, y1), (x1 + c_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + c_len), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - c_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + c_len), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + c_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 + c_len), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - c_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - c_len), color, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# LANE DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

ANCHOR_LEN = 20


def _get_lane_mean_x(lane, anchor_len=ANCHOR_LEN):
    if lane is None:
        return 0.0
    xs, ys, zs, vis = parse_lane_components(lane, anchor_len)
    return float(np.mean(xs[vis])) if vis.sum() >= 2 else 0.0


def _lerp_bgr(a, b, t):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _fill_corridor_gradient(overlay, left_pts, right_pts, danger=False, warning=False):
    """Lime (safe), amber (CIPO warning), red (CIPO danger)."""
    n = min(len(left_pts), len(right_pts))
    if n < 2:
        return
    if danger:
        far_c, near_c = CORRIDOR_DANGER_FAR, CORRIDOR_DANGER_NEAR
    elif warning:
        far_c, near_c = CORRIDOR_WARN_FAR, CORRIDOR_WARN_NEAR
    else:
        far_c, near_c = CORRIDOR_SAFE_FAR, CORRIDOR_SAFE_NEAR
    for i in range(n - 1):
        t = i / max(1, n - 2)
        color = _lerp_bgr(far_c, near_c, t)
        quad = np.array(
            [left_pts[i], left_pts[i + 1], right_pts[i + 1], right_pts[i]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(overlay, quad, color)


def _draw_lane_line(img, pts, color, thickness):
    """Flat anti-aliased polyline, no glow."""
    if len(pts) < 2:
        return
    arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [arr], False, color, int(thickness), cv2.LINE_AA)


def _lane_stroke_style(role):
    if role in ("left", "right"):
        return EGO_LANE_COLOR, 3
    return ADJ_LANE_COLOR, 1


def _corridor_risk(cipo_obj, objects):
    """CIPO band drives fill: red <~15 m, yellow ~15–30 m, else green."""
    z = None
    status = ""
    if cipo_obj is not None and cipo_obj.get("in_path"):
        z = float(cipo_obj.get("Z_3d", 999.0))
        status = str(cipo_obj.get("status") or "")
    elif objects:
        in_path = [obj for obj in objects if obj.get("in_path")]
        if in_path:
            z = float(min(obj["Z_3d"] for obj in in_path))
    if status == "DANGER" or (z is not None and z < 15.0):
        return True, False
    if status == "WARNING" or (z is not None and z < 30.0):
        return False, True
    return False, False


def _project_lane_draw_points(lane, P_matrix, frame_transform, w_img, h_img, scale_x, scale_y):
    """Ego/adjacent lane pixels in the source frame, same filter as the green lines."""
    if lane is None:
        return []
    pts = decode_lane_pixels(
        lane,
        P_matrix,
        flat_ground=False,
        max_y_m=float(getattr(lane_cfg, "DRAW_LANE_Y_M", lane_cfg.MAX_LANE_Y_M)),
        min_y_m=float(getattr(lane_cfg, "CORRIDOR_Y_START_M", 10.0)),
    )
    model_pts = np.asarray(
        [(u, v) for u, v in pts if 0 <= u < 480 and 0 <= v < 360],
        dtype=np.float64,
    )
    if len(model_pts) == 0:
        return []
    if frame_transform is not None:
        target_pts = np.asarray(frame_transform.model_to_source(model_pts), dtype=np.float64)
    else:
        target_pts = model_pts.copy()
        target_pts[:, 0] *= scale_x
        target_pts[:, 1] *= scale_y
    return [
        (int(round(u)), int(round(v)))
        for u, v in target_pts
        if 0 <= u < w_img and 0 <= v < h_img
    ]


def _inset_polylines(left, right, frac):
    """Pull a left/right pair inward so the fill sits inside the paint."""
    L = np.asarray(left, dtype=np.float64)
    R = np.asarray(right, dtype=np.float64)
    if len(L) < 2 or len(R) < 2:
        return None
    L = L[np.argsort(L[:, 1])]
    R = R[np.argsort(R[:, 1])]
    v0 = max(float(L[0, 1]), float(R[0, 1]))
    v1 = min(float(L[-1, 1]), float(R[-1, 1]))
    if v1 - v0 < 8:
        return None
    vs = np.linspace(v0, v1, 16)
    lu = np.interp(vs, L[:, 1], L[:, 0])
    ru = np.interp(vs, R[:, 1], R[:, 0])
    width = ru - lu
    if np.median(width) < 12:
        return None
    lu = lu + float(frac) * width
    ru = ru - float(frac) * width
    left_pts = [(int(round(u)), int(round(v))) for u, v in zip(lu, vs)]
    right_pts = [(int(round(u)), int(round(v))) for u, v in zip(ru, vs)]
    return left_pts, right_pts


def _lanes_for_stroke(sorted_lanes, ego_left, ego_right):
    """Ego pair plus one neighbor each side. One line per paint stripe.

    With no ego lock, draw nothing new — the stabilizer holds or fades
    the last strokes. Picking closest-left/right here is what jumped the
    overlay across the image when the corridor dropped.
    """
    ego_l_idx = ego_r_idx = None
    for idx, lane in enumerate(sorted_lanes):
        if ego_left is not None and (
            np.array_equal(lane, ego_left)
            or abs(_get_lane_mean_x(lane) - _get_lane_mean_x(ego_left)) < 0.80
        ):
            ego_l_idx = idx
        if ego_right is not None and (
            np.array_equal(lane, ego_right)
            or abs(_get_lane_mean_x(lane) - _get_lane_mean_x(ego_right)) < 0.80
        ):
            ego_r_idx = idx
    if ego_l_idx is None and ego_r_idx is None:
        return []
    chosen = []
    for idx, lane in enumerate(sorted_lanes):
        if idx == ego_l_idx:
            chosen.append((lane, "left"))
        elif idx == ego_r_idx:
            chosen.append((lane, "right"))
        elif ego_l_idx is not None and idx == ego_l_idx - 1:
            chosen.append((lane, "adj_l"))
        elif ego_r_idx is not None and idx == ego_r_idx + 1:
            chosen.append((lane, "adj_r"))
    kept = []
    for lane, role in chosen:
        x = _get_lane_mean_x(lane)
        if any(abs(x - _get_lane_mean_x(other)) < 0.80 for other, _role in kept):
            continue
        kept.append((lane, role))
    return kept


def stabilized_lane_polylines(
    proposals,
    ego_left,
    ego_right,
    P_matrix,
    frame_transform,
    frame,
    ground_calib,
    stabilizer,
    freeze=False,
    reset_draw=False,
):
    """OpenLane-warp projection, then one locked stroke per physical lane."""
    h_img, w_img = frame.shape[:2]
    scale_x = w_img / 480.0
    scale_y = h_img / 360.0
    if stabilizer is None:
        stabilizer = LaneDrawStabilizer()
    if reset_draw:
        stabilizer.reset()
    if freeze:
        drawn = stabilizer.hold()
        left_pts = next((d["pts"] for d in drawn if d["role"] == "left"), None)
        right_pts = next((d["pts"] for d in drawn if d["role"] == "right"), None)
        return drawn, left_pts, right_pts
    if proposals is None:
        return stabilizer.hold() if stabilizer._slots else [], None, None
    sorted_lanes = sorted(proposals, key=lambda lane: _get_lane_mean_x(lane))
    items = []
    for lane, role in _lanes_for_stroke(sorted_lanes, ego_left, ego_right):
        pts = _project_lane_draw_points(
            lane, P_matrix, frame_transform, w_img, h_img, scale_x, scale_y
        )
        if len(pts) < 2:
            continue
        items.append({"x": _get_lane_mean_x(lane), "pts": pts, "role": role})
    if not items:
        drawn = stabilizer.hold()
        left_pts = next((d["pts"] for d in drawn if d["role"] == "left"), None)
        right_pts = next((d["pts"] for d in drawn if d["role"] == "right"), None)
        return drawn, left_pts, right_pts
    paint = lane_paint_mask(frame, ground_calib)
    drawn = stabilizer.update(items, frame, ground_calib, paint)
    left_pts = next((d["pts"] for d in drawn if d["role"] == "left"), None)
    right_pts = next((d["pts"] for d in drawn if d["role"] == "right"), None)
    return drawn, left_pts, right_pts


def _clip_side_to_near_row(pts, v_max):
    """Cut a far-to-near polyline so it does not pass the ego-lane near row."""
    if v_max is None or len(pts) < 2:
        return list(pts)
    out = []
    for u, v in pts:
        if v <= v_max:
            out.append((int(u), int(v)))
            continue
        if out:
            pu, pv = out[-1]
            span = float(v) - float(pv)
            if abs(span) > 1e-3:
                t = (float(v_max) - float(pv)) / span
                t = max(0.0, min(1.0, t))
                out.append((int(round(pu + t * (float(u) - pu))), int(round(v_max))))
        break
    return out



# ─────────────────────────────────────────────────────────────────────────────

def draw_front_view_cipo(
    frame,
    proposals,
    objects,
    cipo_obj,
    P_matrix,
    show_drivable=True,
    ego_left=None,
    ego_right=None,
    frame_transform=None,
    road_state_valid=True,
    left_corridor_3d=None,
    right_corridor_3d=None,
    ground_calib=None,
    lane_stabilizer=None,
    reset_draw=False,
    ego_speed_mps=None,
    show_lanes=True,
):
    """
    Front camera overlay:
      - Lime fill inset from ego paint
      - Optional flat yellow ego / 1px light-brown adjacent strokes
      - Thin 2D detection boxes with ID / class / distance chips
    """
    annotated = frame.copy()
    overlay   = frame.copy()
    h_img, w_img = annotated.shape[:2]

    scale_x = w_img / 480.0
    scale_y = h_img / 360.0

    if proposals is not None:
        proposals = fill_missing_lane_gaps(proposals)

    if road_state_valid and ego_left is None and ego_right is None:
        ego_left, ego_right = find_ego_lanes(proposals) if proposals is not None else (None, None)

    danger, warning = _corridor_risk(cipo_obj, objects)

    # OpenLane warp projection, one locked stroke per lane. The corridor fill
    # uses those same strokes so the band and the green lines stay together.
    lane_strokes, ego_left_pts, ego_right_pts = stabilized_lane_polylines(
        proposals,
        ego_left,
        ego_right,
        P_matrix,
        frame_transform,
        frame,
        ground_calib,
        lane_stabilizer,
        freeze=not road_state_valid,
        reset_draw=reset_draw,
    )

    # ── 1. Lime fill, inset from ego paint so the yellow strokes stay clear. ──
    sides = None
    if show_drivable and ego_left_pts and ego_right_pts:
        sides = _inset_polylines(ego_left_pts, ego_right_pts, CORRIDOR_INSET_FRAC)
    if sides is None and show_drivable and road_state_valid and proposals is not None:
        sides = get_ego_corridor_sides_2d(
            proposals, P_matrix,
            img_size=(480, 360), target_size=(w_img, h_img),
            ego_left=ego_left, ego_right=ego_right,
            model_to_target=(frame_transform.model_to_source if frame_transform is not None else None),
            left_corridor_3d=left_corridor_3d,
            right_corridor_3d=right_corridor_3d,
            image_inset_frac=CORRIDOR_INSET_FRAC,
        )
        if sides is not None:
            near_rows = []
            for lane in (ego_left, ego_right):
                tips = _project_lane_draw_points(
                    lane, P_matrix, frame_transform, w_img, h_img, scale_x, scale_y
                )
                if tips:
                    near_rows.append(max(v for _, v in tips))
            if near_rows:
                near_v = max(near_rows)
                left_c = _clip_side_to_near_row(sides[0], near_v)
                right_c = _clip_side_to_near_row(sides[1], near_v)
                sides = (left_c, right_c) if len(left_c) >= 2 and len(right_c) >= 2 else None
    if sides is not None:
        _fill_corridor_gradient(overlay, sides[0], sides[1], danger=danger, warning=warning)
        cv2.addWeighted(overlay, CORRIDOR_FILL_ALPHA, annotated, 1.0 - CORRIDOR_FILL_ALPHA, 0, annotated)

    # ── 2. Detection boxes: thin outline + dark chip (previous style). ────
    box_overlay = annotated.copy()
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        _draw_pro_detection_box(
            annotated, box_overlay,
            x1, y1, x2, y2,
            track_id=obj.get('track_id', -1),
            label=obj['label'],
            dist_m=obj['Z_3d'],
            color=obj['color'],
            is_cipo=bool(obj.get('is_cipo', False)),
            lane_index=obj.get("lane_index"),
        )
    cv2.addWeighted(box_overlay, 0.35, annotated, 0.65, 0, annotated)

    # ── 3. Lane strokes only when the LANES toggle is on ──────────────────
    if show_lanes:
        for stroke in lane_strokes:
            draw_pts = [(int(round(u)), int(round(v))) for u, v in stroke["pts"]]
            if len(draw_pts) < 2:
                continue
            lane_color, thickness = _lane_stroke_style(stroke.get("role"))
            _draw_lane_line(annotated, draw_pts, lane_color, thickness)

    return annotated



def draw_bev_cipo(
    proposals,
    objects,
    max_z=60.0,
    cipo_status="SAFE",
    left_corridor_3d=None,
    right_corridor_3d=None,
):
    """
    Renders top-down Bird's Eye View (BEV) map showing 3D lane lines, drivable area, and object positions.
    """
    if proposals is not None:
        proposals = fill_missing_lane_gaps(proposals)

    in_path_objs = [obj for obj in objects if obj['in_path']]
    min_dist_in_path = min([obj['Z_3d'] for obj in in_path_objs]) if in_path_objs else 999.0
    status_bev = "DANGER" if min_dist_in_path < 15.0 else "SAFE"

    bev = draw_bev(
        proposals,
        ANCHOR_Y_STEPS,
        cipo_status=status_bev,
        left_corridor_3d=left_corridor_3d,
        right_corridor_3d=right_corridor_3d,
        allow_auto_corridor=False,
    )
    h_bev, w_bev = bev.shape[:2]

    def world_to_bev_px(x, y):
        return world_to_canvas(x, y)

    for obj in objects:
        y_3d = obj['Z_3d']
        li = obj.get("lane_index")
        if li is not None and abs(int(li)) <= 2:
            x_3d = float(li) * STANDARD_LANE_WIDTH + float(obj.get("lane_offset_m", 0.0))
        else:
            x_3d = obj['X_3d']
        if 0 < y_3d <= 80.0:
            px, py = world_to_bev_px(x_3d, y_3d)
            if 0 <= px < w_bev and 0 <= py < h_bev:
                color = obj['color']
                radius = 6 if obj['in_path'] else 4
                cv2.circle(bev, (px, py), radius, color, -1)

                track_id = obj.get('track_id', -1)
                id_str = f"#{track_id} " if track_id > 0 else ""
                label = f"{id_str}{obj['label'].upper()}"
                cv2.putText(bev, label, (px + 8, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return bev


def create_split_window(front_view, bev_view, cipo_obj, fps_val, canvas_size=(720, 1080)):
    """
    Combines Front View, BEV Map, and Telemetry HUD into a unified multi-panel split window.
    """
    target_h, target_w = canvas_size
    hud_height = 80
    main_h = target_h - hud_height

    front_w = int(target_w * 0.6)
    bev_w = target_w - front_w

    resized_front = cv2.resize(front_view, (front_w, main_h))
    resized_bev   = cv2.resize(bev_view,   (bev_w,   main_h))

    top_split = np.hstack((resized_front, resized_bev))

    hud = np.ones((hud_height, target_w, 3), dtype=np.uint8) * 20
    cv2.putText(hud, f"PERFORMANCE: {fps_val:.1f} FPS", (20, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(hud, "ENGINE: TensorRT FP16 + YOLO ByteTrack + ground-plane range", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.putText(hud, "MONITOR: DRIVABLE AREA SAFETY ACTIVE", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)
    cv2.putText(hud, "RULES: <15m RED | 15-30m YELLOW | >30m GREEN", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    canvas = np.vstack((top_split, hud))
    return canvas
