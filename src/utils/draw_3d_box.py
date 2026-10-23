import numpy as np
import cv2

def draw_3d_wireframe_box(img, bbox, distance_z, is_truck=False, color=(0, 0, 255), thickness=1):
    """
    Renders thin-line 3D wireframe boxes in Red (0, 0, 255) using Perspective Scale Ratio (k = Z / (Z + L)).
    """
    x1, y1, x2, y2 = bbox
    w_box = x2 - x1
    h_box = y2 - y1

    if w_box <= 0 or h_box <= 0:
        return

    # Image dimensions and horizon row (v_horizon = 0.40 * height)
    h_img, w_img = img.shape[:2]
    v_horizon = h_img * 0.40

    # Vehicle length L in meters (Car: 4.2m, Truck: 6.5m)
    L = 6.5 if is_truck else 4.2

    # Perspective depth scale ratio k for vehicle length extension
    z_clamped = max(3.0, float(distance_z))
    k = max(0.65, min(0.88, z_clamped / (z_clamped + L)))
    depth_factor = 1.0 - k

    # Lateral shrinking towards box center
    dw = (w_box / 2.0) * depth_factor

    # 1. Front Face (at distance Z): TL, TR, BR, BL
    f_tl = (int(x1), int(y1))
    f_tr = (int(x2), int(y1))
    f_br = (int(x2), int(y2))
    f_bl = (int(x1), int(y2))

    # 2. Rear Face (at distance Z + L) extending towards horizon
    r_x1 = int(x1 + dw)
    r_x2 = int(x2 - dw)
    r_y1 = int(y1 + depth_factor * (v_horizon - y1))
    r_y2 = int(y2 + depth_factor * (v_horizon - y2))

    r_tl = (r_x1, r_y1)
    r_tr = (r_x2, r_y1)
    r_br = (r_x2, r_y2)
    r_bl = (r_x1, r_y2)

    # 3. Semi-transparent top/side face fill overlay (Red tint)
    overlay = img.copy()
    top_poly = np.array([f_tl, f_tr, r_tr, r_tl], dtype=np.int32)
    side_poly = np.array([f_tr, f_br, r_br, r_tr], dtype=np.int32)
    cv2.fillPoly(overlay, [top_poly, side_poly], (0, 0, 255))
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

    # 4. Draw 12 connecting 3D thin wireframe edges (Red color: 0, 0, 255, thickness=1)
    # Front Face (thickness=1)
    cv2.rectangle(img, f_tl, f_br, (0, 0, 255), 1, cv2.LINE_AA)

    # Rear Face (thickness=1)
    cv2.rectangle(img, r_tl, r_br, (0, 0, 255), 1, cv2.LINE_AA)

    # 4 Depth Connecting Lines (Front to Rear)
    cv2.line(img, f_tl, r_tl, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(img, f_tr, r_tr, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(img, f_br, r_br, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(img, f_bl, r_bl, (0, 0, 255), 1, cv2.LINE_AA)

    # 5. Draw subtle small 1px corner vertex dots
    for p in [f_tl, f_tr, f_br, f_bl, r_tl, r_tr, r_br, r_bl]:
        cv2.circle(img, p, 1, (255, 255, 255), -1, cv2.LINE_AA)
