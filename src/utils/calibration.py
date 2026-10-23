import numpy as np

def projection_g2im(cam_pitch, cam_height, K):
    P_g2c = np.array([
        [1, 0, 0, 0],
        [0, np.cos(np.pi/2 + cam_pitch), -np.sin(np.pi/2 + cam_pitch), cam_height],
        [0, np.sin(np.pi/2 + cam_pitch),  np.cos(np.pi/2 + cam_pitch), 0]
    ])
    return K @ P_g2c

def homography_crop_resize(org_hw, crop_y, resize_hw):
    ratio_x = resize_hw[1] / org_hw[1]
    ratio_y = resize_hw[0] / (org_hw[0] - crop_y)
    return np.array([[ratio_x, 0, 0],
                     [0, ratio_y, -ratio_y * crop_y],
                     [0, 0, 1]])

# Intrinsics for 1920x1280 OpenLane-style sensor → resized to 480x360
K = np.array([[2015.0, 0.0, 960.0],
              [0.0, 2015.0, 540.0],
              [0.0, 0.0, 1.0]])

# MUST match Anchor3DLane / OpenLane training frame.
# Do NOT retune pitch for Garmin via P — 3D lane outputs assume this camera.
# Garmin domain gap is handled by HUD crop + ego-pair logic, not extrinsics hacks.
CAM_HEIGHT = 1.5       # meters
CAM_PITCH_DEG = -3   # degrees

OPENLANE_CAM_HEIGHT = CAM_HEIGHT
OPENLANE_CAM_PITCH_DEG = CAM_PITCH_DEG


def make_P_matrix(pitch_deg=None, height_m=None):
    """Build 3x4 ground→model-image projection (480x360)."""
    pitch = CAM_PITCH_DEG if pitch_deg is None else float(pitch_deg)
    height = CAM_HEIGHT if height_m is None else float(height_m)
    P_g2im = projection_g2im(np.radians(pitch), height, K)
    H_crop = homography_crop_resize([1280, 1920], 0, [360, 480])
    return H_crop @ P_g2im


def preset_for_video(video_path):
    """
    Always return OpenLane training extrinsics.

    Changing pitch/height for Garmin breaks P vs model-3D consistency
    (corridor lifts/shears while lane polylines look elsewhere).
    """
    return OPENLANE_CAM_PITCH_DEG, OPENLANE_CAM_HEIGHT


P_g2im = projection_g2im(np.radians(CAM_PITCH_DEG), CAM_HEIGHT, K)
H_crop = homography_crop_resize([1280, 1920], 0, [360, 480])
P_final = make_P_matrix(CAM_PITCH_DEG, CAM_HEIGHT)
