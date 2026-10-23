import numpy as np
from src.inference.postprocess import ANCHOR_Y_STEPS

class EKFLaneTracker:
    """
    Extended Kalman Filter for a single 3D Lane Track.
    
    State vector x_k in R^8:
      x[0] = a0 : Lateral offset at Y=0 (meters)
      x[1] = a1 : Heading angle tangent / slope error
      x[2] = a2 : Road curvature (1/m)
      x[3] = a3 : Curvature change rate (1/m^2)
      x[4] = a0_dot : Lateral drift velocity (m/s)
      x[5] = b0 : Vertical height offset at Y=0 (meters)
      x[6] = b1 : Road pitch slope
      x[7] = b2 : Vertical road grade curvature

    3D Polynomial Model:
      X(Y) = a0 + a1*Y + a2*Y^2 + a3*Y^3
      Z(Y) = b0 + b1*Y + b2*Y^2
    """
    def __init__(self, init_points_3d, track_id=0, confirm_hits=None):
        from src.inference import lane_filter_config as cfg

        self.track_id = track_id
        self.hits = 1
        self.misses = 0
        self.age = 1
        self.confirm_hits = cfg.EKF_CONFIRM_HITS if confirm_hits is None else confirm_hits
        self.is_confirmed = False

        # Fit initial 3D polynomial coefficients from observed 3D points
        # points_3d: N x 3 array of (X, Y, Z)
        ys = init_points_3d[:, 1]
        xs = init_points_3d[:, 0]
        zs = init_points_3d[:, 2]

        try:
            poly_x = np.polyfit(ys, xs, deg=3) # [a3, a2, a1, a0]
            a3, a2, a1, a0 = poly_x[0], poly_x[1], poly_x[2], poly_x[3]
        except Exception:
            a3, a2, a1, a0 = 0.0, 0.0, 0.0, np.mean(xs)

        try:
            poly_z = np.polyfit(ys, zs, deg=2) # [b2, b1, b0]
            b2, b1, b0 = poly_z[0], poly_z[1], poly_z[2]
        except Exception:
            b2, b1, b0 = 0.0, 0.0, np.mean(zs)

        # State vector x_k
        self.x = np.array([a0, a1, a2, a3, 0.0, b0, b1, b2], dtype=np.float64)

        # State Covariance matrix P_k
        self.P = np.diag([0.5, 0.05, 1e-3, 1e-4, 0.5, 0.2, 0.02, 1e-3]).astype(np.float64)

        # Process Noise Covariance Q_k
        self.Q_base = np.diag([1e-2, 1e-3, 1e-5, 1e-6, 1e-1, 1e-2, 1e-3, 1e-5]).astype(np.float64)

        # Measurement Noise R_k for a single 3D point (X, Z)
        self.R_std_x = 0.3 # meters
        self.R_std_z = 0.2 # meters

    def predict(self, dt=0.033, speed_mps=None):
        """Predict state forward by dt seconds (a0_dot only).

        HUD ego-speed is applied later via apply_ego_coast() on unmatched
        tracks only. Detections are already in the current vehicle frame, so
        shifting every track by v*dt double-counts motion and breaks association.
        """
        dt = float(dt)
        F = np.eye(8, dtype=np.float64)
        F[0, 4] = dt
        self.x = F @ self.x
        Q = self.Q_base * dt
        self.P = F @ self.P @ F.T + Q
        self.age += 1
        self.misses += 1
        return self.x

    def apply_ego_coast(self, dt, speed_mps):
        """Shift the polynomial along Y after a miss: ds = v * dt."""
        if speed_mps is None or float(speed_mps) <= 0.5:
            return
        ds = float(speed_mps) * float(dt)
        if ds <= 0.0:
            return
        a0, a1, a2, a3, a0d, b0, b1, b2 = self.x
        ds2, ds3 = ds * ds, ds * ds * ds
        self.x = np.array(
            [
                a0 + a1 * ds + a2 * ds2 + a3 * ds3,
                a1 + 2.0 * a2 * ds + 3.0 * a3 * ds2,
                a2 + 3.0 * a3 * ds,
                a3,
                a0d,
                b0 + b1 * ds + b2 * ds2,
                b1 + 2.0 * b2 * ds,
                b2,
            ],
            dtype=np.float64,
        )
        # Extra process noise: coasting is less certain than a measurement
        self.P = self.P + np.diag([1e-2, 2e-3, 2e-5, 2e-6, 0.0, 5e-3, 1e-3, 1e-5])

    def update(self, points_3d, confirm_hits=None):
        """Update state using observed 3D points (N x 3: X, Y, Z)."""
        if confirm_hits is not None:
            self.confirm_hits = confirm_hits
        if len(points_3d) == 0:
            return

        ys = points_3d[:, 1]
        xs_obs = points_3d[:, 0]
        zs_obs = points_3d[:, 2]

        num_pts = len(ys)

        # Construct Measurement Jacobian H (2*num_pts x 8) and Innovation y
        H_list = []
        y_list = []
        R_diag = []

        for i in range(num_pts):
            y_i = ys[i]
            x_pred = self.x[0] + self.x[1]*y_i + self.x[2]*(y_i**2) + self.x[3]*(y_i**3)
            z_pred = self.x[5] + self.x[6]*y_i + self.x[7]*(y_i**2)

            # Jacobian row for X measurement
            hx = [1.0, y_i, y_i**2, y_i**3, 0.0, 0.0, 0.0, 0.0]
            # Jacobian row for Z measurement
            hz = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, y_i, y_i**2]

            H_list.append(hx)
            H_list.append(hz)

            y_list.append(xs_obs[i] - x_pred)
            y_list.append(zs_obs[i] - z_pred)

            R_diag.append(self.R_std_x**2)
            R_diag.append(self.R_std_z**2)

        H = np.array(H_list, dtype=np.float64)
        y_innov = np.array(y_list, dtype=np.float64)
        R = np.diag(R_diag).astype(np.float64)

        # Innovation Covariance S = H P H^T + R
        S = H @ self.P @ H.T + R

        try:
            # Kalman Gain K = P H^T S^-1
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y_innov
            I = np.eye(8, dtype=np.float64)
            self.P = (I - K @ H) @ self.P
        except np.linalg.LinAlgError:
            pass

        self.hits += 1
        self.misses = 0
        if self.hits >= self.confirm_hits:
            self.is_confirmed = True

    def get_lane_points(self, y_steps=ANCHOR_Y_STEPS):
        """Evaluate current EKF state to generate smooth 3D lane points along Y steps."""
        a0, a1, a2, a3, _, b0, b1, b2 = self.x
        pts_3d = []
        for y in y_steps:
            x_val = a0 + a1 * y + a2 * (y**2) + a3 * (y**3)
            z_val = b0 + b1 * y + b2 * (y**2)
            pts_3d.append([x_val, y, z_val])
        return np.array(pts_3d, dtype=np.float32)
