import numpy as np
from src.tracking.ekf_lane_tracker import EKFLaneTracker
from src.inference.postprocess import ANCHOR_Y_STEPS, active_y_steps, clip_proposal_max_y
from src.inference import lane_filter_config as cfg


class LaneTrackerManager:
    """
    Multi-Lane EKF Tracking Manager.
    Associates raw Anchor3DLane detections with active EKF tracks across frames.
    """

    def __init__(
        self,
        max_missed_frames=None,
        dist_threshold=None,
        confirm_hits=None,
        require_confirmed=None,
    ):
        self.trackers = []
        self.next_track_id = 1
        self.max_missed_frames = (
            cfg.EKF_MAX_MISSED_FRAMES if max_missed_frames is None else max_missed_frames
        )
        self.dist_threshold = (
            cfg.EKF_DIST_THRESHOLD_M if dist_threshold is None else dist_threshold
        )
        self.confirm_hits = cfg.EKF_CONFIRM_HITS if confirm_hits is None else confirm_hits
        self.require_confirmed = (
            cfg.EKF_REQUIRE_CONFIRMED if require_confirmed is None else require_confirmed
        )

    def _y_steps(self):
        return active_y_steps()

    def _proposal_to_pts(self, prop):
        """Visible 3D points, clipped to MAX_LANE_Y_M."""
        y_steps = self._y_steps()
        if isinstance(prop, np.ndarray) and prop.ndim == 1:
            prop = clip_proposal_max_y(prop)
            xs = prop[5 : 5 + len(ANCHOR_Y_STEPS)]
            zs = prop[5 + len(ANCHOR_Y_STEPS) : 5 + 2 * len(ANCHOR_Y_STEPS)]
            vis = prop[5 + 2 * len(ANCHOR_Y_STEPS) : 5 + 3 * len(ANCHOR_Y_STEPS)] > 0
            keep = vis & (ANCHOR_Y_STEPS <= float(y_steps[-1]) + 1e-6)
            if int(keep.sum()) < 2:
                return None
            return np.column_stack((xs[keep], ANCHOR_Y_STEPS[keep], zs[keep])).astype(np.float64)
        pts = np.asarray(prop, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return None
        pts = pts[pts[:, 1] <= float(y_steps[-1]) + 1e-6]
        return pts if len(pts) >= 2 else None

    def _compute_distance(self, tracker_pts, proposal_pts):
        """Mean BEV lateral distance on overlapping near-field Y samples."""
        # Align by nearest Y to avoid length mismatch after Y-clip.
        n = min(len(tracker_pts), len(proposal_pts))
        if n < 2:
            return 1e6
        return float(np.mean(np.abs(tracker_pts[:n, 0] - proposal_pts[:n, 0])))

    def update(self, detected_proposals, dt=0.033, speed_mps=None):
        """
        Update all EKF trackers with new detected proposals.
        Returns list of confirmed smoothed 3D lane proposals.
        """
        y_steps = self._y_steps()
        for trk in self.trackers:
            trk.predict(dt=dt)

        if detected_proposals is None or len(detected_proposals) == 0:
            proposal_pts_list = []
        else:
            proposal_pts_list = []
            for prop in detected_proposals:
                pts = self._proposal_to_pts(prop)
                if pts is not None:
                    proposal_pts_list.append(pts)

        num_trackers = len(self.trackers)
        num_proposals = len(proposal_pts_list)

        matched_trackers = set()
        matched_proposals = set()

        if num_trackers > 0 and num_proposals > 0:
            cost_matrix = np.zeros((num_trackers, num_proposals), dtype=np.float32)
            for i, trk in enumerate(self.trackers):
                trk_pts = trk.get_lane_points(y_steps)
                for j, prop_pts in enumerate(proposal_pts_list):
                    cost_matrix[i, j] = self._compute_distance(trk_pts, prop_pts)

            for _ in range(min(num_trackers, num_proposals)):
                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                min_dist = cost_matrix[min_idx]

                if min_dist > self.dist_threshold:
                    break

                trk_idx, prop_idx = min_idx
                if trk_idx not in matched_trackers and prop_idx not in matched_proposals:
                    matched_trackers.add(trk_idx)
                    matched_proposals.add(prop_idx)
                    self.trackers[trk_idx].update(
                        proposal_pts_list[prop_idx], confirm_hits=self.confirm_hits
                    )

                cost_matrix[trk_idx, :] = 1e6
                cost_matrix[:, prop_idx] = 1e6

        # Coast unmatched tracks with HUD speed (miss only — not on live paint)
        for i, trk in enumerate(self.trackers):
            if i not in matched_trackers:
                trk.apply_ego_coast(dt, speed_mps)

        for j in range(num_proposals):
            if j not in matched_proposals:
                new_trk = EKFLaneTracker(
                    proposal_pts_list[j],
                    track_id=self.next_track_id,
                    confirm_hits=self.confirm_hits,
                )
                self.next_track_id += 1
                self.trackers.append(new_trk)

        self.trackers = [trk for trk in self.trackers if trk.misses <= self.max_missed_frames]

        smoothed_lanes = []
        for trk in self.trackers:
            if self.require_confirmed:
                if trk.is_confirmed:
                    smoothed_lanes.append(trk.get_lane_points(y_steps))
            elif trk.is_confirmed or trk.hits >= max(1, self.confirm_hits // 2):
                smoothed_lanes.append(trk.get_lane_points(y_steps))

        return smoothed_lanes
