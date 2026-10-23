import numpy as np

from src.inference import lane_filter_config as cfg

ANCHOR_LEN = 20
ANCHOR_Y_STEPS = np.array(
    [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    dtype=np.float64,
)


def active_y_steps(max_y_m=None):
    """Y samples used for geometry / tracking / draw (≤ MAX_LANE_Y_M)."""
    max_y = float(cfg.MAX_LANE_Y_M if max_y_m is None else max_y_m)
    return ANCHOR_Y_STEPS[ANCHOR_Y_STEPS <= max_y + 1e-6]


def clip_proposal_max_y(proposal, max_y_m=None):
    """Zero visibility beyond max_y so far anchors are ignored (model still predicts them)."""
    max_y = float(cfg.MAX_LANE_Y_M if max_y_m is None else max_y_m)
    if isinstance(proposal, np.ndarray) and proposal.ndim == 2 and proposal.shape[1] == 3:
        return proposal[proposal[:, 1] <= max_y + 1e-6]
    out = np.asarray(proposal, dtype=np.float64).copy()
    vis = out[5 + 2 * ANCHOR_LEN : 5 + 3 * ANCHOR_LEN]
    vis[ANCHOR_Y_STEPS > max_y + 1e-6] = 0.0
    out[5 + 2 * ANCHOR_LEN : 5 + 3 * ANCHOR_LEN] = vis
    return out


def softmax(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _lane_geometry_ok(
    proposal,
    min_visible_points=None,
    max_abs_mean_x=None,
    min_abs_mean_x=None,
    max_lateral_jump=None,
    max_abs_slope=None,
):
    """Reject short, far, near-center ghosts, zig-zag, or steep proposals."""
    min_visible_points = cfg.MIN_VISIBLE_POINTS if min_visible_points is None else min_visible_points
    max_abs_mean_x = cfg.MAX_ABS_MEAN_X_M if max_abs_mean_x is None else max_abs_mean_x
    min_abs_mean_x = (
        getattr(cfg, "MIN_ABS_MEAN_X_M", 0.0) if min_abs_mean_x is None else min_abs_mean_x
    )
    max_lateral_jump = cfg.MAX_LATERAL_JUMP_M if max_lateral_jump is None else max_lateral_jump
    max_abs_slope = cfg.MAX_ABS_SLOPE if max_abs_slope is None else max_abs_slope

    proposal = clip_proposal_max_y(proposal)
    xs = proposal[5 : 5 + ANCHOR_LEN].astype(np.float64)
    vis = proposal[5 + 2 * ANCHOR_LEN : 5 + 3 * ANCHOR_LEN] > 0
    if int(vis.sum()) < min_visible_points:
        return False

    xs_v = xs[vis]
    ys_v = ANCHOR_Y_STEPS[vis]
    # Match lane_mean_x(near_only=True): far-curve bias must not hide a near
    # shoulder/ghost that sits outside the lateral gate.
    near = vis & (ANCHOR_Y_STEPS <= 40.0)
    if int(near.sum()) >= 2:
        mean_x = abs(float(np.mean(xs[near])))
    else:
        mean_x = abs(float(np.mean(xs_v)))
    if mean_x > max_abs_mean_x:
        return False
    # Ghost centerlines often sit near X≈0 inside the ego lane (not real paint).
    if mean_x < float(min_abs_mean_x):
        return False

    if len(xs_v) >= 2:
        jumps = np.abs(np.diff(xs_v))
        if float(np.max(jumps)) > max_lateral_jump:
            return False
        dy = float(ys_v[-1] - ys_v[0])
        if abs(dy) > 1e-3:
            slope = abs(float(xs_v[-1] - xs_v[0]) / dy)
            if slope > max_abs_slope:
                return False
    return True


def postprocess_onnx_output(
    reg_proposals,
    conf_threshold=None,
    nms_thres=None,
    max_lanes=None,
    apply_geometry_filter=True,
):
    """
    Decode Anchor3DLane raw proposals -> filtered lanes + scores.

    Defaults come from src.inference.lane_filter_config (easy to tune).
    """
    conf_threshold = cfg.CONF_THRESHOLD if conf_threshold is None else conf_threshold
    nms_thres = cfg.NMS_THRES_M if nms_thres is None else nms_thres
    max_lanes = cfg.MAX_LANES if max_lanes is None else max_lanes

    proposals = reg_proposals[0].copy()
    # Undo the network's lateral scale bias before anything reads X, so every
    # gate, width and draw downstream is in real metres.
    x_scale = float(getattr(cfg, "LANE_X_SCALE", 1.0))
    if abs(x_scale - 1.0) > 1e-6:
        proposals[:, 5 : 5 + ANCHOR_LEN] /= x_scale
    logits = softmax(proposals[:, 5 + 3 * ANCHOR_LEN :], axis=1)
    score = 1 - logits[:, 0]
    proposals[:, 1] = score
    proposals[:, 5 + 3 * ANCHOR_LEN :] = logits

    keep = score > conf_threshold
    proposals, kept_scores = proposals[keep], score[keep]
    if proposals.shape[0] == 0:
        return proposals, None

    # Drop far anchors before geometry / NMS / ranking.
    proposals = np.stack([clip_proposal_max_y(p) for p in proposals], axis=0)

    if apply_geometry_filter:
        geo_keep = np.array([_lane_geometry_ok(p) for p in proposals], dtype=bool)
        proposals, kept_scores = proposals[geo_keep], kept_scores[geo_keep]
        if proposals.shape[0] == 0:
            return proposals, None

    order = np.argsort(-kept_scores)
    proposals, kept_scores = proposals[order], kept_scores[order]

    # Greedy duplicate suppression by mean lateral distance (meters)
    lane_xs_all = proposals[:, 5 : 5 + ANCHOR_LEN]
    vis_all = proposals[:, 5 + 2 * ANCHOR_LEN : 5 + 3 * ANCHOR_LEN] > 0

    suppressed = np.zeros(len(proposals), dtype=bool)
    final_idx = []
    for i in range(len(proposals)):
        if suppressed[i]:
            continue
        final_idx.append(i)
        for j in range(i + 1, len(proposals)):
            if suppressed[j]:
                continue
            common = vis_all[i] & vis_all[j]
            if common.sum() < 2:
                continue
            dist = np.mean(np.abs(lane_xs_all[i][common] - lane_xs_all[j][common]))
            if dist < nms_thres:
                suppressed[j] = True

    proposals = proposals[final_idx]
    kept_scores = kept_scores[final_idx]

    # Prefer near-ego lanes when capping: re-rank by |mean_x| among survivors,
    # but break ties with score so strong far lanes can still win a slot.
    if max_lanes and len(proposals) > max_lanes:
        mean_x = []
        for p in proposals:
            vis = p[5 + 2 * ANCHOR_LEN : 5 + 3 * ANCHOR_LEN] > 0
            xs = p[5 : 5 + ANCHOR_LEN][vis]
            mean_x.append(float(np.mean(xs)) if vis.sum() else 1e9)
        mean_x = np.asarray(mean_x, dtype=np.float64)
        # lower |mean_x| first, then higher score
        rank = np.lexsort((-kept_scores, np.abs(mean_x)))
        pick = rank[:max_lanes]
        # keep score-descending order for callers
        pick = pick[np.argsort(-kept_scores[pick])]
        proposals = proposals[pick]
        kept_scores = kept_scores[pick]

    return proposals, kept_scores


def _projective_transformation(P, x, y, z):
    ones = np.ones((1, len(z)))
    coords = np.vstack((x, y, z, ones))
    trans = P @ coords
    u = trans[0, :] / (trans[2, :] + 1e-8)
    v = trans[1, :] / (trans[2, :] + 1e-8)
    return u, v


def decode_lane_pixels(proposal, P_matrix, flat_ground=False, max_y_m=None, min_y_m=None):
    """
    Project a 3D lane proposal into model-space pixels (480x360).

    flat_ground=True forces Z=0 (same ground-plane style as BEV), which
    removes wavy/floating height noise in the front-camera overlay.
    """
    max_y = float(cfg.MAX_LANE_Y_M if max_y_m is None else max_y_m)
    min_y = None if min_y_m is None else float(min_y_m)
    if isinstance(proposal, np.ndarray) and proposal.ndim == 2 and proposal.shape[1] == 3:
        xs = proposal[:, 0].astype(np.float64)
        ys = proposal[:, 1].astype(np.float64)
        zs = proposal[:, 2].astype(np.float64)
        keep = ys <= max_y + 1e-6
        if min_y is not None:
            keep = keep & (ys >= min_y - 1e-6)
        xs, ys, zs = xs[keep], ys[keep], zs[keep]
    else:
        proposal = clip_proposal_max_y(proposal, max_y)
        lane_xs = proposal[5 : 5 + ANCHOR_LEN]
        lane_zs = proposal[5 + ANCHOR_LEN : 5 + 2 * ANCHOR_LEN]
        lane_vis = proposal[5 + 2 * ANCHOR_LEN : 5 + 3 * ANCHOR_LEN] > 0
        if lane_vis.sum() < 2:
            return []
        xs = lane_xs[lane_vis].astype(np.float64)
        ys = ANCHOR_Y_STEPS[lane_vis]
        zs = lane_zs[lane_vis].astype(np.float64)
        if min_y is not None:
            keep = ys >= min_y - 1e-6
            xs, ys, zs = xs[keep], ys[keep], zs[keep]

    if len(xs) < 2:
        return []

    if flat_ground:
        zs = np.zeros_like(xs)

    u, v = _projective_transformation(P_matrix, xs, ys, zs)
    return list(zip(u, v))
