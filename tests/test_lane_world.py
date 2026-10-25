"""Ego-centred lane world, depth-aware object lanes, and side-on vehicle centring."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.postprocess import ANCHOR_Y_STEPS
from src.tracking.lane_world import LaneWorld, ObjectLaneAssigner, boundary_ids
from src.utils.ground_calib import GroundCalibration
from src.utils.ground_gate import measure_ground, vehicle_center_x

DT = 1.0 / 30.0
SPEED = 20.0
W = 3.7


def line(x0, c2=0.0, ys=ANCHOR_Y_STEPS):
    ys = np.asarray(ys, dtype=np.float64)
    xs = x0 + c2 * ys ** 2
    return np.column_stack([xs, ys, np.zeros_like(ys)]).astype(np.float32)


def road(shift=0.0, c2=0.0, keep=None, width=W, extra=()):
    """Boundaries of an equal-width road; the car is `shift` m right of lane 0's centre."""
    xs = [(k + 0.5) * width - shift for k in range(-3, 3)]
    xs = [x for x in xs if abs(x) < 8.0]
    if keep is not None:
        xs = [x for x in xs if keep(x)]
    return [line(x, c2) for x in list(xs) + list(extra)]


def run(world, frames, **kw):
    states = []
    for _ in range(frames):
        states.append(world.update(road(**kw), dt=DT, speed_mps=SPEED))
    return states


def test_stable_lane_locks_and_stays():
    lw = LaneWorld()
    st = run(lw, 60)[-1]
    assert st.valid and st.ego_lane_id == 0 and st.lane_changes == 0
    assert abs(st.offset(-1) + 1.85) < 0.1 and abs(st.offset(1) - 1.85) < 0.1
    assert abs(st.ego_width - W) < 0.15
    assert st.source == "both" and st.confidence > 0.9


def test_only_adjacent_lane_visible_does_not_renumber():
    """Frames where the net sees only the right neighbour's pair must not move the ego lane."""
    lw = LaneWorld()
    run(lw, 40)
    right_only = run(lw, 25, keep=lambda x: x > 0.0)
    st = right_only[-1]
    assert st.valid and st.ego_lane_id == 0 and st.lane_changes == 0
    assert abs(st.offset(1) - 1.85) < 0.15
    assert abs(st.offset(-1) + 1.85) < 0.3
    st = run(lw, 25, keep=lambda x: x < -2.0)[-1]
    assert st.ego_lane_id == 0 and st.lane_changes == 0
    assert abs(st.offset(-1) + 1.85) < 0.3


def test_ghost_line_inside_the_lane_is_not_latched():
    lw = LaneWorld()
    run(lw, 40)
    for i in range(60):
        extra = (-0.9,) if i % 2 == 0 else ()
        st = lw.update(road(extra=extra), dt=DT, speed_mps=SPEED)
    assert st.ego_lane_id == 0 and st.lane_changes == 0
    assert abs(st.offset(-1) + 1.85) < 0.15
    assert abs(st.ego_width - W) < 0.2


def test_short_dropout_coasts_and_resumes_same_lane():
    lw = LaneWorld()
    run(lw, 40)
    for _ in range(30):
        st = lw.update([], dt=DT, speed_mps=SPEED)
    assert st.valid and st.held and st.ego_lane_id == 0
    st = run(lw, 10)[-1]
    assert st.ego_lane_id == 0 and st.lane_changes == 0 and not st.held


def test_genuine_lane_change_is_recognised_once():
    lw = LaneWorld()
    run(lw, 30)
    changes = []
    for d in np.linspace(0.0, W, 90):
        st = lw.update(road(shift=float(d)), dt=DT, speed_mps=SPEED)
        if st.event:
            changes.append(st.event)
    st = run(lw, 30, shift=W)[-1]
    assert st.ego_lane_id == 1 and st.lane_changes == 1
    assert sum("lane change right" in e for e in changes) == 1
    assert abs(st.offset(-1) + 1.85) < 0.15 and abs(st.offset(1) - 1.85) < 0.15


def test_lane_change_and_back():
    lw = LaneWorld()
    run(lw, 30)
    for d in np.linspace(0.0, W, 90):
        lw.update(road(shift=float(d)), dt=DT, speed_mps=SPEED)
    for d in np.linspace(W, 0.0, 90):
        st = lw.update(road(shift=float(d)), dt=DT, speed_mps=SPEED)
    st = run(lw, 20)[-1]
    assert st.ego_lane_id == 0 and st.lane_changes == 2


def test_weaving_on_the_line_is_not_a_lane_change():
    lw = LaneWorld()
    run(lw, 30)
    for i in range(150):
        shift = 1.85 + 0.2 * np.sin(i * 0.3)
        st = lw.update(road(shift=float(shift)), dt=DT, speed_mps=SPEED)
    assert st.lane_changes == 0 and st.ego_lane_id == 0


def test_unequal_lane_widths_are_kept():
    lw = LaneWorld()
    xs = [-6.0, -2.1, 2.0, 5.1]
    for _ in range(60):
        st = lw.update([line(x) for x in xs], dt=DT, speed_mps=SPEED)
    assert abs(st.ego_width - 4.1) < 0.15
    assert abs((st.offset(-1) - st.offset(-2)) - 3.9) < 0.2
    assert abs((st.offset(2) - st.offset(1)) - 3.1) < 0.2


def test_lane_of_is_evaluated_at_the_object_range_on_a_curve():
    c2 = 0.002
    lw = LaneWorld()
    st = run(lw, 90, c2=c2)[-1]
    y = 40.0
    bend = c2 * y * y
    # Right-adjacent lane centre at 40 m sits 3.2 m further right on this curve.
    assert st.lane_of(W + bend, y).index == 1
    assert st.lane_of(bend, y).index == 0
    assert st.lane_of(-W + bend, y).index == -1
    # Judged at the bumper instead, the ego-lane car would read two lanes right.
    assert st.lane_of(bend, 0.0).index != 0


def test_boundary_ids_survive_a_lane_change():
    a = boundary_ids(0)
    b = boundary_ids(1)
    # The right ego boundary before the change is the left one after it.
    assert a[1] == b[-1]
    assert a[2] == b[1]


# ------------------------------------------------------------------ objects
def _stable_world(**kw):
    lw = LaneWorld()
    return lw, run(lw, 60, **kw)[-1]


def test_adjacent_car_is_assigned_to_its_physical_lane():
    _lw, st = _stable_world()
    asg = ObjectLaneAssigner()
    assert asg.update(1, -3.6, 12.0, st)[0] == -1
    assert asg.update(2, 0.3, 25.0, st)[0] == 0
    assert asg.update(3, 3.9, 60.0, st)[0] == 1


def test_object_on_the_line_holds_its_lane():
    _lw, st = _stable_world()
    asg = ObjectLaneAssigner()
    asg.update(7, 1.4, 20.0, st)
    labels = []
    for i in range(30):
        x = 1.85 + (0.1 if i % 2 else -0.1)
        labels.append(asg.update(7, x, 20.0, st)[0])
    assert set(labels) == {0}


def test_object_changing_lane_switches_after_confirmation():
    _lw, st = _stable_world()
    asg = ObjectLaneAssigner()
    asg.update(5, 0.0, 30.0, st)
    labels = [asg.update(5, 3.6, 30.0, st)[0] for _ in range(4)]
    assert labels[0] == 0 and labels[-1] == 1


def test_ego_lane_change_relabels_without_moving_the_car():
    """A car parked in the right lane becomes the ego-lane car when the ego moves over."""
    lw = LaneWorld()
    run(lw, 30)
    asg = ObjectLaneAssigner()
    seen = []
    for d in np.linspace(0.0, W, 90):
        st = lw.update(road(shift=float(d)), dt=DT, speed_mps=SPEED)
        seen.append(asg.update(9, W - float(d), 30.0, st)[0])
    st = run(lw, 10, shift=W)[-1]
    seen.append(asg.update(9, 0.0, 30.0, st)[0])
    assert seen[0] == 1 and seen[-1] == 0
    assert set(seen) <= {0, 1}
    # Exactly one relabel: no flicker through other lanes.
    assert sum(1 for a, b in zip(seen, seen[1:]) if a != b) == 1


# ------------------------------------------------------- lateral centring
CALIB = GroundCalibration(
    source_width=960, source_height=540, f_px=685.5,
    u_vp=480.0, v_vp=300.0, cam_height_m=1.35,
)


def _box_for(x_c, y, width=1.85, length=4.5, height=1.5):
    """Image box of a car whose rear-centre is at (x_c, y): rear face + inner side."""
    pts = []
    for dx in (-0.5 * width, 0.5 * width):
        for dy in (0.0, length):
            u, v = CALIB.project_ground(x_c + dx, y + dy)
            pts.append((u, v))
            pts.append((u, v - CALIB.f_px * height / (y + dy)))
    us = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    return [min(us), min(vs), max(us), max(vs)]


def test_side_on_car_is_centred_from_its_outer_edge():
    for x_true in (-3.6, 3.6):
        box = _box_for(x_true, 12.0)
        bottom = CALIB.backproject_ground(0.5 * (box[0] + box[2]), box[3])[0]
        x = measure_ground(box, CALIB, label="car", frame_shape=(540, 960)).x_m
        assert abs(x - x_true) < 0.2
        assert abs(bottom - x_true) > abs(x - x_true) + 0.2


def test_car_ahead_keeps_the_bottom_centre():
    box = _box_for(0.0, 20.0)
    bottom = CALIB.backproject_ground(0.5 * (box[0] + box[2]), box[3])[0]
    assert abs(vehicle_center_x(box, CALIB, "car", bottom, (540, 960)) - bottom) < 1e-9


def test_clipped_outer_edge_keeps_the_bottom_centre():
    box = _box_for(-3.6, 6.0)
    box[0] = 0.0
    bottom = CALIB.backproject_ground(0.5 * (box[0] + box[2]), box[3])[0]
    assert vehicle_center_x(box, CALIB, "car", bottom, (540, 960)) == bottom


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:
                fails += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if fails else 0)
