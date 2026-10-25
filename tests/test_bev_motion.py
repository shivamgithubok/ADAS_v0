"""Bird's-eye placement in the ego frame, lane-change motion, and detection smoothing."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.postprocess import ANCHOR_Y_STEPS
from src.tracking.lane_frame import LaneFrameModel
from src.tracking.lane_world import LaneWorld, ObjectLaneAssigner
from src.ui.bev_quick3d import ease_boundaries, lane_change_target, smooth_measurement

W = 3.7
DT = 0.05


def _pair(center, half=1.85):
    ys = np.linspace(5.0, 50.0, 12)
    z = np.zeros_like(ys)
    left = np.column_stack([np.full_like(ys, center - half), ys, z])
    right = np.column_stack([np.full_like(ys, center + half), ys, z])
    return left, right


def _road(shift=0.0):
    ys = ANCHOR_Y_STEPS.astype(np.float64)
    out = []
    for k in range(-3, 3):
        x = (k + 0.5) * W - shift
        if abs(x) < 8.0:
            out.append(np.column_stack([np.full_like(ys, x), ys, np.zeros_like(ys)]).astype(np.float32))
    return out


def _driver():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QObject

    from src.ui.bev_quick3d import BevSceneDriver

    app = QCoreApplication.instance() or QCoreApplication([])
    drv = BevSceneDriver(QObject())
    drv._extrap.stop()
    return app, drv


def _settle(drv, n=40):
    for _ in range(n):
        drv._ease_lanes()
    drv._publish_slid_geometry()
    drv._publish_traffic()
    return json.loads(drv.mini_bev_json())


def _band(lanes, x):
    return next((i for i, (a, b) in enumerate(zip(lanes, lanes[1:])) if a < x < b), None)


def test_ease_boundaries_starts_new_lines_in_place_and_drops_old_ones():
    disp = {0: -1.85, 1: 1.85}
    out = ease_boundaries(disp, {0: -1.65, 1: 2.05, 2: 5.6}, gain=0.5)
    assert abs(out[0] + 1.75) < 1e-9 and abs(out[1] - 1.95) < 1e-9
    assert out[2] == 5.6
    out = ease_boundaries(out, {1: 2.05, 2: 5.6})
    assert 0 not in out


def test_lane_lines_are_drawn_at_the_physical_boundaries():
    app, drv = _driver()
    lw = LaneWorld()
    xs = [-6.0, -2.1, 2.0, 5.1]
    ys = ANCHOR_Y_STEPS.astype(np.float64)
    lines = [
        np.column_stack([np.full_like(ys, x), ys, np.zeros_like(ys)]).astype(np.float32)
        for x in xs
    ]
    for _ in range(40):
        st = lw.update(lines, dt=DT, speed_mps=20.0)
        drv.update_bev_data([], [], speed_mps=20.0, dt=DT, alerts={"lane_world": st})
    mini = _settle(drv)
    assert np.allclose(mini["lanes"], xs, atol=0.15)
    assert abs(mini["w"] - 4.1) < 0.2
    drv._extrap.stop()
    del app


def test_adjacent_car_is_drawn_in_the_adjacent_lane():
    """The GRMN6693 case: a car in the left lane must not be drawn two lanes over."""
    app, drv = _driver()
    lw = LaneWorld()
    asg = ObjectLaneAssigner()
    obj = {"track_id": 3, "X_3d": -3.8, "Z_3d": 14.0, "label": "car",
           "bbox": [0, 0, 80, 40], "vx": 0.0, "vy": 0.0}
    for _ in range(30):
        st = lw.update(_road(), dt=DT, speed_mps=20.0)
        obj["lane_index"] = asg.update(3, obj["X_3d"], obj["Z_3d"], st)[0]
        drv.update_bev_data([], [obj], speed_mps=20.0, dt=DT, alerts={"lane_world": st})
    mini = _settle(drv)
    assert obj["lane_index"] == -1
    assert _band(mini["lanes"], mini["cars"][0]["x"]) == 0
    drv._extrap.stop()
    del app


def test_ego_lane_change_moves_lines_not_cars():
    """Ego changes lane right; a car parked in that lane ends up dead ahead, smoothly."""
    app, drv = _driver()
    lw = LaneWorld()
    asg = ObjectLaneAssigner()
    obj = {"track_id": 4, "X_3d": W, "Z_3d": 30.0, "label": "car",
           "bbox": [0, 0, 80, 40], "vx": 0.0, "vy": 0.0}
    for _ in range(20):
        st = lw.update(_road(), dt=DT, speed_mps=20.0)
        obj["lane_index"] = asg.update(4, obj["X_3d"], 30.0, st)[0]
        drv.update_bev_data([], [obj], speed_mps=20.0, dt=DT, alerts={"lane_world": st})
    xs, bands = [], []
    for d in np.linspace(0.0, W, 60):
        st = lw.update(_road(float(d)), dt=DT, speed_mps=20.0)
        obj["X_3d"] = W - float(d)
        obj["lane_index"] = asg.update(4, obj["X_3d"], 30.0, st)[0]
        drv.update_bev_data([], [obj], speed_mps=20.0, dt=DT, alerts={"lane_world": st})
        for _ in range(2):
            drv._ease_lanes()
        drv._publish_traffic()
        mini = json.loads(drv.mini_bev_json())
        xs.append(mini["cars"][0]["x"])
        bands.append(_band(mini["lanes"], xs[-1]))
    assert st.lane_changes == 1 and st.ego_lane_id == 1
    steps = np.abs(np.diff(xs))
    assert steps.max() < 0.6, "the car must not jump a lane when the ego lane is renumbered"
    mini = _settle(drv)
    assert _band(mini["lanes"], mini["cars"][0]["x"]) == 1
    assert abs(mini["cars"][0]["x"]) < 0.6
    drv._extrap.stop()
    del app


def test_detector_relock_does_not_jump_lines_or_cars():
    """Frames that only see the neighbour pair: the drawn road and the car stay put."""
    app, drv = _driver()
    lw = LaneWorld()
    asg = ObjectLaneAssigner()
    obj = {"track_id": 5, "X_3d": 0.2, "Z_3d": 28.0, "label": "car",
           "bbox": [0, 0, 80, 40], "vx": 0.0, "vy": 0.0}
    for _ in range(30):
        st = lw.update(_road(), dt=DT, speed_mps=20.0)
        obj["lane_index"] = asg.update(5, 0.2, 28.0, st)[0]
        drv.update_bev_data([], [obj], speed_mps=20.0, dt=DT, alerts={"lane_world": st})
    before = _settle(drv)
    for _ in range(20):
        lines = [l for l in _road() if l[0, 0] > 0.0]
        st = lw.update(lines, dt=DT, speed_mps=20.0)
        obj["lane_index"] = asg.update(5, 0.2, 28.0, st)[0]
        drv.update_bev_data([], [obj], speed_mps=20.0, dt=DT, alerts={"lane_world": st})
        mini = _settle(drv, 2)
        assert np.allclose(mini["lanes"], before["lanes"], atol=0.3)
        assert abs(mini["cars"][0]["x"] - before["cars"][0]["x"]) < 0.3
    assert st.lane_changes == 0
    drv._extrap.stop()
    del app


def test_corridor_fallback_without_a_pipeline_world():
    """Callers that only pass the ego corridor still get a lane world."""
    app, drv = _driver()
    left, right = _pair(0.0)
    for _ in range(10):
        drv.update_bev_data([], [], left_3d=left, right_3d=right, speed_mps=20.0, dt=DT)
    mini = _settle(drv)
    assert len(mini["lanes"]) == 4
    assert mini["lanes"][1] < 0.0 < mini["lanes"][2]
    drv._extrap.stop()
    del app


def test_painted_corridor_is_in_the_ego_lane():
    from types import SimpleNamespace

    app, drv = _driver()
    lw = LaneWorld()
    for _ in range(20):
        st = lw.update(_road(0.8), dt=DT, speed_mps=20.0)
        drv.update_bev_data([], [], speed_mps=20.0, dt=DT, alerts={"lane_world": st})
    _settle(drv)
    drv._threat = SimpleNamespace(lead=SimpleNamespace(level=2, range_m=40.0))
    rows = drv._corridor_payload()
    assert rows
    mean_x = sum(r["x"] for r in rows) / len(rows)
    # The car sits 0.8 m right of its lane centre; the band is on the lane, not the car.
    assert abs(mean_x + 0.8) < 0.2
    drv._extrap.stop()
    del app


def test_lane_change_slides_the_car():
    held, _yaw = lane_change_target(0.0, "normal", "normal")
    assert held == 0.0
    right, yaw = lane_change_target(1.6, "normal", "normal")
    assert right == 1.6
    assert yaw < 0
    warned, wy = lane_change_target(0.1, "normal", "warning")
    assert warned >= 0.75
    assert wy < 0


def test_lane_relock_is_reported():
    model = LaneFrameModel(pose_alpha=1.0, curv_alpha=1.0, width_alpha=1.0)
    left, right = _pair(0.0)
    assert model.update(left, right, speed_mps=20.0, dt=0.05)
    assert model.relock_jump_m == 0.0
    left, right = _pair(3.7)
    assert model.update(left, right, speed_mps=20.0, dt=0.05)
    assert abs(model.relock_jump_m) > 1.2


def test_new_car_appears_at_the_detection():
    measured = (1.2, 40.0)
    pos = smooth_measurement(None, measured)
    assert abs(pos[0] - measured[0]) < 1e-6
    assert abs(pos[1] - measured[1]) < 1e-6
    assert abs(pos[1]) > 20.0


def test_mini_bev_carries_the_departure_side():
    app, drv = _driver()
    drv._lane_states = ("warning", "normal")
    drv.lane_frame.valid = True
    drv._refresh_mini_bev()
    mini = json.loads(drv.mini_bev_json())
    assert mini["left"] == "warning"
    assert mini["right"] == "normal"
    drv._extrap.stop()
    del app


def test_top_view_draws_a_car_icon():
    qml = open(
        os.path.join(os.path.dirname(__file__), "..", "src", "ui", "qml", "Cluster.qml"),
        encoding="utf-8",
    ).read()
    paint = qml.split("id: miniBev", 1)[1].split("BIRD'S EYE", 1)[0]
    assert "drawCarIcon" in paint
    assert "fillRect(cx - 7" not in paint


def test_driver_spots_the_car_on_the_detection():
    app, drv = _driver()
    obj = {
        "track_id": 7, "X_3d": 0.2, "Z_3d": 32.0, "lane_offset_m": 0.1,
        "lane_index": 0, "label": "car", "bbox": [0, 0, 90, 50], "vx": 0.0, "vy": 0.0,
    }
    left, right = _pair(0.0)
    drv.update_bev_data([], [obj], left_3d=left, right_3d=right, speed_mps=20.0, dt=0.05)
    assert drv._traffic_shown, "a detected car should be on the bird's-eye view"
    spawned = -float(drv._traffic_shown[0]["posZ"])
    assert abs(spawned - 32.0) < 1.5
    mini = _settle(drv)
    assert len(mini["lanes"]) == 4
    assert mini["ego"] == 0.0
    car = mini["cars"][0]
    assert _band(mini["lanes"], car["x"]) == 1
    drv._extrap.stop()
    del app


def test_corridor_stripes_scroll_toward_the_car_at_ego_speed():
    from types import SimpleNamespace

    app, drv = _driver()
    left, right = _pair(0.0)
    drv.update_bev_data([], [], left_3d=left, right_3d=right, speed_mps=20.0, dt=0.05)
    drv._threat = SimpleNamespace(lead=SimpleNamespace(level=2, range_m=40.0))
    drv._paint_m = 0.0
    near = drv._corridor_payload()
    drv._paint_m = 1.6
    far = drv._corridor_payload()
    assert near and far
    # World z is -y, so a stripe moving toward the bumper has a larger z.
    assert sorted(r["z"] for r in far)[2] > sorted(r["z"] for r in near)[2]
    drv._paint_t = 0.0
    drv._advance_paint()
    assert drv._paint_m > 1.6
    drv._extrap.stop()
    del app


def test_smoothing_kills_frame_to_frame_jitter():
    pos = None
    noisy = [(0.0, 30.0), (1.4, 29.2), (-0.8, 31.1), (0.6, 29.4), (0.1, 30.2)]
    path = []
    for sample in noisy:
        pos = smooth_measurement(pos, sample, alpha=0.28, max_step=0.55)
        path.append(pos)
    jumps = [
        abs(path[i][0] - path[i - 1][0]) for i in range(1, len(path))
    ]
    assert max(jumps) <= 0.55 + 1e-6
    assert abs(path[-1][0]) < 1.4


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
