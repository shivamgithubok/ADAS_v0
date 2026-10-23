"""Lane-change motion, four-lane bird's-eye marks, and detection smoothing."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tracking.lane_frame import LaneFrameModel
from src.ui.bev_quick3d import (
    four_lane_offsets,
    lane_change_target,
    lane_ladder,
    lane_slide_target,
    occupied_lane_center,
    continue_slide,
    smooth_measurement,
)


def _pair(center, half=1.85):
    ys = np.linspace(5.0, 50.0, 12)
    left = np.column_stack([np.full_like(ys, center - half), ys])
    right = np.column_stack([np.full_like(ys, center + half), ys])
    return left, right


def test_lane_ladder_hands_the_outer_line_off():
    w = 3.7
    before = [x for x, _pal in lane_ladder(w, 0.0)]
    after = [x for x, _pal in lane_ladder(w, w)]
    assert len(before) == 4 and len(after) == 4
    # The old second line now sits where the old outer line was.
    assert abs(after[0] - (before[1] - w)) < 0.05
    # A new line has come in on the right, in the old outer slot.
    assert abs(after[-1] - before[-1]) < 0.05
    assert abs(after[-2] - (before[-1] - w)) < 0.05


def test_corridor_follows_the_lane_the_car_is_in():
    w = 3.7
    assert abs(occupied_lane_center(w, 0.0)) < 0.05
    # Still in the original lane: the corridor has slid with that lane.
    drifting = occupied_lane_center(w, 1.2)
    assert drifting < -0.8
    # After the car has crossed, the corridor is in the lane now under it.
    entered = occupied_lane_center(w, w)
    assert abs(entered) < 0.3


def test_painted_corridor_is_in_the_lane_under_the_car():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from types import SimpleNamespace

    from PySide6.QtCore import QCoreApplication, QObject

    from src.ui.bev_quick3d import BevSceneDriver

    app = QCoreApplication.instance() or QCoreApplication([])
    drv = BevSceneDriver(QObject())
    drv._extrap.stop()
    left, right = _pair(0.0)
    drv.update_bev_data([], [], left_3d=left, right_3d=right, speed_mps=20.0, dt=0.05)
    drv._threat = SimpleNamespace(lead=SimpleNamespace(level=2, range_m=40.0))
    drv._lane_slide = 3.7
    drv._lane_slide_target = 3.7
    for _ in range(40):
        drv._ease_slide()
    rows = drv._corridor_payload()
    assert rows
    mean_x = sum(r["x"] for r in rows) / len(rows)
    # A full lane of line travel used to park the band one lane to the left.
    assert abs(mean_x) < 0.45
    drv._extrap.stop()
    del app


def test_four_lane_marks_use_detections():
    marks = four_lane_offsets(3.7, detected=[-5.4, -1.7, 1.9, 5.6])
    assert len(marks) == 4
    xs = [m[0] for m in marks]
    assert xs[0] < xs[1] < xs[2] < xs[3]
    assert abs(xs[1] - (-1.7)) < 0.2
    assert abs(xs[2] - 1.9) < 0.2
    assert [m[1] for m in marks] == [0, 1, 2, 0]


def test_four_lane_marks_without_detections():
    marks = four_lane_offsets(3.7, detected=[])
    assert len(marks) == 4
    assert marks[1][0] < 0 < marks[2][0]


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
    # Next lane is about one width to the right of the camera origin.
    left, right = _pair(3.7)
    assert model.update(left, right, speed_mps=20.0, dt=0.05)
    assert abs(model.relock_jump_m) > 1.2


def test_lane_slide_only_when_lanes_are_real():
    assert lane_slide_target(1.6, False, "normal", "normal") == 0.0
    assert abs(lane_slide_target(0.1, True, "normal", "normal") - 0.1) < 1e-6
    assert abs(lane_slide_target(1.6, True, "normal", "normal") - 1.6) < 1e-6
    warned = lane_slide_target(0.1, True, "normal", "warning")
    assert warned >= 0.75
    # A warning on the line just crossed must not reverse the lane slide.
    assert abs(lane_slide_target(3.2, True, "warning", "normal") - 3.2) < 1e-6


def test_pair_swap_continues_the_slide_instead_of_reversing():
    w = 3.7
    # Already 1.7 m into a rightward change. The tracker then locks the next
    # pair, whose wrapped centre is on the other side of zero.
    assert abs(continue_slide(-1.2, 1.7, w, jump=2.9) - 2.5) < 0.05
    # From rest, the swap itself is the maneuver and the paint has to travel.
    assert abs(continue_slide(-3.7, 0.0, w, jump=3.7) - (-3.7)) < 1e-6


def test_video_style_swap_keeps_the_car_in_its_lane():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import json
    from PySide6.QtCore import QCoreApplication, QObject

    from src.ui.bev_quick3d import BevSceneDriver

    app = QCoreApplication.instance() or QCoreApplication([])
    drv = BevSceneDriver(QObject())
    drv._extrap.stop()
    lead = {
        "track_id": 4, "X_3d": 0.0, "Z_3d": 30.0, "lane_offset_m": 0.0,
        "lane_index": 0, "label": "car", "bbox": [0, 0, 80, 40],
        "vx": 0.0, "vy": 0.0,
    }
    for center in (0.0, -0.6, -1.2, -1.7):
        left, right = _pair(center)
        drv.update_bev_data([], [lead], left_3d=left, right_3d=right, speed_mps=20.0, dt=0.05)
        for _ in range(18):
            drv._ease_slide()
        drv._refresh_mini_bev()
    before = json.loads(drv.mini_bev_json())
    slide_at_swap = float(drv._lane_slide)
    assert slide_at_swap > 1.0
    assert min(before["lanes"]) < -4.0
    lead["lane_index"] = -1
    left, right = _pair(1.2)
    xs = []
    for _ in range(8):
        drv.update_bev_data([], [lead], left_3d=left, right_3d=right, speed_mps=20.0, dt=0.05)
        for _e in range(4):
            drv._ease_slide()
        drv._publish_traffic()
        xs.append(json.loads(drv.mini_bev_json())["cars"][0]["x"])
    after = json.loads(drv.mini_bev_json())
    # Lines keep travelling with the change. The lead stays in the lane it
    # occupied; the index renumber does not throw it across the road.
    assert drv._lane_slide > slide_at_swap + 0.3
    assert max(xs) - min(xs) < 2.2
    lanes = after["lanes"]
    assert any(a < xs[-1] < b for a, b in zip(lanes, lanes[1:]))
    drv._extrap.stop()
    del app


def test_relock_slides_one_lane_and_keeps_the_object():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import json
    from PySide6.QtCore import QCoreApplication, QObject

    from src.ui.bev_quick3d import BevSceneDriver

    app = QCoreApplication.instance() or QCoreApplication([])
    drv = BevSceneDriver(QObject())
    drv._extrap.stop()
    obj = {
        "track_id": 4,
        "X_3d": 0.1,
        "Z_3d": 28.0,
        "lane_offset_m": 0.1,
        "lane_index": 0,
        "label": "car",
        "bbox": [0, 0, 80, 40],
        "vx": 0.0,
        "vy": 0.0,
    }
    left, right = _pair(0.0)
    for _ in range(8):
        drv.update_bev_data([], [obj], left_3d=left, right_3d=right, speed_mps=20.0, dt=0.05)
    before = json.loads(drv.mini_bev_json())
    left, right = _pair(3.7)
    drv.update_bev_data([], [obj], left_3d=left, right_3d=right, speed_mps=20.0, dt=0.05)
    jumped = json.loads(drv.mini_bev_json())
    # The re-lock must not teleport the paint. The shift is the ease.
    assert abs(jumped["lanes"][1] - before["lanes"][1]) < 1.2

    def follow(prev, lanes):
        return min(lanes, key=lambda x: abs(x - prev))

    tracked = before["lanes"][1]
    samples = [tracked]
    for _ in range(24):
        drv._ease_slide()
        drv._publish_traffic()
        lanes = json.loads(drv.mini_bev_json())["lanes"]
        tracked = follow(tracked, lanes)
        samples.append(tracked)
    assert abs(samples[-1] - samples[0]) > 2.0
    steps = [abs(b - a) for a, b in zip(samples, samples[1:])]
    assert max(steps) < 1.0
    car = json.loads(drv.mini_bev_json())["cars"][0]
    lanes = json.loads(drv.mini_bev_json())["lanes"]
    assert any(a < car["x"] < b for a, b in zip(lanes, lanes[1:]))
    drv._extrap.stop()
    del app


def test_new_car_appears_at_the_detection():
    measured = (1.2, 40.0)
    pos = smooth_measurement(None, measured)
    assert abs(pos[0] - measured[0]) < 1e-6
    assert abs(pos[1] - measured[1]) < 1e-6
    assert abs(pos[1]) > 20.0


def test_mini_bev_carries_the_departure_side():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QObject

    from src.ui.bev_quick3d import BevSceneDriver

    app = QCoreApplication.instance() or QCoreApplication([])
    drv = BevSceneDriver(QObject())
    drv._extrap.stop()
    drv._lane_states = ("warning", "normal")
    drv.lane_frame.valid = True
    drv._refresh_mini_bev()
    import json
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
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QObject

    from src.ui.bev_quick3d import BevSceneDriver

    app = QCoreApplication.instance() or QCoreApplication([])
    drv = BevSceneDriver(QObject())
    drv._extrap.stop()
    obj = {
        "track_id": 7,
        "X_3d": 0.2,
        "Z_3d": 32.0,
        "lane_offset_m": 0.1,
        "lane_index": 0,
        "label": "car",
        "bbox": [0, 0, 90, 50],
        "vx": 0.0,
        "vy": 0.0,
    }
    left, right = _pair(0.0)
    drv.update_bev_data([], [obj], left_3d=left, right_3d=right, speed_mps=20.0, dt=0.05)
    assert drv._traffic_shown, "a detected car should be on the bird's-eye view"
    spawned = -float(drv._traffic_shown[0]["posZ"])
    assert abs(spawned - 32.0) < 1.5

    left, right = _pair(1.7)
    drv.update_bev_data([], [obj], left_3d=left, right_3d=right, speed_mps=20.0, dt=0.05)
    for _ in range(36):
        drv._ease_slide()
    drv._publish_traffic()
    import json
    mini = json.loads(drv.mini_bev_json())
    assert 4 <= len(mini["lanes"]) <= 5
    assert mini["ego"] == 0.0
    assert abs(drv._lane_slide) > 0.6
    car = mini["cars"][0]
    lanes = mini["lanes"]
    assert any(a < car["x"] < b for a, b in zip(lanes, lanes[1:]))
    drv._extrap.stop()
    del app


def test_corridor_stripes_scroll_toward_the_car_at_ego_speed():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from types import SimpleNamespace

    from PySide6.QtCore import QCoreApplication, QObject

    from src.ui.bev_quick3d import BevSceneDriver

    app = QCoreApplication.instance() or QCoreApplication([])
    drv = BevSceneDriver(QObject())
    drv._extrap.stop()
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
