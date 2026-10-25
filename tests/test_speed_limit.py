"""Posted speed limit must not walk through every YOLO class on one sign."""

from src.inference.speed_limit_tracker import SpeedLimitTracker
from src.utils.speed_source import HudSpeedSource, make_speed_source

# GRMN6695 frames 256–344, sign detector every 4th frame. One SPEED LIMIT 50.
# Tiny distant boxes say 65/45; the readable plate says 50; the last frames
# disagree (30 and 35 on the same box) as the sign leaves the frame.
_FRAMES = [
    (256, [("speedLimit65", 0.51, (525, 268, 542, 293))]),
    (260, [("speedLimit65", 0.66, (528, 267, 545, 291))]),
    (264, [("speedLimit65", 0.61, (533, 264, 550, 290))]),
    (268, [("speedLimit65", 0.74, (537, 262, 554, 290))]),
    (276, [("speedLimit65", 0.78, (547, 263, 565, 290))]),
    (292, [("speedLimit45", 0.69, (573, 253, 593, 283))]),
    (300, [("speedLimit45", 0.44, (590, 243, 614, 279))]),
    (304, [("speedLimit50", 0.35, (600, 238, 626, 278))]),
    (312, [("speedLimit50", 0.43, (628, 228, 656, 271))]),
    (316, [("speedLimit50", 0.71, (643, 220, 675, 267))]),
    (320, [("speedLimit50", 0.69, (663, 213, 696, 264))]),
    (324, [
        ("speedLimit35", 0.44, (687, 202, 722, 259)),
        ("speedLimit50", 0.38, (687, 202, 722, 259)),
    ]),
    (328, [("speedLimit50", 0.80, (717, 189, 756, 252))]),
    (332, [
        ("speedLimit30", 0.54, (753, 173, 793, 245)),
        ("speedLimit35", 0.36, (753, 173, 793, 245)),
    ]),
    (336, [("speedLimit30", 0.84, (797, 155, 840, 234))]),
    (340, [
        ("speedLimit30", 0.79, (854, 131, 897, 222)),
        ("speedLimit35", 0.64, (853, 131, 897, 222)),
    ]),
    (344, [
        ("speedLimit35", 0.48, (919, 103, 960, 205)),
        ("speedLimit30", 0.42, (921, 103, 960, 204)),
    ]),
]


def _dets(items):
    out = []
    for name, conf, box in items:
        out.append({"class": name, "conf": conf, "bbox": list(box), "mph": None})
    return out


def test_one_sign_stays_on_50():
    isa = SpeedLimitTracker(confirm_hits=3, act_conf=0.55, replace_hits=5)
    posted = []
    for _frame, items in _FRAMES:
        snap = isa.update(_dets(items), ran_infer=True, frame_shape=(540, 960))
        posted.append(snap["posted_mph"])
    assert 65 not in posted
    assert 30 not in posted
    assert 45 not in posted
    assert posted[-1] == 50
    assert isa.status == "CONFIRMED"


def test_auto_speed_does_not_borrow_another_clip():
    src = make_speed_source("auto", "GRMN6695_540p30.mp4")
    assert isinstance(src, HudSpeedSource)
    assert src.get_mps(800) is None
