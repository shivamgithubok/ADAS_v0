"""Unit tests for cluster threat levels and alert arbitration.

Run:  python -m pytest -q tests/   (or: python tests/test_threat.py)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.adas.threat import (  # noqa: E402
    CAUTION, NEUTRAL, SAFE, WARNING,
    ThreatAssessor, format_lead_tag, longitudinal_level,
)

DT = 1.0 / 15.0


def obj(tid, z, x=0.0, vy=0.0, vx=0.0, lane=0, in_path=None, cipo=False):
    return {
        "track_id": tid, "Z_3d": z, "X_3d": x, "vy": vy, "vx": vx,
        "lane_index": lane,
        "in_path": (lane == 0) if in_path is None else in_path,
        "is_cipo": cipo,
    }


def run(assessor, frames, objs, cipo=None, status="SAFE", alerts=None, ego=25.0):
    out = None
    for _ in range(frames):
        out = assessor.update(objs, cipo, status, alerts or {}, ego, DT)
    return out


def test_longitudinal_bands():
    # 25 m/s ego: THW = range / 25
    assert longitudinal_level(60, 0.0, 25.0) == SAFE        # 2.4 s
    assert longitudinal_level(40, 0.0, 25.0) == CAUTION     # 1.6 s
    assert longitudinal_level(20, 0.0, 25.0) == WARNING     # 0.8 s
    assert longitudinal_level(60, 18.0, 25.0) == CAUTION    # TTC 3.3 s
    assert longitudinal_level(60, 25.0, 25.0) == WARNING    # TTC 2.4 s
    assert longitudinal_level(60, 0.0, 25.0, fcw_active=True) == WARNING


def test_closing_fast_is_caution():
    # 70 m at 25 m/s is THW 2.8 s, TTC 11 s, but closing 6.4 m/s (> 20 km/h)
    assert longitudinal_level(70, 6.4, 25.0) == CAUTION


def test_unknown_speed_uses_range_band():
    assert longitudinal_level(25, 0.0, None, band="WARNING") == CAUTION
    assert longitudinal_level(50, 0.0, None, band="SAFE") == SAFE


def test_red_holds_then_downgrades():
    a = ThreatAssessor()
    lead = obj(5, 15.0, vy=-10.0, cipo=True)
    out = run(a, 1, [lead], cipo=lead)
    assert out.levels[5] == WARNING
    safe = obj(5, 70.0, vy=0.0, cipo=True)
    # Red must hold for 1 s even after the threat clears.
    out = run(a, 10, [safe], cipo=safe)        # 0.67 s
    assert out.levels[5] == WARNING
    out = run(a, 20, [safe], cipo=safe)        # past red hold + downgrade hold
    assert out.levels[5] == SAFE


def test_escalation_is_immediate():
    a = ThreatAssessor()
    far = obj(3, 70.0, cipo=True)
    run(a, 5, [far], cipo=far)
    near = obj(3, 30.0, vy=-9.0, cipo=True)   # TTC 3.3 s → caution
    out = run(a, 1, [near], cipo=near)
    assert out.levels[3] == CAUTION


def test_adjacent_close_and_lane_change_conflict():
    a = ThreatAssessor()
    side = obj(8, 9.0, x=3.6, lane=1, in_path=False)
    out = run(a, 1, [side])
    assert out.levels[8] == CAUTION
    alerts = {"ldw": "RIGHT", "ldw_side": "RIGHT"}
    out = run(a, 1, [side], alerts=alerts)
    assert out.levels[8] == WARNING
    assert out.banner == "lane_change" and out.banner_side == "RIGHT"


def test_far_objects_are_neutral():
    a = ThreatAssessor()
    out = run(a, 1, [obj(1, 95.0, cipo=True)], cipo=obj(1, 95.0, cipo=True))
    assert out.levels[1] == NEUTRAL


def test_banner_priority_and_audio():
    a = ThreatAssessor()
    lead = obj(2, 10.0, vy=-8.0, cipo=True)
    alerts = {"fcw": "FCW+", "ldw": "LEFT", "ldw_side": "LEFT"}
    out = a.update([lead], lead, "DANGER", alerts, 20.0, DT)
    assert out.banner == "fcw_plus"
    assert out.vignette and out.red_active
    assert out.audio_loop == "fcw_plus"


def test_ldw_lane_states():
    a = ThreatAssessor()
    out = a.update([], None, "SAFE", {"ldw": "LEFT", "ldw_side": "LEFT"}, 30.0, DT)
    assert out.lane_left == "warning" and out.lane_right == "normal"
    assert out.banner == "ldw" and out.audio == "ldw_left"
    out = a.update([], None, "SAFE", {"ldw": "WATCH", "left_gap": 0.6, "right_gap": 0.3}, 30.0, DT)
    assert out.lane_right == "caution" and out.lane_left == "normal"
    assert out.banner is None


def test_ldw_side_flip_retriggers_tone():
    a = ThreatAssessor()
    out = a.update([], None, "SAFE", {"ldw": "LEFT", "ldw_side": "LEFT"}, 30.0, DT)
    assert out.audio == "ldw_left"
    out = a.update([], None, "SAFE", {"ldw": "LEFT", "ldw_side": "LEFT"}, 30.0, DT)
    assert out.audio is None
    out = a.update([], None, "SAFE", {"ldw": "RIGHT", "ldw_side": "RIGHT"}, 30.0, DT)
    assert out.audio == "ldw_right"


def test_speed_limit_levels_mph():
    a = ThreatAssessor(units="mph")
    isa = {"isa": {"posted_mph": 55}}
    mps = 0.44704
    assert a.update([], None, "SAFE", isa, 50 * mps, DT).over_limit == SAFE
    assert a.update([], None, "SAFE", isa, 58 * mps, DT).over_limit == CAUTION
    out = a.update([], None, "SAFE", isa, 66 * mps, DT)
    assert out.over_limit == WARNING and out.audio == "overspeed"
    # Tone fires once per limit, not every frame.
    assert a.update([], None, "SAFE", isa, 67 * mps, DT).audio is None


def test_lead_tag_text():
    a = ThreatAssessor()
    lead = obj(4, 30.0, vy=-9.0, cipo=True)
    out = a.update([lead], lead, "WARNING", {}, 22.8, DT)
    tag = format_lead_tag(out.lead, units="kmh")
    assert tag["text"].startswith("30 m")
    assert "1.3 s" in tag["text"]
    assert tag["sub"].endswith("km/h")


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if fails else 0)
