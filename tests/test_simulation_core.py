from pathlib import Path
from types import SimpleNamespace

from prosthetic_sim.mapper import GestureMapper
from prosthetic_sim.model_builder import build_urdf
from prosthetic_sim.camera_control import _gesture_from_landmarks


def test_mapper_close_open():
    m=GestureMapper()
    c=m.apply('CLOSE_HAND')
    assert c.index == 1 and c.little == 1
    c=m.apply('OPEN_HAND')
    assert c.index == 0 and c.thumb == 0


def test_wrist_mapping():
    m=GestureMapper()
    assert m.apply('WRIST_FLEXION').wrist_pitch_deg > 0
    assert m.apply('WRIST_EXTENSION').wrist_pitch_deg < 0
    assert m.apply('WRIST_SUPINATION').forearm_roll_deg > 0


def test_urdf_generated(tmp_path):
    p=build_urdf(tmp_path/'arm.urdf')
    txt=Path(p).read_text()
    assert 'wrist_deviation' in txt and 'index_mcp' in txt


def test_rest_returns_to_neutral_after_compound_pose():
    m = GestureMapper()
    m.apply("CLOSE_HAND")
    m.apply("WRIST_FLEXION")
    m.apply("WRIST_SUPINATION")
    neutral = m.apply("REST")
    assert neutral.gesture == "REST"
    assert neutral.index == neutral.middle == neutral.ring == neutral.little == 0.0
    assert neutral.thumb == 0.0
    assert neutral.wrist_pitch_deg == 0.0
    assert neutral.forearm_roll_deg == 0.0
    assert neutral.wrist_deviation_deg == 0.0
    assert neutral.elbow_flex_deg == 25.0


def _hand(points):
    return SimpleNamespace(landmark=[SimpleNamespace(x=x, y=y) for x, y in points])


def test_camera_recognizes_open_and_closed_hand():
    # Upright open palm: fingertips are farther from wrist than PIP joints.
    points = [(0.50, 0.80)] * 21
    for tip, pip, x in ((8, 6, .40), (12, 10, .46), (16, 14, .54), (20, 18, .60)):
        points[pip] = (x, .50)
        points[tip] = (x, .25)
    points[5] = (.40, .55)
    points[17] = (.60, .55)
    points[4] = (.15, .55)
    assert _gesture_from_landmarks(_hand(points)) == "OPEN_HAND"

    # Closed fist: fingertips remain close to the wrist and thumb is tucked.
    closed = [(0.50, 0.80)] * 21
    closed[5] = (.46, .76)
    closed[17] = (.54, .76)
    closed[4] = (.48, .77)
    assert _gesture_from_landmarks(_hand(closed)) == "CLOSE_HAND"


def test_camera_pose_mapping_does_not_inherit_previous_pose():
    mapper = GestureMapper()
    mapper.apply("CLOSE_HAND")
    mapper.apply("WRIST_FLEXION")
    mapper.reset()
    wrist_only = mapper.apply("WRIST_SUPINATION")
    assert wrist_only.forearm_roll_deg > 0
    assert wrist_only.index == wrist_only.middle == wrist_only.ring == wrist_only.little == 0.0
    assert wrist_only.wrist_pitch_deg == 0.0
