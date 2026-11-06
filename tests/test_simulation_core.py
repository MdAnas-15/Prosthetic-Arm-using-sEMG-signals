from pathlib import Path
from prosthetic_sim.mapper import GestureMapper
from prosthetic_sim.model_builder import build_urdf

def test_mapper_close_open():
    m=GestureMapper(); c=m.apply('CLOSE_HAND'); assert c.index==1 and c.little==1
    c=m.apply('OPEN_HAND'); assert c.index==0 and c.thumb==0

def test_wrist_mapping():
    m=GestureMapper(); assert m.apply('WRIST_FLEXION').wrist_pitch_deg>0
    assert m.apply('WRIST_EXTENSION').wrist_pitch_deg<0
    assert m.apply('WRIST_SUPINATION').forearm_roll_deg>0

def test_urdf_generated(tmp_path):
    p=build_urdf(tmp_path/'arm.urdf'); txt=Path(p).read_text(); assert 'wrist_deviation' in txt and 'index_mcp' in txt


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
