from prosthetic_sim.mapper import GestureMapper
from prosthetic_semg.config import CLASS_NAMES

def test_all_semg_classes_map_to_simulation():
    m=GestureMapper()
    for g in CLASS_NAMES:
        c=m.apply(g,0.9)
        assert c.gesture==g
