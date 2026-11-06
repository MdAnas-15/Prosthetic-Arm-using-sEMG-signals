from __future__ import annotations
import time
from .mapper import GestureMapper, GESTURES
from .simulator import PyBulletProstheticSimulator
from .logger import SimulationLogger

DEMO_SEQUENCE=["OPEN_HAND","CLOSE_HAND","OPEN_HAND","WRIST_SUPINATION","WRIST_PRONATION","WRIST_FLEXION","WRIST_EXTENSION","RADIAL_DEVIATION","ULNAR_DEVIATION","REST"]

def run_demo(gui=True, hold_seconds=0.8):
    mapper=GestureMapper()
    with PyBulletProstheticSimulator(gui=gui) as sim, SimulationLogger() as log:
        for gesture in DEMO_SEQUENCE:
            cmd=mapper.apply(gesture,1.0); sim.apply(cmd); log.write(cmd)
            steps=max(1,int(240*hold_seconds))
            sim.step(steps,realtime=gui)
            print(f"{gesture:20s} -> simulation command applied")
        if gui:
            print('Demo complete. Window will close in 2 seconds.')
            time.sleep(2)
