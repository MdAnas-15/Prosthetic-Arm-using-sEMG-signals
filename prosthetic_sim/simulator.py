from __future__ import annotations
import math
import time
from .model_builder import build_urdf
from .command import ProstheticCommand

class PyBulletProstheticSimulator:
    """Small deterministic PyBullet wrapper with bounded position control."""
    def __init__(self, gui=True, urdf_path="generated/prosthetic_arm.urdf"):
        try:
            import pybullet as p
        except ImportError as exc:
            raise RuntimeError("PyBullet is not installed for this Python. Run: python -m pip install -r requirements.txt") from exc
        self.p = p
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet.")
        self.closed = False
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1/240, physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=1/240, numSolverIterations=100,
                                    enableConeFriction=1, physicsClientId=self.client)
        plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane,
                          basePosition=[0, 0, 0], physicsClientId=self.client)
        path = build_urdf(urdf_path)
        flags = p.URDF_USE_INERTIA_FROM_FILE
        self.body = p.loadURDF(str(path), [0, 0, 0.02], useFixedBase=True,
                               flags=flags, physicsClientId=self.client)
        self.joints = {}
        self.limits = {}
        for idx in range(p.getNumJoints(self.body, physicsClientId=self.client)):
            info = p.getJointInfo(self.body, idx, physicsClientId=self.client)
            name = info[1].decode("utf-8")
            self.joints[name] = idx
            self.limits[name] = (float(info[8]), float(info[9]))
            p.setJointMotorControl2(self.body, idx, p.VELOCITY_CONTROL, force=0,
                                    physicsClientId=self.client)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(cameraDistance=0.95, cameraYaw=45,
                                         cameraPitch=-18, cameraTargetPosition=[0, 0, 0.48],
                                         physicsClientId=self.client)

    def _set(self, name, target, force=5.0):
        if name not in self.joints:
            return
        lower, upper = self.limits[name]
        target = max(lower, min(upper, float(target)))
        self.p.setJointMotorControl2(self.body, self.joints[name],
            self.p.POSITION_CONTROL, targetPosition=target, force=force,
            positionGain=0.18, velocityGain=1.0, maxVelocity=3.0,
            physicsClientId=self.client)

    def apply(self, cmd: ProstheticCommand):
        cmd = cmd.clamp()
        rad = math.radians
        self._set("shoulder_yaw", rad(cmd.shoulder_yaw_deg), 8)
        self._set("elbow_flex", rad(cmd.elbow_flex_deg), 8)
        self._set("forearm_roll", rad(cmd.forearm_roll_deg), 5)
        self._set("wrist_pitch", rad(cmd.wrist_pitch_deg), 5)
        self._set("wrist_deviation", rad(cmd.wrist_deviation_deg), 5)
        for finger, value in (("index",cmd.index),("middle",cmd.middle),
                              ("ring",cmd.ring),("little",cmd.little)):
            self._set(f"{finger}_mcp", 1.20*value, 2.5)
            self._set(f"{finger}_pip", 1.40*value, 2.0)
            self._set(f"{finger}_dip", 1.05*value, 1.5)
        self._set("thumb_opposition", 0.80*cmd.thumb, 2.5)
        self._set("thumb_mcp", 1.05*cmd.thumb, 2.0)
        self._set("thumb_ip", 0.95*cmd.thumb, 1.5)

    def step(self, steps=1, realtime=False):
        for _ in range(max(0, int(steps))):
            self.p.stepSimulation(physicsClientId=self.client)
            if realtime:
                time.sleep(1/240)

    def reset(self):
        for name, idx in self.joints.items():
            self.p.resetJointState(self.body, idx, 0.0, targetVelocity=0,
                                   physicsClientId=self.client)
            self._set(name, 0.0)
        self.step(24)

    def close(self):
        if not self.closed and self.client >= 0:
            self.p.disconnect(self.client)
            self.client = -1
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
