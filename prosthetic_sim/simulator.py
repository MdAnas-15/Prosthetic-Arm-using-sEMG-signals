from __future__ import annotations
from pathlib import Path
import math, time
from .model_builder import build_urdf
from .command import ProstheticCommand

class PyBulletProstheticSimulator:
    def __init__(self, gui=True, urdf_path="generated/prosthetic_arm.urdf"):
        try:
            import pybullet as p
        except ImportError as e:
            raise RuntimeError("PyBullet is not installed. Run: pip install -r requirements.txt") from e
        self.p=p
        self.client=p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0: raise RuntimeError("Could not connect to PyBullet")
        p.setGravity(0,0,-9.81,physicsClientId=self.client)
        p.setTimeStep(1/240,physicsClientId=self.client)
        plane_shape=p.createCollisionShape(p.GEOM_PLANE,physicsClientId=self.client)
        p.createMultiBody(baseMass=0,baseCollisionShapeIndex=plane_shape,basePosition=[0,0,0],physicsClientId=self.client)
        path=build_urdf(urdf_path)
        self.body=p.loadURDF(str(path),[0,0,0.02],useFixedBase=True,physicsClientId=self.client)
        self.joints={}
        for i in range(p.getNumJoints(self.body,physicsClientId=self.client)):
            info=p.getJointInfo(self.body,i,physicsClientId=self.client)
            self.joints[info[1].decode()]=i
        if gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.05,cameraYaw=45,cameraPitch=-25,cameraTargetPosition=[0,0,0.45],physicsClientId=self.client)

    def _set(self,name,target,force=2.0):
        if name in self.joints:
            self.p.setJointMotorControl2(self.body,self.joints[name],self.p.POSITION_CONTROL,targetPosition=float(target),force=force,physicsClientId=self.client)

    def apply(self, cmd: ProstheticCommand):
        r=math.radians
        self._set('shoulder_yaw',r(cmd.shoulder_yaw_deg),6)
        self._set('elbow_flex',r(cmd.elbow_flex_deg),6)
        self._set('forearm_roll',r(cmd.forearm_roll_deg),4)
        self._set('wrist_pitch',r(cmd.wrist_pitch_deg),4)
        self._set('wrist_deviation',r(cmd.wrist_deviation_deg),4)
        for finger,value in [('index',cmd.index),('middle',cmd.middle),('ring',cmd.ring),('little',cmd.little)]:
            self._set(f'{finger}_mcp',1.20*value,1.5)
            self._set(f'{finger}_pip',1.40*value,1.2)
            self._set(f'{finger}_dip',1.05*value,1.0)
        self._set('thumb_opposition',0.80*cmd.thumb,1.5)
        self._set('thumb_mcp',1.05*cmd.thumb,1.2)
        self._set('thumb_ip',0.95*cmd.thumb,1.0)

    def step(self, steps=1, realtime=False):
        for _ in range(int(steps)):
            self.p.stepSimulation(physicsClientId=self.client)
            if realtime: time.sleep(1/240)

    def reset(self):
        for idx in self.joints.values(): self.p.resetJointState(self.body,idx,0.0,physicsClientId=self.client)

    def close(self):
        if self.client >= 0:
            self.p.disconnect(self.client); self.client=-1

    def __enter__(self): return self
    def __exit__(self,*_): self.close()
