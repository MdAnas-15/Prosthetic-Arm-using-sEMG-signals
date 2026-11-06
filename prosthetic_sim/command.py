from dataclasses import dataclass, field

@dataclass
class ProstheticCommand:
    gesture: str = "REST"
    confidence: float = 1.0
    shoulder_yaw_deg: float = 0.0
    elbow_flex_deg: float = 25.0
    forearm_roll_deg: float = 0.0
    wrist_pitch_deg: float = 0.0
    wrist_deviation_deg: float = 0.0
    thumb: float = 0.0
    index: float = 0.0
    middle: float = 0.0
    ring: float = 0.0
    little: float = 0.0

    def finger_values(self):
        return {"thumb": self.thumb, "index": self.index, "middle": self.middle,
                "ring": self.ring, "little": self.little}

    def clamp(self):
        self.confidence = min(1.0, max(0.0, float(self.confidence)))
        self.shoulder_yaw_deg = min(80.0, max(-80.0, self.shoulder_yaw_deg))
        self.elbow_flex_deg = min(135.0, max(0.0, self.elbow_flex_deg))
        self.forearm_roll_deg = min(90.0, max(-90.0, self.forearm_roll_deg))
        self.wrist_pitch_deg = min(60.0, max(-50.0, self.wrist_pitch_deg))
        self.wrist_deviation_deg = min(30.0, max(-35.0, self.wrist_deviation_deg))
        for name in ["thumb","index","middle","ring","little"]:
            setattr(self, name, min(1.0, max(0.0, float(getattr(self, name)))))
        return self
