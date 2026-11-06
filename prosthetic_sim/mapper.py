from copy import deepcopy
from .command import ProstheticCommand

GESTURES = [
    "REST", "OPEN_HAND", "CLOSE_HAND", "WRIST_SUPINATION", "WRIST_PRONATION",
    "WRIST_FLEXION", "WRIST_EXTENSION", "RADIAL_DEVIATION", "ULNAR_DEVIATION"
]

class GestureMapper:
    """Stateful gesture-to-pose mapper. REST always returns to neutral."""
    def __init__(self):
        self.state = ProstheticCommand()

    def reset(self):
        self.state = ProstheticCommand()
        return deepcopy(self.state)

    def apply(self, gesture: str, confidence: float = 1.0) -> ProstheticCommand:
        gesture = gesture.upper()
        if gesture not in GESTURES:
            raise ValueError(f"Unknown gesture: {gesture}")
        if gesture == "REST":
            neutral = self.reset()
            neutral.confidence = confidence
            return neutral
        s = self.state
        if gesture == "OPEN_HAND":
            s.thumb = s.index = s.middle = s.ring = s.little = 0.0
        elif gesture == "CLOSE_HAND":
            s.thumb = 0.90
            s.index = s.middle = s.ring = s.little = 1.0
        elif gesture == "WRIST_SUPINATION":
            s.forearm_roll_deg = 80.0
        elif gesture == "WRIST_PRONATION":
            s.forearm_roll_deg = -80.0
        elif gesture == "WRIST_FLEXION":
            s.wrist_pitch_deg = 55.0
        elif gesture == "WRIST_EXTENSION":
            s.wrist_pitch_deg = -45.0
        elif gesture == "RADIAL_DEVIATION":
            s.wrist_deviation_deg = 25.0
        elif gesture == "ULNAR_DEVIATION":
            s.wrist_deviation_deg = -30.0
        s.gesture = gesture
        s.confidence = confidence
        return deepcopy(s.clamp())
