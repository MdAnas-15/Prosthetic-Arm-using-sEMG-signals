from dataclasses import dataclass

GESTURE_MAP = {
    0: "REST",
    5: "OPEN_HAND",
    6: "CLOSE_HAND",
    9: "WRIST_SUPINATION",
    10: "WRIST_PRONATION",
    13: "WRIST_FLEXION",
    14: "WRIST_EXTENSION",
    15: "RADIAL_DEVIATION",
    16: "ULNAR_DEVIATION",
}

CLASS_IDS = list(GESTURE_MAP.keys())
CLASS_NAMES = [GESTURE_MAP[k] for k in CLASS_IDS]
LABEL_TO_INDEX = {label: i for i, label in enumerate(CLASS_IDS)}
INDEX_TO_LABEL = {i: label for label, i in LABEL_TO_INDEX.items()}
REST_INDEX = LABEL_TO_INDEX[0]

@dataclass(frozen=True)
class SignalConfig:
    sample_rate: int = 200
    window_ms: int = 250
    stride_ms: int = 50
    highpass_hz: float = 10.0
    lowpass_hz: float = 95.0
    notch_hz: float = 50.0
    min_stable_fraction: float = 1.0
    causal_filter: bool = True

    @property
    def window_samples(self) -> int:
        return round(self.sample_rate * self.window_ms / 1000)

    @property
    def stride_samples(self) -> int:
        return round(self.sample_rate * self.stride_ms / 1000)

TRAIN_REPS = (1, 3, 4)
VAL_REPS = (6,)
TEST_REPS = (2, 5)
