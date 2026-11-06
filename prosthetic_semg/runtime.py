from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict
import time
import numpy as np
import torch

from .config import CLASS_NAMES, REST_INDEX
from .preprocess import StreamingEMGFilter, normalize
from .train_utils import load_checkpoint


@dataclass
class Decision:
    gesture: str
    class_index: int
    confidence: float
    accepted: bool
    latency_ms: float

    def as_dict(self):
        return asdict(self)


class WindowClassifier:
    def __init__(self, checkpoint: str, threshold=0.70, votes=3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.ckpt = load_checkpoint(checkpoint, self.device)
        self.center = np.asarray(self.ckpt["center"], np.float32)
        self.scale = np.asarray(self.ckpt["scale"], np.float32)
        self.threshold = threshold
        self.votes = deque(maxlen=votes)

    def reset(self): self.votes.clear()

    def classify_filtered_window(self, window: np.ndarray) -> Decision:
        x = normalize(window, self.center, self.scale)
        t0 = time.perf_counter()
        with torch.no_grad():
            prob = torch.softmax(self.model(torch.from_numpy(x[None]).to(self.device)), dim=1)[0]
            conf, pred = prob.max(dim=0)
        latency = (time.perf_counter() - t0) * 1000.0
        raw_idx = int(pred.item()); confidence = float(conf.item())
        accepted = confidence >= self.threshold
        self.votes.append(raw_idx if accepted else REST_INDEX)
        vals, counts = np.unique(np.asarray(self.votes), return_counts=True)
        smooth_idx = int(vals[np.argmax(counts)])
        return Decision(CLASS_NAMES[smooth_idx], smooth_idx, confidence, accepted, latency)


class StreamingClassifier:
    def __init__(self, checkpoint: str, threshold=0.70, votes=3, device=None):
        self.window_model = WindowClassifier(checkpoint, threshold, votes, device)
        meta = self.window_model.ckpt.get("meta", {})
        self.sample_rate = int(meta.get("sample_rate", 200))
        self.window_samples = int(meta.get("window_samples", 50))
        self.stride_samples = int(meta.get("stride_samples", 10))
        channels = int(self.window_model.ckpt["channels"])
        self.filter = StreamingEMGFilter(
            channels=channels,
            sample_rate=self.sample_rate,
            highpass_hz=float(meta.get("highpass_hz", 10.0)),
            lowpass_hz=float(meta.get("lowpass_hz", 95.0)),
            notch_hz=float(meta.get("notch_hz", 50.0)),
        )
        self.buffer = deque(maxlen=self.window_samples)
        self.count = 0

    def reset(self):
        self.buffer.clear(); self.count = 0; self.filter.reset(); self.window_model.reset()

    def push_sample(self, raw_sample: np.ndarray) -> Decision | None:
        filtered = self.filter.push(raw_sample)
        self.buffer.append(filtered)
        self.count += 1
        if len(self.buffer) < self.window_samples:
            return None
        if (self.count - self.window_samples) % self.stride_samples != 0:
            return None
        return self.window_model.classify_filtered_window(np.asarray(self.buffer, np.float32))
