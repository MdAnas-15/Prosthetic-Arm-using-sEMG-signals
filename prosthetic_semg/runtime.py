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
    """Classify pre-filtered 250 ms windows with probability-based temporal smoothing."""

    def __init__(self, checkpoint: str, threshold=0.70, votes=3, device=None):
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if int(votes) < 1:
            raise ValueError("votes must be at least 1")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.ckpt = load_checkpoint(checkpoint, self.device)
        self.center = np.asarray(self.ckpt["center"], np.float32)
        self.scale = np.asarray(self.ckpt["scale"], np.float32)
        self.channels = int(self.ckpt["channels"])
        meta = self.ckpt.get("meta", {})
        self.window_samples = int(meta.get("window_samples", 50))
        if self.channels <= 0 or self.window_samples <= 0:
            raise ValueError("Checkpoint channel count and window length must be positive")
        if self.center.shape != (self.channels,) or self.scale.shape != (self.channels,):
            raise ValueError("Checkpoint normalizer dimensions do not match its channel count")
        if not np.isfinite(self.center).all() or not np.isfinite(self.scale).all():
            raise ValueError("Checkpoint normalizer contains non-finite values")
        if np.any(self.scale <= 0):
            raise ValueError("Checkpoint normalizer scales must all be positive")

        self.threshold = float(threshold)
        self.vote_window = int(votes)
        self.probabilities = deque(maxlen=self.vote_window)

    def reset(self):
        """Clear temporal history at a subject/repetition or recording boundary."""
        self.probabilities.clear()

    def classify_filtered_window(self, window: np.ndarray) -> Decision:
        window = np.asarray(window, dtype=np.float32)
        expected = (self.window_samples, self.channels)
        if window.shape != expected:
            raise ValueError(f"Expected a filtered sEMG window shaped {expected}, got {window.shape}")
        if not np.isfinite(window).all():
            raise ValueError("sEMG window contains NaN or infinite values")

        x = normalize(window, self.center, self.scale)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x[None]).to(self.device))
            current = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        self.probabilities.append(current.astype(np.float32, copy=False))

        # Average class probabilities, rather than majority-voting class IDs.
        # Confidence therefore always belongs to the gesture returned below.
        smoothed = np.mean(np.asarray(self.probabilities), axis=0)
        raw_idx = int(np.argmax(smoothed))
        confidence = float(smoothed[raw_idx])
        accepted = confidence >= self.threshold
        class_index = raw_idx if accepted else REST_INDEX
        gesture = CLASS_NAMES[class_index]
        latency = (time.perf_counter() - t0) * 1000.0
        return Decision(gesture, class_index, confidence, accepted, latency)


class StreamingClassifier:
    """Causal raw-sample filter + rolling-window classifier for live sensor streams."""

    def __init__(self, checkpoint: str, threshold=0.70, votes=3, device=None):
        self.window_model = WindowClassifier(checkpoint, threshold, votes, device)
        meta = self.window_model.ckpt.get("meta", {})
        self.sample_rate = int(meta.get("sample_rate", 200))
        self.window_samples = int(meta.get("window_samples", 50))
        self.stride_samples = int(meta.get("stride_samples", 10))
        channels = self.window_model.channels
        if self.sample_rate <= 0 or self.window_samples <= 0:
            raise ValueError("Checkpoint sample rate and window length must be positive")
        if not 1 <= self.stride_samples <= self.window_samples:
            raise ValueError("Checkpoint stride must be between 1 and the window length")

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
        self.buffer.clear()
        self.count = 0
        self.filter.reset()
        self.window_model.reset()

    def push_sample(self, raw_sample: np.ndarray) -> Decision | None:
        filtered = self.filter.push(raw_sample)
        self.buffer.append(filtered)
        self.count += 1
        if len(self.buffer) < self.window_samples:
            return None
        if (self.count - self.window_samples) % self.stride_samples != 0:
            return None
        return self.window_model.classify_filtered_window(
            np.asarray(self.buffer, dtype=np.float32)
        )
