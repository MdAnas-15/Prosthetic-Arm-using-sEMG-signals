from __future__ import annotations
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, tf2sos


def ensure_2d_channels_last(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"EMG must be 2-D, got {x.shape}")
    if x.shape[0] <= 32 and x.shape[0] < x.shape[1]:
        x = x.T
    return x


def design_sos(sample_rate: int, highpass_hz: float, lowpass_hz: float, notch_hz: float | None):
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    nyq = sample_rate / 2.0
    lo = highpass_hz / nyq
    hi = min(lowpass_hz, nyq - 1.0) / nyq
    if not (0 < lo < hi < 1):
        raise ValueError(f"Invalid filter band {highpass_hz}-{lowpass_hz} Hz for fs={sample_rate}")
    band = butter(4, [lo, hi], btype="bandpass", output="sos")
    if notch_hz and 0 < notch_hz < nyq:
        b, a = iirnotch(notch_hz / nyq, Q=30.0)
        notch = tf2sos(b, a)
        return np.vstack([band, notch]).astype(np.float64)
    return band.astype(np.float64)


def filter_emg(x: np.ndarray, sample_rate=200, highpass_hz=10.0, lowpass_hz=95.0,
               notch_hz=50.0, causal=True) -> np.ndarray:
    x = ensure_2d_channels_last(x)
    if not np.isfinite(x).all():
        raise ValueError("EMG input contains NaN or infinite values")
    sos = design_sos(sample_rate, highpass_hz, lowpass_hz, notch_hz)
    if causal:
        y = sosfilt(sos, x, axis=0)
    else:
        y = sosfiltfilt(sos, x, axis=0)
    return np.asarray(y, dtype=np.float32)


class StreamingEMGFilter:
    def __init__(self, channels: int, sample_rate=200, highpass_hz=10.0,
                 lowpass_hz=95.0, notch_hz=50.0):
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.sos = design_sos(sample_rate, highpass_hz, lowpass_hz, notch_hz)
        self.zi = np.zeros((self.sos.shape[0], 2, self.channels), dtype=np.float64)

    def reset(self):
        self.zi[...] = 0.0

    def push(self, sample: np.ndarray) -> np.ndarray:
        sample = np.asarray(sample, dtype=np.float64)
        if sample.ndim != 1 or sample.size != self.channels:
            raise ValueError(f"Expected one sample with {self.channels} channels, got {sample.shape}")
        if not np.isfinite(sample).all():
            raise ValueError("Raw sEMG sample contains NaN or infinite values")
        y, self.zi = sosfilt(self.sos, sample.reshape(1, -1), axis=0, zi=self.zi)
        return y[0].astype(np.float32)


def robust_channel_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    center = np.median(x, axis=0).astype(np.float32)
    mad = np.median(np.abs(x - center), axis=0).astype(np.float32)
    scale = 1.4826 * mad
    scale[scale < 1e-6] = 1.0
    return center, scale


def fit_normalizer_from_windows(X: np.ndarray, mask: np.ndarray, max_windows: int = 40000, seed: int = 42):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError("Training mask is empty")
    if len(idx) > max_windows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_windows, replace=False)
    flat = X[idx].reshape(-1, X.shape[-1])
    return robust_channel_stats(flat)


def normalize(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) - center) / scale).astype(np.float32)
