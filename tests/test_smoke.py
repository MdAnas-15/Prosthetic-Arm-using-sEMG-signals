from collections import deque

import numpy as np
import pytest
import torch

from prosthetic_semg.config import SignalConfig, CLASS_NAMES, LABEL_TO_INDEX
from prosthetic_semg.model import CNNGRU
from prosthetic_semg.preprocess import (
    filter_emg, robust_channel_stats, normalize, StreamingEMGFilter
)
from prosthetic_semg.data import nearest_active_context
from prosthetic_semg.runtime import WindowClassifier


def test_model_shape():
    y = CNNGRU(16, len(CLASS_NAMES))(torch.randn(4, 50, 16))
    assert y.shape == (4, 9)


def test_signal_config():
    c = SignalConfig()
    assert c.window_samples == 50 and c.stride_samples == 10


def test_preprocess_finite():
    x = np.random.randn(500, 16).astype(np.float32)
    f = filter_emg(x)
    c, s = robust_channel_stats(f)
    z = normalize(f, c, s)
    assert z.shape == x.shape and np.isfinite(z).all()


def test_filter_rejects_non_finite_input():
    x = np.zeros((100, 16), dtype=np.float32)
    x[10, 3] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        filter_emg(x)


def test_streaming_filter_rejects_invalid_sample():
    filt = StreamingEMGFilter(16)
    with pytest.raises(ValueError, match="NaN or infinite"):
        filt.push(np.full(16, np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="Expected one sample"):
        filt.push(np.zeros(15, dtype=np.float32))


def test_rest_context():
    stim = np.array([0, 0, 5, 5, 0, 0, 6, 6, 0])
    rep = np.array([0, 0, 1, 1, 0, 0, 2, 2, 0])
    lab, r = nearest_active_context(stim, rep)
    assert np.all(r > 0)
    assert lab[0] == 5 and lab[-1] == 6


def test_class_mapping():
    assert LABEL_TO_INDEX[0] == 0 and LABEL_TO_INDEX[16] == 8


def test_window_classifier_probability_smoothing_matches_confidence_and_resets():
    class FakeModel:
        def __init__(self):
            self.outputs = [
                [0, 6, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 6, 0, 0, 0, 0, 0, 0],
                [0, 0, 6, 0, 0, 0, 0, 0, 0],
            ]

        def __call__(self, _batch):
            return torch.tensor([self.outputs.pop(0)], dtype=torch.float32)

    classifier = WindowClassifier.__new__(WindowClassifier)
    classifier.device = "cpu"
    classifier.model = FakeModel()
    classifier.center = np.zeros(16, dtype=np.float32)
    classifier.scale = np.ones(16, dtype=np.float32)
    classifier.channels = 16
    classifier.window_samples = 50
    classifier.threshold = 0.60
    classifier.probabilities = deque(maxlen=2)

    window = np.zeros((50, 16), dtype=np.float32)
    classifier.classify_filtered_window(window)
    mixed = classifier.classify_filtered_window(window)
    # The averaged distribution is ambiguous; it must not report the current raw
    # class's high confidence beside a different, stale majority-vote gesture.
    assert mixed.gesture == "REST"
    assert mixed.class_index == 0
    assert mixed.confidence < classifier.threshold
    assert not mixed.accepted

    classifier.reset()
    after_reset = classifier.classify_filtered_window(window)
    assert after_reset.gesture == "CLOSE_HAND"
    assert after_reset.class_index == 2
    assert after_reset.accepted
    assert after_reset.confidence > classifier.threshold


def test_window_classifier_checks_input_shape():
    classifier = WindowClassifier.__new__(WindowClassifier)
    classifier.window_samples = 50
    classifier.channels = 16
    with pytest.raises(ValueError, match="Expected a filtered sEMG window"):
        classifier.classify_filtered_window(np.zeros((20, 16), dtype=np.float32))
