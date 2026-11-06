import numpy as np
import torch
from prosthetic_semg.config import SignalConfig, CLASS_NAMES, LABEL_TO_INDEX
from prosthetic_semg.model import CNNGRU
from prosthetic_semg.preprocess import filter_emg, robust_channel_stats, normalize
from prosthetic_semg.data import nearest_active_context

def test_model_shape():
    y=CNNGRU(16,len(CLASS_NAMES))(torch.randn(4,50,16)); assert y.shape==(4,9)

def test_signal_config():
    c=SignalConfig(); assert c.window_samples==50 and c.stride_samples==10

def test_preprocess_finite():
    x=np.random.randn(500,16).astype(np.float32); f=filter_emg(x); c,s=robust_channel_stats(f); z=normalize(f,c,s)
    assert z.shape==x.shape and np.isfinite(z).all()

def test_rest_context():
    stim=np.array([0,0,5,5,0,0,6,6,0]); rep=np.array([0,0,1,1,0,0,2,2,0])
    lab,r=nearest_active_context(stim,rep)
    assert np.all(r>0); assert lab[0]==5 and lab[-1]==6

def test_class_mapping():
    assert LABEL_TO_INDEX[0]==0 and LABEL_TO_INDEX[16]==8
