"""Fast software-path test. It does NOT measure real sEMG accuracy."""
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from prosthetic_semg.config import CLASS_NAMES
from prosthetic_semg.model import CNNGRU
from prosthetic_semg.preprocess import filter_emg, robust_channel_stats, normalize
from prosthetic_semg.train_utils import save_checkpoint, load_checkpoint

rng=np.random.default_rng(7); n=9*24; X=[]; y=[]
t=np.arange(50)/200
for cls in range(9):
    for _ in range(24):
        x=rng.normal(0,0.15,(50,16)).astype(np.float32)
        if cls:
            ch=(cls*2)%16; x[:,ch]+=0.8*np.sin(2*np.pi*(12+cls*3)*t)
            x[:,(ch+1)%16]+=0.4*np.cos(2*np.pi*(8+cls*2)*t)
        X.append(x); y.append(cls)
X=np.asarray(X,np.float32); y=np.asarray(y,np.int64)
center,scale=robust_channel_stats(X.reshape(-1,16)); Xn=normalize(X,center,scale)
loader=DataLoader(TensorDataset(torch.from_numpy(Xn),torch.from_numpy(y)),batch_size=64,shuffle=True)
model=CNNGRU(16,9); opt=torch.optim.Adam(model.parameters(),lr=3e-3); loss_fn=nn.CrossEntropyLoss()
for _ in range(2):
    model.train()
    for xb,yb in loader:
        opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/"m.pt"
    save_checkpoint(p,model,center,scale,{"channels":16,"num_classes":9,"sample_rate":200,"window_samples":50,"stride_samples":10})
    loaded,ckpt=load_checkpoint(p)
    out=loaded(torch.from_numpy(Xn[:4]))
    assert out.shape==(4,9)
f=filter_emg(X[0],200,10,95,50,causal=True)
assert f.shape==(50,16) and np.isfinite(f).all()
print("SELF TEST PASSED")
print("Model, filtering, normalization, training step, checkpoint save/load all work.")
print("This proves software integrity only; DB5 evaluation provides the real accuracy.")
