import argparse, json
import numpy as np
import torch
from prosthetic_semg.train_utils import repetition_masks, loso_masks, make_loaders, load_checkpoint, predict, metric_dict, report_text

p=argparse.ArgumentParser()
p.add_argument("--data", default="data/processed/db5_9class.npz")
p.add_argument("--model", default="artifacts/cnn_gru_db5.pt")
p.add_argument("--threshold", type=float, default=0.70)
a=p.parse_args()
z=np.load(a.data); X=z["X"].astype(np.float32); y=z["y"].astype(np.int64); reps=z["repetitions"]; subjects=z["subjects"]
device="cuda" if torch.cuda.is_available() else "cpu"
model, ckpt=load_checkpoint(a.model, device); meta=ckpt.get("meta",{})
if meta.get("protocol") == "loso": masks=loso_masks(subjects,reps,int(meta["held_out_subject"]))
else: masks=repetition_masks(reps)
loaders=make_loaders(X,y,masks,ckpt["center"],ckpt["scale"],512)
yt,yp,conf=predict(model,loaders["test"],device)
m=metric_dict(yt,yp,conf,a.threshold)
print(json.dumps({k:f"{v*100:.2f}%" for k,v in m.items()},indent=2))
print(report_text(yt,yp))
