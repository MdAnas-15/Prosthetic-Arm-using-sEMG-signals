import argparse, time
from collections import deque
import numpy as np
import torch
from prosthetic_semg.config import CLASS_NAMES, REST_INDEX
from prosthetic_semg.preprocess import normalize
from prosthetic_semg.train_utils import repetition_masks, load_checkpoint

p=argparse.ArgumentParser()
p.add_argument("--data", default="data/processed/db5_9class.npz")
p.add_argument("--model", default="artifacts/cnn_gru_db5.pt")
p.add_argument("--threshold",type=float,default=0.70)
p.add_argument("--max-windows",type=int,default=5000)
a=p.parse_args()
z=np.load(a.data); X=z["X"].astype(np.float32); y=z["y"].astype(np.int64); reps=z["repetitions"]; subs=z["subjects"]
mask=repetition_masks(reps)["test"]; X=X[mask]; y=y[mask]; reps=reps[mask]; subs=subs[mask]
model,ckpt=load_checkpoint(a.model,"cuda" if torch.cuda.is_available() else "cpu"); device=next(model.parameters()).device
center=ckpt["center"]; scale=ckpt["scale"]
n=min(a.max_windows,len(X)); raw_correct=0; smooth_correct=0; accepted=0; lat=[]; votes=deque(maxlen=3); last_key=None
for i in range(n):
    key=(int(subs[i]),int(reps[i]))
    if key != last_key: votes.clear(); last_key=key
    x=normalize(X[i],center,scale)
    t0=time.perf_counter()
    with torch.no_grad():
        prob=torch.softmax(model(torch.from_numpy(x[None]).to(device)),dim=1)[0]
        conf,pred=prob.max(0)
    lat.append((time.perf_counter()-t0)*1000); pred=int(pred); conf=float(conf)
    raw_correct += int(pred==int(y[i]))
    accepted += int(conf>=a.threshold)
    votes.append(pred if conf>=a.threshold else REST_INDEX)
    vals,counts=np.unique(votes,return_counts=True); smooth=int(vals[np.argmax(counts)])
    smooth_correct += int(smooth==int(y[i]))
    if i<25: print(f"{i:04d} true={CLASS_NAMES[int(y[i])]:20s} pred={CLASS_NAMES[smooth]:20s} conf={conf*100:5.1f}%")
print("\nREPLAY SUMMARY")
print(f"Windows tested: {n}")
print(f"Raw accuracy: {100*raw_correct/n:.2f}%")
print(f"Smoothed accuracy: {100*smooth_correct/n:.2f}%")
print(f"Coverage @ {a.threshold:.2f}: {100*accepted/n:.2f}%")
print(f"Mean inference latency: {np.mean(lat):.3f} ms")
print(f"P95 inference latency: {np.percentile(lat,95):.3f} ms")
