import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prosthetic_semg.train_utils import load_checkpoint

p = argparse.ArgumentParser(description="Fine-tune a trained model on a user's labeled calibration NPZ.")
p.add_argument("--model", default="artifacts/cnn_gru_db5.pt")
p.add_argument("--calibration", required=True, help="NPZ with X [N,T,C] normalized like DB5 and y [N]")
p.add_argument("--out", default="artifacts/cnn_gru_calibrated.pt")
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-4)
a = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, ckpt = load_checkpoint(a.model, device)
z = np.load(a.calibration)
X, y = z["X"].astype(np.float32), z["y"].astype(np.int64)
loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=64, shuffle=True)

# Keep the convolutional feature extractor mostly stable; adapt temporal/head layers.
for param in model.features.parameters():
    param.requires_grad = False

opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=a.lr)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
for epoch in range(1, a.epochs + 1):
    model.train(); losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
        losses.append(float(loss.item()))
    print(f"epoch={epoch} loss={np.mean(losses):.4f}")

ckpt["model_state"] = model.state_dict()
ckpt.setdefault("extra", {})["calibrated"] = True
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, a.out)
print("Saved calibrated model:", a.out)
