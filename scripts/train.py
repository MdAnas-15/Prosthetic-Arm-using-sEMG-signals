import argparse, json, csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from prosthetic_semg.config import CLASS_NAMES, SignalConfig
from prosthetic_semg.model import CNNGRU
from prosthetic_semg.preprocess import fit_normalizer_from_windows
from prosthetic_semg.train_utils import (
    seed_everything, repetition_masks, loso_masks, make_loaders, predict,
    metric_dict, class_weights, save_checkpoint, save_confusion,
    save_history, report_text
)

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/processed/db5_9class.npz")
p.add_argument("--out", default="artifacts/cnn_gru_db5.pt")
p.add_argument("--epochs", type=int, default=45)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--patience", type=int, default=8)
p.add_argument("--threshold", type=float, default=0.70)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--protocol", choices=["repetition","loso"], default="repetition")
p.add_argument("--held-out-subject", type=int)
a = p.parse_args()

seed_everything(a.seed)
z = np.load(a.data)
X = z["X"].astype(np.float32); y = z["y"].astype(np.int64)
reps = z["repetitions"].astype(np.int64); subjects = z["subjects"].astype(np.int64)

if a.protocol == "repetition":
    masks = repetition_masks(reps)
else:
    if a.held_out_subject is None:
        raise SystemExit("--held-out-subject is required for --protocol loso")
    masks = loso_masks(subjects, reps, a.held_out_subject)

for name, mask in masks.items():
    if not mask.any(): raise SystemExit(f"{name} split is empty")
    if len(np.unique(y[mask])) != len(CLASS_NAMES):
        raise SystemExit(f"{name} split does not contain all 9 classes: {np.unique(y[mask])}")

center, scale = fit_normalizer_from_windows(X, masks["train"], seed=a.seed)
loaders = make_loaders(X, y, masks, center, scale, a.batch_size)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
model = CNNGRU(X.shape[-1], len(CLASS_NAMES)).to(device)
weights = class_weights(y[masks["train"]], len(CLASS_NAMES)).to(device)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
optimizer = AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

reports = Path("reports"); reports.mkdir(exist_ok=True)
best = -1.0; stale = 0
history = {"epoch":[], "train_loss":[], "val_balanced_accuracy":[]}
cfg = SignalConfig()

for epoch in range(1, a.epochs+1):
    model.train(); losses=[]
    for xb, yb in tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        losses.append(float(loss.item()))
    vy, vp, vc = predict(model, loaders["val"], device)
    vm = metric_dict(vy, vp, vc, a.threshold)
    scheduler.step(vm["balanced_accuracy"])
    mean_loss = float(np.mean(losses))
    history["epoch"].append(epoch); history["train_loss"].append(mean_loss)
    history["val_balanced_accuracy"].append(vm["balanced_accuracy"])
    print(f"epoch={epoch:02d} loss={mean_loss:.4f} val_acc={vm['accuracy']*100:.2f}% val_bal_acc={vm['balanced_accuracy']*100:.2f}%")
    if vm["balanced_accuracy"] > best:
        best = vm["balanced_accuracy"]; stale = 0
        meta = {
            "channels": X.shape[-1], "num_classes": len(CLASS_NAMES),
            "classes": CLASS_NAMES, "protocol": a.protocol,
            "held_out_subject": a.held_out_subject, "best_val_balanced_accuracy": best,
            "sample_rate": int(z.get("sample_rate", 200)),
            "window_samples": int(z.get("window_samples", 50)),
            "stride_samples": int(z.get("stride_samples", 10)),
            "highpass_hz": cfg.highpass_hz, "lowpass_hz": cfg.lowpass_hz,
            "notch_hz": cfg.notch_hz, "causal_filter": True,
            "threshold": a.threshold,
        }
        save_checkpoint(a.out, model, center, scale, meta)
    else:
        stale += 1
        if stale >= a.patience:
            print("early stopping"); break

save_history(history, reports/"training_history.png")
model, ckpt = __import__('prosthetic_semg.train_utils', fromlist=['load_checkpoint']).load_checkpoint(a.out, device)
ty, tp, tc = predict(model, loaders["test"], device)
metrics = metric_dict(ty, tp, tc, a.threshold)
print("\nFINAL HELD-OUT TEST RESULTS")
for k, v in metrics.items(): print(f"{k:24s}: {v*100:.2f}%" if 'latency' not in k else f"{v:.3f}")
print("\n" + report_text(ty, tp))

save_confusion(ty, tp, reports/"confusion_matrix.png")
with open(reports/"metrics.json", "w") as f: json.dump(metrics, f, indent=2)
with open(reports/"classification_report.txt", "w") as f: f.write(report_text(ty, tp))

# Per-subject final-test accuracy for the repetition protocol.
if a.protocol == "repetition":
    test_subjects = subjects[masks["test"]]
    rows=[]
    for s in sorted(np.unique(test_subjects)):
        m = test_subjects == s
        sm = metric_dict(ty[m], tp[m], tc[m], a.threshold)
        rows.append([int(s), sm["accuracy"], sm["balanced_accuracy"], sm["macro_f1"]])
    with open(reports/"per_subject_metrics.csv", "w", newline="") as f:
        w=csv.writer(f); w.writerow(["subject","accuracy","balanced_accuracy","macro_f1"]); w.writerows(rows)

print(f"\nSaved model: {a.out}")
print("Saved reports/metrics.json, classification_report.txt, confusion_matrix.png, training_history.png")
