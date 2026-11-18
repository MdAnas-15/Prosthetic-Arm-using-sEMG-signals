from __future__ import annotations
from pathlib import Path
import json, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from .config import CLASS_NAMES, TRAIN_REPS, VAL_REPS, TEST_REPS
from .model import CNNGRU
from .preprocess import normalize


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def repetition_masks(reps):
    return {
        "train": np.isin(reps, TRAIN_REPS),
        "val": np.isin(reps, VAL_REPS),
        "test": np.isin(reps, TEST_REPS),
    }


def loso_masks(subjects, reps, held_out_subject: int):
    held = subjects == held_out_subject
    other = ~held
    return {
        "train": other & ~np.isin(reps, VAL_REPS),
        "val": other & np.isin(reps, VAL_REPS),
        "test": held,
    }


class EMGDataset(Dataset):
    def __init__(self, X, y, center, scale, augment=False):
        self.X, self.y = X, y
        self.center = center; self.scale = scale; self.augment = augment
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = normalize(self.X[i], self.center, self.scale)
        if self.augment:
            gain = np.random.normal(1.0, 0.04, size=(1, x.shape[1])).astype(np.float32)
            noise = np.random.normal(0.0, 0.02, size=x.shape).astype(np.float32)
            x = x * gain + noise
        return torch.from_numpy(x), torch.tensor(int(self.y[i]), dtype=torch.long)


def make_loaders(X, y, masks, center, scale, batch_size=256):
    result = {}
    for split, mask in masks.items():
        ds = EMGDataset(X[mask], y[mask], center, scale, augment=(split == "train"))
        result[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)
        print(f"{split:>5}: {len(ds):,} windows")
    return result


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); ys=[]; ps=[]; cs=[]
    for xb, yb in loader:
        prob = torch.softmax(model(xb.to(device)), dim=1)
        conf, pred = prob.max(dim=1)
        ys.append(yb.numpy()); ps.append(pred.cpu().numpy()); cs.append(conf.cpu().numpy())
    if not ys: return np.array([]), np.array([]), np.array([])
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(cs)


def metric_dict(y, pred, conf=None, threshold=0.70):
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    if conf is not None and len(conf):
        accepted = conf >= threshold
        out["coverage_at_threshold"] = float(accepted.mean())
        out["accepted_accuracy"] = float(accuracy_score(y[accepted], pred[accepted])) if accepted.any() else 0.0
        out["mean_confidence"] = float(conf.mean())
    return out


def class_weights(y_train, n_classes):
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1
    w = len(y_train) / (n_classes * counts)
    w = np.clip(w, 0.25, 4.0)
    return torch.tensor(w, dtype=torch.float32)


def save_checkpoint(path, model, center, scale, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "channels": int(meta["channels"]),
        "num_classes": int(meta["num_classes"]),
        "center": np.asarray(center, np.float32),
        "scale": np.asarray(scale, np.float32),
        "meta": meta,
    }, path)


def load_checkpoint(path, device="cpu"):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained sEMG checkpoint not found: {checkpoint_path}. "
            "Expected artifacts/cnn_gru_db5.pt; restore the trained model before running inference."
        )
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read sEMG checkpoint {checkpoint_path}: {exc}") from exc

    required = {"model_state", "channels", "num_classes", "center", "scale"}
    missing = sorted(required.difference(ckpt if isinstance(ckpt, dict) else {}))
    if missing:
        raise ValueError(f"Invalid sEMG checkpoint; missing fields: {', '.join(missing)}")

    channels = int(ckpt["channels"])
    num_classes = int(ckpt["num_classes"])
    if channels <= 0:
        raise ValueError("Checkpoint channel count must be positive")
    if num_classes != len(CLASS_NAMES):
        raise ValueError(
            f"Checkpoint has {num_classes} output classes; this project expects "
            f"{len(CLASS_NAMES)} in this order: {', '.join(CLASS_NAMES)}"
        )

    meta = ckpt.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("Checkpoint metadata must be a dictionary")
    saved_classes = meta.get("classes")
    if saved_classes is not None and list(saved_classes) != CLASS_NAMES:
        raise ValueError(
            "Checkpoint class order does not match the current gesture map. "
            "Retrain or restore the matching model/configuration."
        )

    center = np.asarray(ckpt["center"], dtype=np.float32)
    scale = np.asarray(ckpt["scale"], dtype=np.float32)
    if center.shape != (channels,) or scale.shape != (channels,):
        raise ValueError(
            f"Checkpoint normalizer must contain {channels} channel values "
            f"(center={center.shape}, scale={scale.shape})"
        )
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Checkpoint normalizer contains invalid or non-positive values")

    model = CNNGRU(channels, num_classes)
    try:
        model.load_state_dict(ckpt["model_state"])
    except Exception as exc:
        raise ValueError(f"Checkpoint weights do not match the CNN-GRU architecture: {exc}") from exc
    model.to(device).eval()
    return model, ckpt


def save_confusion(y, pred, path):
    cm = confusion_matrix(y, pred, labels=range(len(CLASS_NAMES)), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Recall-normalized")
    ax.set(xticks=np.arange(len(CLASS_NAMES)), yticks=np.arange(len(CLASS_NAMES)),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True", title="Normalized confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def save_history(history, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="train loss")
    ax2 = ax.twinx()
    ax2.plot(history["epoch"], history["val_balanced_accuracy"], label="val balanced accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax2.set_ylabel("Balanced accuracy")
    ax.set_title("Training history")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def report_text(y, pred):
    return classification_report(y, pred, target_names=CLASS_NAMES, digits=4, zero_division=0)
