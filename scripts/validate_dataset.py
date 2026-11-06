import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from prosthetic_semg.config import CLASS_NAMES, TRAIN_REPS, VAL_REPS, TEST_REPS
from prosthetic_semg.train_utils import repetition_masks

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/processed/db5_9class.npz")
p.add_argument("--reports", default="reports")
a = p.parse_args()

z = np.load(a.data)
required = ["X","y","repetitions","subjects"]
missing = [k for k in required if k not in z]
if missing: raise SystemExit(f"FAIL: missing arrays {missing}")
X = z["X"]; y = z["y"]; reps = z["repetitions"]; subs = z["subjects"]
checks = {}
checks["shape_3d"] = X.ndim == 3
checks["window_50"] = X.shape[1] == 50
checks["channels_16"] = X.shape[2] == 16
checks["finite"] = bool(np.isfinite(X).all())
checks["classes_0_to_8"] = np.array_equal(np.unique(y), np.arange(len(CLASS_NAMES)))
checks["no_rep_zero"] = not np.any(reps == 0)
checks["subjects_present"] = len(np.unique(subs)) >= 1
masks = repetition_masks(reps)
for split, mask in masks.items():
    checks[f"{split}_nonempty"] = bool(mask.any())
    checks[f"{split}_all_classes"] = np.array_equal(np.unique(y[mask]), np.arange(len(CLASS_NAMES)))

print(f"X shape: {X.shape}")
print(f"Subjects: {np.unique(subs).tolist()}")
print(f"Repetitions: {np.unique(reps).tolist()}")
print("\nClass counts:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i}: {name:20s} {int((y==i).sum()):,}")
print("\nSplit sizes:")
for split, mask in masks.items():
    print(f"  {split:>5}: {int(mask.sum()):,} | classes {np.unique(y[mask]).tolist()}")
print("\nChecks:")
for k, v in checks.items(): print(f"  {'PASS' if v else 'FAIL'}  {k}")

Path(a.reports).mkdir(parents=True, exist_ok=True)
with open(Path(a.reports)/"dataset_validation.json", "w") as f:
    json.dump({"shape": list(X.shape), "subjects": np.unique(subs).tolist(),
               "repetitions": np.unique(reps).tolist(), "checks": checks}, f, indent=2)

counts = np.bincount(y, minlength=len(CLASS_NAMES))
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(np.arange(len(CLASS_NAMES)), counts)
ax.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
ax.set_ylabel("Windows"); ax.set_title("DB5 9-class window distribution")
fig.tight_layout(); fig.savefig(Path(a.reports)/"class_distribution.png", dpi=160); plt.close(fig)

if not all(checks.values()):
    raise SystemExit("Dataset validation FAILED. Fix failed checks before training.")
print("\nDATASET VALIDATION PASSED")
