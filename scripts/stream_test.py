import argparse
import numpy as np
from prosthetic_semg.config import GESTURE_MAP, LABEL_TO_INDEX, TEST_REPS, CLASS_NAMES
from prosthetic_semg.data import discover_db5_e2, load_ninapro_mat, nearest_active_context, subject_from_path
from prosthetic_semg.runtime import StreamingClassifier

p = argparse.ArgumentParser(
    description="Hardware-like test: stream raw DB5 samples through the causal live pipeline."
)
p.add_argument("--raw", default="data/raw/db5")
p.add_argument("--model", default="artifacts/cnn_gru_db5.pt")
p.add_argument("--subjects", nargs="*", type=int, default=[1])
p.add_argument("--threshold", type=float, default=0.70)
a = p.parse_args()

paths = discover_db5_e2(a.raw)
wanted = set(a.subjects)
paths = [path for path in paths if subject_from_path(path) in wanted]
if not paths:
    raise SystemExit("No requested E2 files found")

selected = set(GESTURE_MAP)
selected_active = selected - {0}
all_true = []
all_pred = []
all_conf = []
lat = []

for path in paths:
    emg, stim, rep, subject, fs = load_ninapro_mat(path)
    rt = StreamingClassifier(a.model, a.threshold)
    if fs != rt.sample_rate:
        raise SystemExit(
            f"{path.name}: dataset sample rate is {fs} Hz but the checkpoint expects "
            f"{rt.sample_rate} Hz"
        )
    ctx_lab, ctx_rep = nearest_active_context(stim, rep)
    last_segment = None
    print(f"Streaming S{subject}: {len(emg):,} raw samples at {fs} Hz")

    for i, sample in enumerate(emg):
        decision = rt.push_sample(sample)
        if decision is None:
            continue

        start = i - rt.window_samples + 1
        if start < 0:
            continue
        labs = stim[start:i + 1]
        vals, counts = np.unique(labs, return_counts=True)
        j = int(np.argmax(counts))
        label = int(vals[j])
        if counts[j] != len(labs) or label not in selected:
            continue
        context = int(np.bincount(ctx_lab[start:i + 1]).argmax())
        repetition = int(np.bincount(ctx_rep[start:i + 1]).argmax())
        if repetition not in TEST_REPS:
            continue
        if label == 0 and context not in selected_active:
            continue

        segment = (int(subject), repetition)
        if segment != last_segment:
            # Keep the continuous causal filter state, but discard probabilities
            # from the previous repetition before classifying this first window.
            rt.window_model.reset()
            last_segment = segment
            decision = rt.window_model.classify_filtered_window(
                np.asarray(rt.buffer, dtype=np.float32)
            )

        all_true.append(LABEL_TO_INDEX[label])
        all_pred.append(decision.class_index)
        all_conf.append(decision.confidence)
        lat.append(decision.latency_ms)

if not all_true:
    raise SystemExit("No valid held-out test decisions produced")

y = np.asarray(all_true)
pred = np.asarray(all_pred)
confidence = np.asarray(all_conf)
print("\nRAW-STREAM END-TO-END RESULTS")
print(f"Decisions: {len(y):,}")
print(f"Accuracy: {100*np.mean(y == pred):.2f}%")
print(f"Coverage @ {a.threshold:.2f}: {100*np.mean(confidence >= a.threshold):.2f}%")
print(f"Mean model latency: {np.mean(lat):.3f} ms")
for i, name in enumerate(CLASS_NAMES):
    mask = y == i
    if mask.any():
        print(f"  {name:20s}: {100*np.mean(pred[mask] == y[mask]):.2f}% ({mask.sum()} decisions)")
