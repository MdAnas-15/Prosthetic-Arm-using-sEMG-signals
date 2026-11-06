from __future__ import annotations
from pathlib import Path
import csv, time
import numpy as np
from prosthetic_sim.mapper import GestureMapper
from prosthetic_sim.simulator import PyBulletProstheticSimulator
from prosthetic_semg.runtime import WindowClassifier, StreamingClassifier
from prosthetic_semg.train_utils import repetition_masks
from prosthetic_semg.config import CLASS_NAMES
from prosthetic_semg.hardware import read_csv_stream, read_serial_stream

def _load_test_windows(data_path):
    z = np.load(data_path)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.int64)
    mask = repetition_masks(z["repetitions"])["test"]
    return X[mask], y[mask]

def run_semg_output(data_path="data/processed/db5_9class.npz",
                    model_path="artifacts/cnn_gru_db5.pt", limit=1000, threshold=0.70,
                    log_path="logs/semg_predictions.csv"):
    """sEMG-only mode: show predictions and confidence, never start PyBullet."""
    X, y = _load_test_windows(data_path)
    if not len(X):
        raise RuntimeError("No held-out sEMG windows found in the processed dataset.")
    clf = WindowClassifier(model_path, threshold=threshold)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    n = min(max(1, int(limit)), len(X))
    correct = 0
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["window", "true_gesture", "predicted_gesture", "confidence", "accepted", "latency_ms"])
        for i in range(n):
            decision = clf.classify_filtered_window(X[i])
            truth = CLASS_NAMES[int(y[i])]
            correct += int(decision.gesture == truth)
            writer.writerow([i, truth, decision.gesture, decision.confidence,
                             decision.accepted, decision.latency_ms])
            print(f"{i+1:04d}/{n:04d} | predicted={decision.gesture:20s} | "
                  f"confidence={decision.confidence*100:5.1f}% | accepted={decision.accepted}")
    print(f"Raw window accuracy: {100*correct/n:.2f}% ({correct}/{n})")
    print("Prediction log:", log_path)

def replay_processed(data_path, model_path, gui=True, limit=1000, threshold=0.70,
                     command_hold=0.8, log_path="logs/semg_simulation_replay.csv",
                     stable_votes=3):
    """Replay test-set windows; apply only stable gesture transitions and label each movement."""
    X, y = _load_test_windows(data_path)
    if not len(X):
        raise RuntimeError("No held-out sEMG windows found in the processed dataset.")
    clf = WindowClassifier(model_path, threshold=threshold, votes=stable_votes)
    mapper = GestureMapper()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    n = min(max(1, int(limit)), len(X))
    raw_correct = 0
    events = 0
    previous = None
    with PyBulletProstheticSimulator(gui=gui) as sim, \
         open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event", "window_index", "true_gesture", "predicted_gesture",
                         "confidence", "accepted", "latency_ms", "hold_seconds"])
        for i in range(n):
            decision = clf.classify_filtered_window(X[i])
            truth = CLASS_NAMES[int(y[i])]
            raw_correct += int(decision.gesture == truth)
            gesture = decision.gesture if decision.accepted else "REST"
            if gesture == previous:
                continue
            previous = gesture
            command = mapper.apply(gesture, decision.confidence)
            sim.apply(command)
            sim.step(max(1, int(240 * command_hold)), realtime=gui)
            events += 1
            writer.writerow([events, i, truth, gesture, decision.confidence,
                             decision.accepted, decision.latency_ms, command_hold])
            f.flush()
            print(f"LABEL {events:03d}: {gesture} | confidence={decision.confidence*100:.1f}% "
                  f"| source window={i} | held={command_hold:.2f}s")
            # Keep the pose visible for the requested interval; never stream jittery
            # per-window motor targets into the physics engine.
        if gui:
            sim.step(240, realtime=True)
    print(f"Raw window accuracy (not movement-event accuracy): {100*raw_correct/n:.2f}% ({n} windows)")
    print(f"Movement events applied: {events}")
    print("Movement log:", log_path)

def live_stream(model_path, gui=True, csv_path=None, serial_port=None, baud=115200,
                threshold=0.70, log_path="logs/live_semg_simulation.csv"):
    clf = StreamingClassifier(model_path, threshold=threshold)
    mapper = GestureMapper()
    channels = int(clf.window_model.ckpt["channels"])
    if csv_path:
        stream = read_csv_stream(csv_path, channels)
    elif serial_port:
        stream = read_serial_stream(serial_port, baud, channels)
    else:
        raise ValueError("Provide csv_path or serial_port")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    previous = None
    with PyBulletProstheticSimulator(gui=gui) as sim, \
         open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gesture", "confidence", "accepted", "latency_ms"])
        t0 = time.perf_counter()
        for sample in stream:
            decision = clf.push_sample(sample)
            if decision is None:
                continue
            gesture = decision.gesture if decision.accepted else "REST"
            if gesture != previous:
                command = mapper.apply(gesture, decision.confidence)
                sim.apply(command)
                sim.step(max(1, clf.stride_samples), realtime=gui)
                writer.writerow([time.perf_counter()-t0, gesture, decision.confidence,
                                 decision.accepted, decision.latency_ms])
                f.flush()
                print(f"LABEL: {gesture} | confidence={decision.confidence*100:.1f}%")
                previous = gesture
