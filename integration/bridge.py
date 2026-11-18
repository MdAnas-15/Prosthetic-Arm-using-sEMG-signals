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
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Processed sEMG dataset not found: {data_path}")
    with np.load(data_path) as z:
        required = {"X", "y", "repetitions", "subjects"}
        missing = sorted(required.difference(z.files))
        if missing:
            raise ValueError(f"Processed dataset is missing arrays: {', '.join(missing)}")
        X = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
        repetitions = z["repetitions"].astype(np.int64)
        subjects = z["subjects"].astype(np.int64)

    if X.ndim != 3 or len(X) != len(y) or len(y) != len(repetitions) or len(y) != len(subjects):
        raise ValueError("Processed dataset arrays have incompatible lengths or shapes")
    if not len(X):
        raise RuntimeError("Processed sEMG dataset contains no windows")

    mask = repetition_masks(repetitions)["test"]
    if not mask.any():
        raise RuntimeError("No held-out test repetitions (2 and 5) found in the processed dataset")
    return X[mask], y[mask], repetitions[mask], subjects[mask]


def run_semg_output(data_path="data/processed/db5_9class.npz",
                    model_path="artifacts/cnn_gru_db5.pt", limit=1000, threshold=0.70,
                    log_path="logs/semg_predictions.csv"):
    """sEMG-only mode: show smoothed predictions and confidence, never start PyBullet."""
    X, y, repetitions, subjects = _load_test_windows(data_path)
    clf = WindowClassifier(model_path, threshold=threshold)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    n = min(max(1, int(limit)), len(X))
    correct = 0
    last_segment = None
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window", "subject", "repetition", "true_gesture",
            "predicted_gesture", "confidence", "accepted", "latency_ms"
        ])
        for i in range(n):
            segment = (int(subjects[i]), int(repetitions[i]))
            if segment != last_segment:
                clf.reset()
                last_segment = segment
            decision = clf.classify_filtered_window(X[i])
            truth = CLASS_NAMES[int(y[i])]
            correct += int(decision.gesture == truth)
            writer.writerow([
                i, segment[0], segment[1], truth, decision.gesture,
                decision.confidence, decision.accepted, decision.latency_ms
            ])
            print(
                f"{i+1:04d}/{n:04d} | {decision.gesture:20s} | "
                f"confidence={decision.confidence*100:5.1f}% | accepted={decision.accepted}"
            )
    print(f"Smoothed window accuracy: {100*correct/n:.2f}% ({correct}/{n})")
    print("Prediction log:", log_path)


def replay_processed(data_path, model_path, gui=True, limit=1000, threshold=0.80,
                     command_hold=1.0, inter_movement_gap=1.5,
                     log_path="logs/semg_simulation_replay.csv", stable_votes=5):
    """Replay held-out sEMG windows as stable, labelled poses with neutral pauses."""
    X, y, repetitions, subjects = _load_test_windows(data_path)
    clf = WindowClassifier(model_path, threshold=threshold, votes=stable_votes)
    mapper = GestureMapper()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    n = min(max(1, int(limit)), len(X))
    correct = 0
    events = 0
    previous = None
    last_segment = None
    hold_seconds = max(0.5, float(command_hold))
    gap_seconds = max(1.5, float(inter_movement_gap))

    with PyBulletProstheticSimulator(gui=gui) as sim, \
         open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event", "window_index", "subject", "repetition",
            "true_gesture", "predicted_gesture", "confidence", "accepted",
            "latency_ms", "hold_seconds", "neutral_gap_seconds"
        ])
        for i in range(n):
            segment = (int(subjects[i]), int(repetitions[i]))
            if segment != last_segment:
                # Never carry the vote history or previous output into a new recording.
                clf.reset()
                previous = None
                last_segment = segment
                print(f"\nDATA SEGMENT: subject S{segment[0]}, repetition {segment[1]}")

            decision = clf.classify_filtered_window(X[i])
            truth = CLASS_NAMES[int(y[i])]
            correct += int(decision.gesture == truth)
            gesture = decision.gesture if decision.accepted else "REST"
            if gesture == previous:
                continue
            previous = gesture

            # Apply absolute targets so one gesture cannot inherit another gesture's joints.
            mapper.reset()
            command = mapper.apply(gesture, decision.confidence)
            sim.apply(command)
            sim.step(max(1, int(240 * hold_seconds)), realtime=gui)
            events += 1
            writer.writerow([
                events, i, segment[0], segment[1], truth, gesture,
                decision.confidence, decision.accepted, decision.latency_ms,
                hold_seconds, gap_seconds
            ])
            f.flush()
            print(
                f"LABEL {events:03d}: {gesture} | confidence={decision.confidence*100:.1f}% "
                f"| subject=S{segment[0]} repetition={segment[1]} window={i} "
                f"| held={hold_seconds:.2f}s"
            )

            # Return to neutral and visibly pause before accepting another active pose.
            if gesture != "REST":
                neutral = mapper.apply("REST", decision.confidence)
                sim.apply(neutral)
                sim.step(max(1, int(240 * gap_seconds)), realtime=gui)
                print(f"NEUTRAL GAP: {gap_seconds:.1f}s before next movement")

        if gui:
            sim.step(120, realtime=True)
    print(f"Smoothed window accuracy (not movement-event accuracy): {100*correct/n:.2f}% ({n} windows)")
    print(f"Movement events applied: {events}")
    print("Movement log:", log_path)


def live_stream(model_path, gui=True, csv_path=None, serial_port=None, baud=115200,
                threshold=0.70, log_path="logs/live_semg_simulation.csv",
                inter_movement_gap=1.5):
    clf = StreamingClassifier(model_path, threshold=threshold)
    mapper = GestureMapper()
    channels = clf.window_model.channels
    if csv_path:
        stream = read_csv_stream(csv_path, channels)
    elif serial_port:
        stream = read_serial_stream(serial_port, baud, channels)
    else:
        raise ValueError("Provide csv_path or serial_port")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    previous = None
    next_command_at = 0.0
    gap_seconds = max(1.5, float(inter_movement_gap))
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
            now = time.perf_counter()
            if gesture != previous and now >= next_command_at:
                mapper.reset()
                command = mapper.apply(gesture, decision.confidence)
                sim.apply(command)
                sim.step(max(1, clf.stride_samples), realtime=gui)
                writer.writerow([now-t0, gesture, decision.confidence,
                                 decision.accepted, decision.latency_ms])
                f.flush()
                print(f"LABEL: {gesture} | confidence={decision.confidence*100:.1f}%")
                previous = gesture
                next_command_at = now + gap_seconds
