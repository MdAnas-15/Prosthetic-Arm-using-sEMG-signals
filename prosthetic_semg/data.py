from __future__ import annotations
from pathlib import Path
import re
import numpy as np
from scipy.io import loadmat

from .config import GESTURE_MAP, LABEL_TO_INDEX, SignalConfig
from .preprocess import ensure_2d_channels_last, filter_emg


def _vec(mat: dict, *names: str) -> np.ndarray:
    for name in names:
        if name in mat:
            return np.asarray(mat[name]).reshape(-1)
    raise KeyError(f"Missing variables: {names}")


def subject_from_path(path: Path) -> int:
    m = re.search(r"[sS](\d+)", path.stem)
    return int(m.group(1)) if m else -1


def discover_db5_e2(raw_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.rglob("*.mat"))
    e2 = [p for p in files if re.search(r"_E2_", p.name, re.I)]
    return e2


def load_ninapro_mat(path: str | Path):
    path = Path(path)
    mat = loadmat(path, squeeze_me=True)
    if "emg" not in mat:
        raise KeyError(f"{path}: missing emg")
    emg = ensure_2d_channels_last(mat["emg"])
    stimulus = _vec(mat, "restimulus", "stimulus").astype(np.int64)
    repetition = _vec(mat, "rerepetition", "repetition").astype(np.int64)
    n = min(len(emg), len(stimulus), len(repetition))
    frequency = int(np.asarray(mat.get("frequency", 200)).reshape(-1)[0])
    return emg[:n], stimulus[:n], repetition[:n], subject_from_path(path), frequency


def nearest_active_context(stimulus: np.ndarray, repetition: np.ndarray):
    """For each sample, attach the nearest non-rest movement label and repetition.

    This lets REST windows inherit a repetition for leakage-safe train/val/test splits,
    and prevents REST after unselected Exercise-B gestures from dominating the dataset.
    """
    stimulus = np.asarray(stimulus, dtype=np.int64)
    repetition = np.asarray(repetition, dtype=np.int64)
    n = len(stimulus)
    prev_idx = np.full(n, -1, np.int64)
    prev_lab = np.zeros(n, np.int64)
    prev_rep = np.zeros(n, np.int64)
    last_i = -1; last_l = 0; last_r = 0
    for i in range(n):
        if stimulus[i] != 0 and repetition[i] > 0:
            last_i, last_l, last_r = i, int(stimulus[i]), int(repetition[i])
        prev_idx[i], prev_lab[i], prev_rep[i] = last_i, last_l, last_r

    next_idx = np.full(n, -1, np.int64)
    next_lab = np.zeros(n, np.int64)
    next_rep = np.zeros(n, np.int64)
    last_i = -1; last_l = 0; last_r = 0
    for i in range(n - 1, -1, -1):
        if stimulus[i] != 0 and repetition[i] > 0:
            last_i, last_l, last_r = i, int(stimulus[i]), int(repetition[i])
        next_idx[i], next_lab[i], next_rep[i] = last_i, last_l, last_r

    ctx_lab = stimulus.copy()
    ctx_rep = repetition.copy()
    rest = stimulus == 0
    for i in np.flatnonzero(rest):
        p_ok = prev_idx[i] >= 0
        n_ok = next_idx[i] >= 0
        if p_ok and n_ok:
            use_prev = (i - prev_idx[i]) <= (next_idx[i] - i)
        else:
            use_prev = p_ok
        if use_prev:
            ctx_lab[i], ctx_rep[i] = prev_lab[i], prev_rep[i]
        elif n_ok:
            ctx_lab[i], ctx_rep[i] = next_lab[i], next_rep[i]
    return ctx_lab, ctx_rep


def _mode_int(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=np.int64)
    if len(x) == 0:
        return 0
    vals, counts = np.unique(x, return_counts=True)
    return int(vals[np.argmax(counts)])


def make_windows(emg, stimulus, repetition, subject: int, cfg: SignalConfig):
    selected = set(GESTURE_MAP.keys())
    selected_active = selected - {0}
    ctx_lab, ctx_rep = nearest_active_context(stimulus, repetition)
    X, y, reps, subs = [], [], [], []
    w, s = cfg.window_samples, cfg.stride_samples

    for start in range(0, len(emg) - w + 1, s):
        stop = start + w
        lab_win = stimulus[start:stop]
        vals, counts = np.unique(lab_win, return_counts=True)
        j = int(np.argmax(counts))
        label = int(vals[j])
        stable = float(counts[j] / w)
        if stable < cfg.min_stable_fraction or label not in selected:
            continue

        context_label = _mode_int(ctx_lab[start:stop])
        rep = _mode_int(ctx_rep[start:stop])
        if rep <= 0:
            continue
        if label == 0 and context_label not in selected_active:
            continue

        X.append(emg[start:stop])
        y.append(LABEL_TO_INDEX[label])
        reps.append(rep)
        subs.append(subject)

    if not X:
        return (np.empty((0, w, emg.shape[1]), np.float32),
                np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))
    return np.asarray(X, np.float32), np.asarray(y, np.int64), np.asarray(reps, np.int64), np.asarray(subs, np.int64)


def prepare_db5(raw_dir: str | Path, out_file: str | Path,
                cfg: SignalConfig = SignalConfig(), subjects: list[int] | None = None):
    paths = discover_db5_e2(raw_dir)
    if subjects is not None:
        wanted = set(subjects)
        paths = [p for p in paths if subject_from_path(p) in wanted]
    if not paths:
        raise FileNotFoundError(f"No NinaPro DB5 E2 .mat files found recursively under {raw_dir}")

    all_X, all_y, all_r, all_s = [], [], [], []
    channels = None
    print(f"Found {len(paths)} Exercise-B files")
    for path in paths:
        emg, stim, rep, subject, fs = load_ninapro_mat(path)
        if fs != cfg.sample_rate:
            raise ValueError(f"{path.name}: expected {cfg.sample_rate} Hz, got {fs}")
        if channels is None:
            channels = emg.shape[1]
        if emg.shape[1] != channels:
            raise ValueError(f"{path.name}: channel mismatch {emg.shape[1]} != {channels}")
        emg = filter_emg(emg, cfg.sample_rate, cfg.highpass_hz, cfg.lowpass_hz,
                         cfg.notch_hz, causal=cfg.causal_filter)
        X, y, r, s = make_windows(emg, stim, rep, subject, cfg)
        print(f"  S{subject}: {len(X):,} windows")
        if len(X):
            all_X.append(X); all_y.append(y); all_r.append(r); all_s.append(s)

    if not all_X:
        raise RuntimeError("No windows were produced")
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    repetitions = np.concatenate(all_r)
    subject_ids = np.concatenate(all_s)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        X=X, y=y, repetitions=repetitions, subjects=subject_ids,
        sample_rate=np.array(cfg.sample_rate),
        window_samples=np.array(cfg.window_samples),
        stride_samples=np.array(cfg.stride_samples),
        causal_filter=np.array(int(cfg.causal_filter)),
    )
    print(f"Saved {len(X):,} windows to {out_file}")
    print(f"Shape: X={X.shape}, classes={sorted(set(y.tolist()))}, channels={X.shape[-1]}")
