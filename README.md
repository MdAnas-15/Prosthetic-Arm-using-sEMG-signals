# Prosthetic Arm — sEMG + Camera + 3D Simulation

**Recommended environment: Windows 10/11, CPython 3.11 x64.**

Run `python main.py` from the repository root to open the three-mode launcher.

## Three showcase modes

1. **sEMG signal output** — CNN–GRU predictions, confidence, acceptance and latency on held-out NinaPro DB5 test repetitions. No simulation is opened.
2. **Camera simulation** — MediaPipe hand tracking drives the PyBullet arm. Hold a pose for about 0.55 seconds; accepted movements have at least a 1.5-second cooldown. REST returns the virtual hand to neutral. The camera classifier uses wrist-relative landmark distances instead of image-Y-only finger tests. Press Q in the camera window to exit. Each accepted pose is labelled and logged.
3. **Combined sEMG + simulation** — held-out test windows are classified, stabilised through five-window temporal voting with a confidence threshold of 0.80, and converted into absolute movement transitions. Each pose is held for 1.0 second, then the arm returns to neutral for a minimum 1.5-second gap before the next movement. Events are labelled in the console and written to CSV.

**Honest scope:** the default combined showcase replays recorded held-out sEMG data. It is not live muscle-electrode acquisition. The CLI also supports CSV/serial streaming when a compatible sensor stream is supplied.

## One-time Windows setup

Use a clean Python 3.11 x64 environment. Do not reuse a venv created by another Python version.

```powershell
# From the repository root:
deactivate  # only if an old venv is active; ignore if command is not found
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# Install NumPy from a prebuilt wheel, not a local GCC source build:
python -m pip install --only-binary=:all: numpy==1.26.4

# Install remaining pinned packages:
python -m pip install -r requirements.txt

# Install the local package and run tests:
python -m pip install -e .
python -m pytest -q
```

If the NumPy wheel command says no matching distribution, check `python --version` and `py -0p`: the active interpreter must be CPython 3.11, 64-bit. Do not try to compile NumPy with the old MinGW/TDM GCC compiler.

If PyBullet installation itself fails, preserve the full pip error. Do not silently substitute another physics engine; install the Windows C++ Build Tools with MSVC v143 and the Windows SDK, then retry `python -m pip install pybullet==3.2.7`.

## Launch

```powershell
python main.py
```

Command-line alternatives:

```powershell
python main.py camera
python main.py camera --camera 1
python main.py semg-replay --limit 1000
python main.py live --csv recordings/user01.csv
python main.py live --serial COM5 --baud 115200
python main.py demo
```

Camera mode requires a working webcam and Windows camera permission. Close other apps using the webcam before launching it.

## Required trained assets

The signal and combined modes require:
- `data/processed/db5_9class.npz`
- `artifacts/cnn_gru_db5.pt`

These are user-generated dataset/model artifacts and are not included in the source-only branch if they were excluded from Git. Check the files exist before the showcase. The README cannot recreate the exact trained checkpoint; copy your known-good local artifacts into those paths.

**Camera scope:** this webcam mode recognises hand poses from a single hand's 21 MediaPipe landmarks. It does not track the user's shoulder/elbow or reproduce full-body arm kinematics; the 3D arm maps recognised hand poses to prosthetic finger/wrist joint targets. For full anatomical arm tracking, an upper-body pose/hand-position controller would be a separate feature.\n\n## Outputs

- `logs/semg_predictions.csv` — sEMG-only window predictions
- `logs/semg_simulation_replay.csv` — stable sEMG movement transitions
- `logs/simulation_*.csv` — camera/scripted movement labels
- `generated/prosthetic_arm.urdf` — generated robot model

## Report results responsibly

Existing DB5 evaluation: 82.03% raw accuracy, 59.78% balanced accuracy, 59.33% macro F1, and 86.08% accepted-prediction accuracy at 91% coverage. REST is dominant; report balanced accuracy and macro F1 alongside raw accuracy. Simulation replay events are not a new independent evaluation set.

This is a research prototype, not a certified medical device. Webcam gestures are a demonstration controller, not a clinical measurement. Do not connect unvalidated outputs to powered hardware.
