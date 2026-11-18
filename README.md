# Prosthetic Arm — sEMG + Camera + 3D Simulation

**Recommended target environment: Windows 10/11, CPython 3.11 x64.**

Run `python main.py` from the repository root to open the three-mode launcher.

## What the three modes do

1. **sEMG signal output** — runs the CNN–GRU against held-out NinaPro DB5 windows and prints the predicted gesture, averaged confidence, acceptance and inference latency. It does not open PyBullet.
2. **Camera simulation** — MediaPipe tracks one hand's 21 landmarks and maps stable hand poses to the PyBullet prosthetic hand/wrist. Hold a pose for about 0.55 seconds. Accepted movements have a minimum 1.5-second cooldown. REST returns the simulated joints to neutral. Press Q or Esc in the camera window to exit.
3. **Combined sEMG + simulation** — classifies held-out test windows with five-window probability smoothing and a 0.80 confidence threshold. Each accepted pose is labelled in the terminal, held for 1.0 second, then returned to neutral for at least 1.5 seconds before another movement is accepted. A CSV event log is written.

**Important scope distinction:** the combined showcase replays the recorded held-out sEMG dataset. It is not live electrode acquisition. The CLI also supports CSV/serial live streaming when a compatible 16-channel, 200 Hz stream is supplied. The webcam mode classifies hand poses; it does not estimate full shoulder/elbow kinematics or reproduce the user's complete arm trajectory.

## One-time Windows setup

Use a clean Python 3.11 x64 environment. Do not reuse a virtual environment created by another Python version.

```powershell
# From the repository root:
deactivate  # only when an old venv is active; ignore if command is not found
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# Install NumPy from a prebuilt wheel:
python -m pip install --only-binary=:all: numpy==1.26.4

# Install the remaining pinned packages:
python -m pip install -r requirements.txt

# Install this project and run its test suite:
python -m pip install -e .
python -m pytest -q
```

If the NumPy wheel command reports no matching distribution, check `python --version` and `py -0p`: the active interpreter must be CPython 3.11, 64-bit. Do not try to compile NumPy with an old MinGW/TDM GCC compiler.

If PyBullet installation itself fails, preserve the complete pip error. On Windows, install Microsoft C++ Build Tools (MSVC v143) and the Windows SDK, then retry `python -m pip install pybullet==3.2.7`. Do not silently replace the physics engine.

## Launch and verify

Start the mode selector:

```powershell
python main.py
```

Or use the CLI:

```powershell
# Verify that the generated 3D arm can execute a scripted gesture sequence:
python main.py demo

# Camera-tracked hand-pose simulation (Q or Esc exits):
python main.py camera

# First verify the sEMG classifier without opening a 3D window:
python main.py semg-replay --limit 100 --headless

# Then watch the 3D arm replay the held-out sEMG movements:
python main.py semg-replay --limit 100 --hold 1.0

# Full held-out sEMG replay:
python main.py semg-replay --limit 1000 --hold 1.0

# Live CSV or serial stream (requires a real compatible sensor stream):
python main.py live --csv recordings/user01.csv
python main.py live --serial COM5 --baud 115200
```

The expected DB5 input is 16 channels sampled at 200 Hz. CSV input must contain one sample per row; the last 16 numeric columns are used. Serial input must send one comma-separated 16-value sample per line at the configured baud rate and actual 200 Hz sample cadence. A different electrode layout, channel order or sample rate is not compatible merely because it can be read; it requires an explicit adaptation and validation.

## Required trained assets

The current showcase branch includes both required artifacts in Git:

- `data/processed/db5_9class.npz` — processed NinaPro DB5 windows
- `artifacts/cnn_gru_db5.pt` — trained CNN–GRU checkpoint

You do not need to download raw DB5 recordings or retrain the model to run the three showcase modes. Raw recordings under `data/raw/db5/` are only required when rebuilding the processed dataset or running the raw-stream comparison script.

The checkpoint stores its channel normalizer and model metadata. The inference code validates the checkpoint's class order, channel dimensions, window dimensions and normalizer values before accepting input.

## Logs and generated files

- `logs/semg_predictions.csv` — sEMG-only window-level predictions
- `logs/semg_simulation_replay.csv` — accepted sEMG movement events, subject/repetition, confidence and hold timing
- `logs/simulation_*.csv` — camera or scripted simulation commands
- `generated/prosthetic_arm.urdf` — generated PyBullet robot model

## Existing DB5 evaluation

The saved DB5 evaluation reports 82.03% raw accuracy, 59.78% balanced accuracy, 59.33% macro F1, and 86.08% accepted-prediction accuracy at 91% coverage. REST is the dominant class, so raw accuracy alone is not an adequate summary. Re-run the held-out evaluation before quoting these figures as a result from a new model run. Simulation replay events are not an independent evaluation set.

This is a research prototype, not a certified medical device. Webcam gestures are a demonstration controller, not a clinical measurement. Do not connect unvalidated outputs to powered hardware.
