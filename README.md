# Prosthetic Arm — sEMG + Camera + 3D Simulation

Run `python main.py` to open the three-mode showcase launcher.

## Showcase modes

1. **sEMG signal output** — runs the trained CNN–GRU classifier on the held-out DB5 test repetitions and displays predicted gesture, confidence, acceptance and latency. It does not open PyBullet.
2. **Camera-based simulation** — uses the webcam and MediaPipe Hands to recognize discrete hand poses and drive the PyBullet arm. Hold a pose briefly to register it; press Q in the camera window to exit.
3. **Combined sEMG + simulation** — classifies held-out sEMG windows, stabilizes predictions into movement transitions, applies each transition to the arm, holds each pose so it is visible, and writes a labelled CSV event log.

The sEMG-only and combined modes require `data/processed/db5_9class.npz` and `artifacts/cnn_gru_db5.pt`. The trained artifacts are already tracked in this repository.

## Windows setup (Python 3.11 recommended)

**Important PyBullet installation note:** the official `pybullet` package on PyPI does not publish a Windows CPython wheel. On Windows, pip therefore builds it from source. The Microsoft Visual C++ 14.x error is a missing build-toolchain issue, not an error in the arm's Python code or URDF. Before installing requirements, install [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select:
- Desktop development with C++
- MSVC v143 x64/x86 build tools
- Windows 10 or Windows 11 SDK

Restart PowerShell after installation, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python main.py
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe` directly for every command. Install dependencies with that same interpreter:
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Camera mode requires a working webcam and Windows camera permission. If the default camera is unavailable, run `python main.py camera --camera 1`. If MediaPipe cannot be installed in an existing environment, recreate the venv with Python 3.11 rather than mixing Python installations.

## Command-line modes

```powershell
python main.py demo
python main.py camera
python main.py semg-replay --limit 1000
python main.py live --csv recordings/user01.csv
python main.py live --serial COM5 --baud 115200
```

## Model and evaluation

The classifier is a 1D CNN–GRU trained on NinaPro DB5 windows. Repetitions are split into train/validation/test groups; the replay uses held-out test repetitions. A movement-event log is not a new independent test set, and window-level accuracy must not be described as movement accuracy. Existing DB5 evaluation: 82.03% raw accuracy, 59.78% balanced accuracy, 59.33% macro F1, and 86.08% accepted-prediction accuracy at 91% coverage. REST is dominant, so report balanced accuracy and macro F1 alongside raw accuracy.

## Outputs

- `logs/semg_predictions.csv` — sEMG-only predictions
- `logs/semg_simulation_replay.csv` — one row per stable movement transition
- `logs/simulation_*.csv` — camera/scripted simulation movement labels
- `generated/prosthetic_arm.urdf` — generated robot model

This is a research prototype, not a certified medical device. Webcam gestures are a demonstration controller, not a clinical measurement. Do not connect unvalidated outputs to powered hardware.
