# Complete Prosthetic Arm - sEMG + 3D Simulation

This package combines the finished sEMG movement-recognition subsystem with a custom PyBullet prosthetic-arm/hand simulation through a common gesture-command bridge.

## Architecture

```text
NinaPro DB5 / live 16-channel sEMG
            |
      causal filtering
            |
       250 ms window
            |
        CNN + GRU
            |
 gesture + confidence
            |
 confidence rejection + smoothing
            |
       GestureMapper
            |
    ProstheticCommand
            |
      PyBullet arm
```

Recognized commands: REST, OPEN_HAND, CLOSE_HAND, WRIST_SUPINATION, WRIST_PRONATION, WRIST_FLEXION, WRIST_EXTENSION, RADIAL_DEVIATION and ULNAR_DEVIATION.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

## 1. Verify only the simulation

```powershell
python main.py demo
```

## 2. Train sEMG model if needed

The dataset and trained model are not bundled. Put NinaPro DB5 under `data/raw/db5/` and run:

```powershell
python scripts/run_semg_pipeline.py --epochs 45
```

If you already have your trained `cnn_gru_db5.pt`, copy it into `artifacts/`. If you already have `db5_9class.npz`, copy it into `data/processed/`.

## 3. Drive the simulation from held-out sEMG

```powershell
python main.py semg-replay --limit 1000
```

The model classifies held-out windows and the predicted gesture is immediately mapped to the 3D hand/wrist. A CSV is written to `logs/semg_simulation_replay.csv`.

## 4. Future live hardware

CSV stream:

```powershell
python main.py live --csv recordings/user01.csv
```

Serial stream:

```powershell
python main.py live --serial COM5 --baud 115200
```

The expected live interface follows the trained model metadata (DB5: 16 channels, 200 Hz). A different sensor layout/channel count requires retraining or adaptation and wearer-specific calibration.

## Existing DB5 result

`research_results/db5_run01/` contains the previously measured evaluation: 82.03% raw accuracy, 59.78% balanced accuracy, 59.33% macro F1, 86.08% accepted-prediction accuracy at 91% coverage. REST is the dominant class, so balanced accuracy and macro F1 are essential when interpreting performance.

## Git

```powershell
git init
git add .
git commit -m "feat: integrate sEMG recognition with prosthetic simulation"
git branch -M main
```

This project is a research prototype and is not certified medical-device or actuator-safety software.
