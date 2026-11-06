import argparse
from prosthetic_semg.data import prepare_db5
from prosthetic_semg.config import SignalConfig

p = argparse.ArgumentParser()
p.add_argument("--raw", default="data/raw/db5")
p.add_argument("--out", default="data/processed/db5_9class.npz")
p.add_argument("--subjects", nargs="*", type=int)
p.add_argument("--mains", type=float, default=50.0, choices=[50.0, 60.0])
a = p.parse_args()

prepare_db5(a.raw, a.out, SignalConfig(notch_hz=a.mains), a.subjects)
