from __future__ import annotations
import csv
from pathlib import Path
import numpy as np


def read_csv_stream(path: str | Path, channels: int):
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row: continue
            try: vals = [float(v) for v in row[-channels:]]
            except ValueError: continue
            if len(vals) == channels:
                yield np.asarray(vals, np.float32)


def read_serial_stream(port: str, baud: int, channels: int):
    import serial
    with serial.Serial(port, baudrate=baud, timeout=1) as ser:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line: continue
            try: vals = [float(v) for v in line.split(",")]
            except ValueError: continue
            if len(vals) == channels:
                yield np.asarray(vals, np.float32)
