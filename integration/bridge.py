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


def replay_processed(data_path, model_path, gui=True, limit=1000, threshold=0.70, command_hold=0.05, log_path='logs/semg_simulation_replay.csv'):
    z=np.load(data_path); X=z['X'].astype(np.float32); y=z['y'].astype(np.int64); reps=z['repetitions']
    mask=repetition_masks(reps)['test']; X=X[mask]; y=y[mask]
    clf=WindowClassifier(model_path,threshold=threshold); mapper=GestureMapper()
    Path(log_path).parent.mkdir(parents=True,exist_ok=True)
    correct=0; n=min(int(limit),len(X))
    with PyBulletProstheticSimulator(gui=gui) as sim, open(log_path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['index','true_gesture','predicted_gesture','confidence','accepted','latency_ms'])
        for i in range(n):
            d=clf.classify_filtered_window(X[i]); true=CLASS_NAMES[int(y[i])]
            correct += int(d.gesture==true)
            cmd=mapper.apply(d.gesture,d.confidence); sim.apply(cmd)
            sim.step(max(1,int(240*command_hold)),realtime=gui)
            w.writerow([i,true,d.gesture,d.confidence,d.accepted,d.latency_ms]); f.flush()
            if i<50 or i%100==0: print(f'{i:05d} true={true:20s} pred={d.gesture:20s} conf={d.confidence*100:5.1f}%')
    print(f'Integrated replay accuracy over displayed smoothed commands: {100*correct/n:.2f}% ({n} windows)')
    print('Log:',log_path)


def live_stream(model_path, gui=True, csv_path=None, serial_port=None, baud=115200, threshold=0.70, log_path='logs/live_semg_simulation.csv'):
    clf=StreamingClassifier(model_path,threshold=threshold); mapper=GestureMapper()
    channels=int(clf.window_model.ckpt['channels'])
    if csv_path: stream=read_csv_stream(csv_path,channels)
    elif serial_port: stream=read_serial_stream(serial_port,baud,channels)
    else: raise ValueError('Provide csv_path or serial_port')
    Path(log_path).parent.mkdir(parents=True,exist_ok=True)
    with PyBulletProstheticSimulator(gui=gui) as sim, open(log_path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['timestamp','gesture','confidence','accepted','latency_ms'])
        t0=time.perf_counter()
        for sample in stream:
            d=clf.push_sample(sample)
            if d is None: continue
            cmd=mapper.apply(d.gesture,d.confidence); sim.apply(cmd); sim.step(max(1,clf.stride_samples),realtime=False)
            w.writerow([time.perf_counter()-t0,d.gesture,d.confidence,d.accepted,d.latency_ms]); f.flush()
            print(f'{d.gesture:20s} conf={d.confidence*100:5.1f}% accepted={d.accepted} latency={d.latency_ms:.2f} ms')
