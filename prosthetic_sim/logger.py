from pathlib import Path
import csv, time

class SimulationLogger:
    def __init__(self, path=None):
        if path is None:
            path=Path('logs')/f"simulation_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.f=self.path.open('w',newline='',encoding='utf-8')
        self.w=csv.writer(self.f)
        self.w.writerow(['timestamp','gesture','confidence','shoulder_yaw_deg','elbow_flex_deg','forearm_roll_deg','wrist_pitch_deg','wrist_deviation_deg','thumb','index','middle','ring','little'])
        self.t0=time.perf_counter()
    def write(self,c):
        self.w.writerow([time.perf_counter()-self.t0,c.gesture,c.confidence,c.shoulder_yaw_deg,c.elbow_flex_deg,c.forearm_roll_deg,c.wrist_pitch_deg,c.wrist_deviation_deg,c.thumb,c.index,c.middle,c.ring,c.little]); self.f.flush()
    def close(self): self.f.close()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
