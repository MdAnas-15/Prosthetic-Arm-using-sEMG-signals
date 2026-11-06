import argparse, json
from prosthetic_semg.hardware import read_csv_stream, read_serial_stream
from prosthetic_semg.runtime import StreamingClassifier

p=argparse.ArgumentParser()
g=p.add_mutually_exclusive_group(required=True); g.add_argument("--csv"); g.add_argument("--serial")
p.add_argument("--baud",type=int,default=115200); p.add_argument("--model",default="artifacts/cnn_gru_db5.pt"); p.add_argument("--threshold",type=float,default=0.70)
a=p.parse_args()
rt=StreamingClassifier(a.model,a.threshold); channels=int(rt.window_model.ckpt["channels"])
stream=read_csv_stream(a.csv,channels) if a.csv else read_serial_stream(a.serial,a.baud,channels)
for sample in stream:
    d=rt.push_sample(sample)
    if d is not None: print(json.dumps(d.as_dict()), flush=True)
