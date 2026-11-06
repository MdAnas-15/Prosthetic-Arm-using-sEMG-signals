import argparse
from prosthetic_sim.demo import run_demo
from integration.bridge import replay_processed, live_stream

p=argparse.ArgumentParser(description='Combined sEMG + prosthetic simulation')
sub=p.add_subparsers(dest='mode',required=True)
d=sub.add_parser('demo'); d.add_argument('--headless',action='store_true'); d.add_argument('--hold',type=float,default=0.8)
r=sub.add_parser('semg-replay'); r.add_argument('--data',default='data/processed/db5_9class.npz'); r.add_argument('--model',default='artifacts/cnn_gru_db5.pt'); r.add_argument('--limit',type=int,default=1000); r.add_argument('--threshold',type=float,default=0.70); r.add_argument('--headless',action='store_true'); r.add_argument('--hold',type=float,default=0.05)
l=sub.add_parser('live'); l.add_argument('--model',default='artifacts/cnn_gru_db5.pt'); l.add_argument('--csv'); l.add_argument('--serial'); l.add_argument('--baud',type=int,default=115200); l.add_argument('--threshold',type=float,default=0.70); l.add_argument('--headless',action='store_true')
a=p.parse_args()
if a.mode=='demo': run_demo(gui=not a.headless,hold_seconds=a.hold)
elif a.mode=='semg-replay': replay_processed(a.data,a.model,gui=not a.headless,limit=a.limit,threshold=a.threshold,command_hold=a.hold)
elif a.mode=='live': live_stream(a.model,gui=not a.headless,csv_path=a.csv,serial_port=a.serial,baud=a.baud,threshold=a.threshold)
