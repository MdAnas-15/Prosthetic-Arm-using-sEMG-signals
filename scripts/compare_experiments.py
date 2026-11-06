import argparse, json
from pathlib import Path

p = argparse.ArgumentParser(description='Compare saved sEMG experiment metric JSON files.')
p.add_argument('metrics', nargs='+', help='Paths to metrics.json files')
p.add_argument('--names', nargs='*', help='Optional experiment names in the same order')
p.add_argument('--csv', default='reports/dataset_comparison.csv')
a = p.parse_args()
rows=[]
for i, path in enumerate(a.metrics):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    name=(a.names[i] if a.names and i < len(a.names) else Path(path).parent.name)
    rows.append((name,d))
keys=['accuracy','balanced_accuracy','macro_f1','coverage_at_threshold','accepted_accuracy','mean_confidence']
print(f"{'Experiment':24s}" + ''.join(f"{k:>22s}" for k in keys))
for name,d in rows:
    print(f"{name:24s}" + ''.join(f"{100*d.get(k, float('nan')):21.2f}%" for k in keys))
out=Path(a.csv); out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf-8') as f:
    f.write('experiment,'+','.join(keys)+'\n')
    for name,d in rows:
        f.write(name+','+','.join(str(d.get(k,'')) for k in keys)+'\n')
print('Saved',out)
