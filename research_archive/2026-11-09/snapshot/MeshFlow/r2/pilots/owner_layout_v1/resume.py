"""Read-only state by default; conditional generation refuses failed head gate."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_owner_layout_v1'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['status','evaluate-heads'],default='status');a=p.parse_args()
    stages={}
    for name,path in [('MLP',OUT/'heads/MLP/complete.json'),('CONTEXT',OUT/'heads/CONTEXT/complete.json'),
                      ('head_gate',OUT/'gate.json'),('final',OUT/'completion.json')]:
        stages[name]=read(path).get('status','RECORDED') if path.exists() else 'NOT_COMPLETE'
    if a.stage=='status':print(json.dumps(dict(stages=stages,new_model_calls=0),indent=2));return
    if (OUT/'completion.json').exists():raise SystemExit('Fixed task complete; no automatic continuation')
    if not all(stages[k]=='COMPLETE' for k in ['MLP','CONTEXT']):raise SystemExit('Both fixed2000 heads must finish')
    if (OUT/'evaluation_head/started.json').exists():raise SystemExit('Terminal evaluation already attempted; inspect rather than retry')
    raise SystemExit(subprocess.call([sys.executable,'-B','-m','r2.pilots.owner_layout_v1.head','evaluate'],cwd=ROOT))

if __name__=='__main__':main()
