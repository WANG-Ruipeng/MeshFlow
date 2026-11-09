"""Resumable bounded CPU entry; sealed runs only verify their receipts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_backend_counterfactual_v1'
CODE=Path(__file__).parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(entries):
    for p,h in entries.items():
        if sha(p)!=h:raise ValueError('Hash differs: '+p)
def module(name,*args):
    subprocess.run([sys.executable,'-B','-m','r2.pilots.bench_backend_counterfactual_v1.'+name,*args],cwd=ROOT,check=True)
def seal():
    m=read(OUT/'manifest.json');verify(m['protected']);verify(m['source_hashes'])
    core=read(OUT/'core_complete.json');score=read(OUT/'scores/COMPLETE.json');gallery=read(OUT/'gallery/COMPLETE.json')
    assert core['GEO_calls']==10 and core['model_calls']==0 and score['outputs']==10 and gallery['pages']==10
    review=read(OUT/'gallery/visual_review.json');audit=read(OUT/'audits/reference_review.json')
    if audit.get('status')!='PASS':raise ValueError('Independent review not passed')
    assert (OUT/'support/per_output_stage.csv').exists() and (OUT/'REPORT.md').exists()
    archive=OUT/'source_archive';archive.mkdir(exist_ok=False)
    entries={}
    for p in sorted(CODE.glob('*.py')):
        dest=archive/p.name;dest.write_bytes(p.read_bytes());entries[str(p)]=sha(p)
    diffs=[]
    for p in sorted(CODE.glob('*.py')):
        rel=p.relative_to(ROOT).as_posix();lines=p.read_text('utf-8').splitlines(True)
        diffs.append('diff --git a/'+rel+' b/'+rel+'\nnew file mode 100644\n--- /dev/null\n+++ b/'+rel+'\n@@ -0,0 +1,'+str(len(lines))+' @@\n'+''.join('+'+s for s in lines))
    (archive/'code.diff').write_text('\n'.join(diffs),encoding='utf-8')
    (archive/'receipt.json').write_text(json.dumps({'scope':'Actual new experiment CPU code; original production dependencies indexed by manifest hashes.','files':entries},indent=2),encoding='utf-8')
    result={'status':'COMPLETE','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'RAW_GEO':8,'SOURCE_GEO':2,'GEO_calls':10,'new_scores':10,'old_scores_reused':16,'gallery_pages':10,
        'model_calls':0,'D0_calls':0,'training_updates':0,'OT_calls':0,'new_noise':0,
        'original_protected_files':len(m['protected']),'inputs':m['protected'],'source_files':entries,
        'limits':'Two training parents only; no default change, commit/push, training, or extra evaluation.',
        'files':{str(p):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='COMPLETE.json'}}
    # Include nested COMPLETE receipts too; exclude only this future root file.
    for p in OUT.rglob('COMPLETE.json'):
        if p!=OUT/'COMPLETE.json':result['files'][str(p)]=sha(p)
    (OUT/'COMPLETE.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','files':len(result['files']),'new_model_calls':0}))
def run(seal_only=False):
    done=OUT/'COMPLETE.json'
    if done.exists():
        r=read(done);verify(r['inputs']);verify(r['files']);verify(r['source_files'])
        print(json.dumps({'status':'VERIFIED_COMPLETE','files':len(r['files']),'new_model_calls':0,'new_GEO_calls':0}));return
    if seal_only:seal();return
    if not list((OUT/'resources').glob('snapshot_*.json')):module('resource_check')
    module('support')
    module('core','run')
    if not (OUT/'support/per_output_stage.csv').exists():module('support','--analyse')
    module('evaluate','--manifest',str(OUT/'manifest.json'),'--workers','2')
    module('render','--manifest',str(OUT/'manifest.json'))
    module('summarize')
    print(json.dumps({'status':'AWAIT_ACTUAL_VISUAL_REVIEW_AND_REPORT','new_model_calls':0}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');a=p.parse_args();run(a.seal)
