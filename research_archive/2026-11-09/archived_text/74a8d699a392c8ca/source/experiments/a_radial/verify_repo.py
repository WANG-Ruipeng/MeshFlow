from pathlib import Path
import subprocess,json,hashlib,argparse
from .io_utils import write_json
LOCK=Path(__file__).resolve().parent/'assets/UPSTREAM_LOCK.json'
def verify(repo):
    repo=Path(repo).resolve();lock=json.loads(LOCK.read_text())
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    if git('rev-parse',lock['commit']+'^{tree}')!=lock['tree']:raise RuntimeError('Wrong pinned tree')
    git('merge-base','--is-ancestor',lock['commit'],'HEAD')
    listing=git('ls-tree','-r',lock['commit']);expected={}
    for line in listing.splitlines():
        meta,path=line.split('\t',1);mode,typ,sha=meta.split()
        if typ=='blob':expected[path]=sha
    protected=[p for p in expected if p.startswith(('models/','datasets/')) or p in ['utils/ot_utils.py','flow_matching.py','train.py','inference.py','gradio_demo.py']]
    changes=[]
    for p in protected:
        path=repo/p
        if not path.is_file():changes.append(p+':missing');continue
        raw=path.read_bytes();actual=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        if actual!=expected[p]:changes.append(p)
    for p,sha in lock['observed_blobs'].items():
        if expected.get(p)!=sha:raise RuntimeError('Lock/tree inconsistency:'+p)
    result={'repository':lock['repository'],'base_commit':lock['commit'],'current_HEAD':git('rev-parse','HEAD'),'tree':lock['tree'],'protected_files_checked':len(protected),'protected_changes':changes,'pass':not changes,'status_porcelain':git('status','--porcelain')}
    if changes:raise RuntimeError(result)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--report',required=True);a=p.parse_args();r=verify(a.repo);write_json(a.report,r);print(r)
