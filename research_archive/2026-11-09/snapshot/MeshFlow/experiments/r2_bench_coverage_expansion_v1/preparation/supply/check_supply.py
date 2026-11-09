"""Read-only Bench split/supply preparation. No extraction, model or mesh changes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,csv,hashlib,io,json,resource,subprocess,tarfile,time
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/e/MeshFlow')
OUT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'downloaded_data/archives/shapenet-cls.tar.gz'
INVENTORY=ROOT/'experiments/r2_bench_data_chain_audit_20261021/geometry/archive_supply/per_parent.csv'
OLD=ROOT/'experiments/r2_bench_control_transfer_v1/data/manifest.json'
CUR=ROOT/'experiments/r2_bench_clean_multipart_seam_v1/data/manifest.json'
EXPECTED='06707caff4ce465f402e6b69f81056204c579921d371eb2c9e4e9a3929f7aef2'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()

def write(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def snapshot():
    mem={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    args=['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']
    r=subprocess.run(args,capture_output=True,text=True)
    return dict(unix=time.time(),cpu_logical=os.cpu_count(),load_average=list(os.getloadavg()),MemAvailable=mem.get('MemAvailable'),MemTotal=mem.get('MemTotal'),numeric_threads=1,workers=1,GPU_compute_calls=0,nvidia_smi=dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip()))

def main():
    start=time.time();write('resources_before.json',snapshot())
    hashes={str(p):sha(p) for p in (ARCHIVE,INVENTORY,OLD,CUR)}
    if hashes[str(ARCHIVE)]!=EXPECTED:raise ValueError('Archive differs from original protected source')
    original=json.loads(OLD.read_text(encoding='utf-8-sig'));current=json.loads(CUR.read_text(encoding='utf-8-sig'))
    old_train={r['parent_id'] for r in original['train']};old_dev={r['parent_id'] for r in original['dev']};current_train={r['parent_id'] for r in current['train']}
    split_map={};split_sizes={};split_hashes={}
    with tarfile.open(ARCHIVE,'r:gz') as tf:
        for split in ('train','test'):
            name=f'shapenet-02828884/split/{split}.npz';payload=tf.extractfile(name).read();split_hashes[name]=hashlib.sha256(payload).hexdigest()
            # Original official split is an object-record NPZ; this is the same
            # already-registered trusted local metadata reader as historical prep.
            with np.load(io.BytesIO(payload),allow_pickle=True) as z:
                if 'npz_list' not in z.files:raise ValueError('Unexpected official split schema')
                records=z['npz_list'];split_sizes[split]=len(records)
                for rec in records:
                    uid=str(rec['uid'])
                    if uid in split_map and split_map[uid]!=split:raise ValueError('Official train/test UID collision')
                    split_map[uid]=split
    rows=[]
    with INVENTORY.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            uid=r['parent_id'];n=int(r['N']);official=split_map.get(uid,'UNKNOWN')
            checkpass=r['passes_two_exact_checks']=='True'
            if uid in old_train and official!='train':raise ValueError('Old train outside officialtrain')
            if uid in old_dev and r['existing64_8_membership']!='DEV':raise ValueError('Old DEV inventory mismatch')
            eligible=official=='train' and uid not in old_dev
            r.update(official_split=official,legacy_TRAIN64=uid in old_train,legacy_DEV8=uid in old_dev,current_TRAIN58=uid in current_train,
                     train_supply_allowed=eligible,train_supply_reason='OFFICIAL_TRAIN_NOT_OLD_DEV' if eligible else ('PROTECTED_OLD_DEV' if uid in old_dev else 'OFFICIAL_TEST_OR_UNKNOWN'),
                     option_N128_256=eligible and 128<=n<=256,option_N128_400=eligible and 128<=n<=400,
                     new_relative_oldTRAIN64=eligible and uid not in old_train,new_relative_currentTRAIN58=eligible and uid not in current_train)
            rows.append(r)
    if len(rows)!=560:raise ValueError('Expected560 archive member records')
    def summarize(rs):
        return dict(parents=len(rs),official_split_counts=dict(collections.Counter(r['official_split'] for r in rs)),
                    oldTRAIN64=sum(r['legacy_TRAIN64'] for r in rs),oldDEV8=sum(r['legacy_DEV8'] for r in rs),currentTRAIN58=sum(r['current_TRAIN58'] for r in rs),
                    train_supply_allowed=sum(r['train_supply_allowed'] for r in rs),
                    allowed_passing_two_checks=sum(r['train_supply_allowed'] and r['passes_two_exact_checks']=='True' for r in rs),
                    new_allowed_relative_oldTRAIN64=sum(r['new_relative_oldTRAIN64'] for r in rs),new_allowed_relative_currentTRAIN58=sum(r['new_relative_currentTRAIN58'] for r in rs),
                    new_allowed_and_two_checks_relative_oldTRAIN64=sum(r['new_relative_oldTRAIN64'] and r['passes_two_exact_checks']=='True' for r in rs),
                    new_allowed_and_two_checks_relative_currentTRAIN58=sum(r['new_relative_currentTRAIN58'] and r['passes_two_exact_checks']=='True' for r in rs),
                    N_min=min([int(r['N']) for r in rs],default=None),N_max=max([int(r['N']) for r in rs],default=None))
    options={'N128_256':summarize([r for r in rows if 128<=int(r['N'])<=256]),'N128_400':summarize([r for r in rows if 128<=int(r['N'])<=400]),'N257_400_increment':summarize([r for r in rows if 257<=int(r['N'])<=400]),'unselected_N128_256':summarize([r for r in rows if 128<=int(r['N'])<=256 and not r['legacy_TRAIN64'] and not r['legacy_DEV8']])}
    with (OUT/'all_Bench_supply.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/'candidate_supply_N128_400.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(r for r in rows if r['option_N128_400'])
    source_paths=['r2/pilots/bench_control_transfer_v1/data_prepare.py','r2/data_tools/bench_parts_v1/construct.py','r2/pilots/bench_parts_convergence_v1/data.py','r2/src/meshflow_control/seam/slot_data.py','r2/src/meshflow_control/seam/model_adapter.py','r2/src/meshflow_control/models/native.py','r2/src/meshflow_control/config.py']
    hashes.update({str(ROOT/p):sha(ROOT/p) for p in source_paths})
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Protected source changed')
    write('input_hashes.json',hashes)
    write('summary.json',dict(status='COMPLETE',options=options,official_split_records=split_sizes,official_split_member_sha256=split_hashes,unknown_archive_UIDs=[r['parent_id'] for r in rows if r['official_split']=='UNKNOWN'],
        legacy_DEV_split_counts=dict(collections.Counter(r['official_split'] for r in rows if r['legacy_DEV8'])),
        eligibility_boundary='Supply candidates only; no automatic semantic/region/interface eligibility, no fullFF or near-duplicate certification; oldDEV8 alwaysexcluded even officialtrain',
        passing_two_checks='No equal unordered source-ID duplicate face and no exact zero-area face in stored original coordinates; not a clean or intersection-free certification',
        implementation=dict(current_N128_256='Existing production/pilot adapters and data gate enforce128..256; no range change needed',N128_400='Not currently supported end-to-end:requires independently versioned gates in native.py:228,seam/model_adapter.py:73,seam/slot_data.py:46,and bench_parts_convergence_v1/data.py:132; source archive loaderdata_prepare.py:160 also filters256; feature K/N divisors256 must be explicitly retained or changed, not silent',backbone='Config max_length800 supports count embedding capacity in principle; not an executed end-to-end400 guarantee',part_constructor='construct.py reads fixed old72sourcecase manifest and preserves originalfaceIDs; no explicit256 gate in basic_eligible, but needs new source snapshot/manifest and C review'),
        scientific_variables=dict(N128_256=['independentparentcoverage','partconditioncoverage','parentexposureperupdate ifbudgetfixed'],N128_400=['aboveplusNdistribution','permeshfreefacebudget','conditionNembedding/feature distribution','OT/attention computational cost','higherresolutiongeometry maychangequality independentlyofparentcount']),
        CPU_workers=1,numeric_threads=1,GPU_calls=0,model_calls=0,training_updates=0,geometry_construction=0,full_FF='NOT_RUN',all_protected_hashes_unchanged=True,wall_seconds=time.time()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,script_sha256=sha(__file__)))
    write('resources_after.json',snapshot())
    print(json.dumps(options,indent=2))

if __name__=='__main__':main()
