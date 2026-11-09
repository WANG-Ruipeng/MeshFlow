"""Evaluation identities and read-only routing for prior24; no scorer imports."""
from .common import ROOT, OUT, SOURCE_ROOT, read, resolve, sha
from .panel import LABELS, NEW_LABELS, requests, labels_for, arrays, verify, P
from r2.pilots.bench_full_seat_curriculum_v1.eval_runtime import source_assets
BASELINE_LABEL=LABELS[0]
ALL_LABELS=FINAL_LABELS=LABELS
HEAD_ROOT=ROOT/'experiments/r2_bench_three_family_v1/production_contract'

def label_spec(label):
    if label not in LABELS:raise ValueError('Unknown label')
    reg=read(OUT/'registration.json');s=dict(reg['evaluation_labels'][label]);s.setdefault('subset','full')
    updates=reg['updates']
    wanted={LABELS[0]:('NATIVE',6000,18000),LABELS[1]:('NATIVE',updates,18000+updates),LABELS[2]:('SEAM',3000,21000)}[label]
    if (s['arm'],s['local_step'],s['global_step'])!=wanted or s['subset']!='full':raise ValueError('Unexpected endpoint')
    expected=(OUT/'training/NATIVE'/('generator_step'+str(updates)+'.pt')) if label==LABELS[1] else (SOURCE_ROOT/'training'/s['arm']/('generator_step'+str(s['local_step'])+'.pt'))
    if resolve(s['generator_path'])!=expected:raise ValueError('Unexpected generator export')
    s['mode']='SEAM' if label==LABELS[2] else 'PLAIN'
    s['schema']='bench_seat_coverage_generator_v1' if label==LABELS[1] else 'bench_full_seat_curriculum_generator_v1'
    return s

def reuse_spec(q,label):
    return dict(root=str(SOURCE_ROOT),label=label) if q['panel_origin']=='INHERITED24' and label in (LABELS[0],LABELS[2]) else None

def root_for(label,q):
    if label not in labels_for(q):raise ValueError('SEAM has no new-panel samples')
    return SOURCE_ROOT if reuse_spec(q,label) else OUT

def new_requests(label):return [q for q in requests(label) if not reuse_spec(q,label)]
def copy_reference(*args):raise RuntimeError('References must stay read-only; route to original root')
def raw_completion(label):return OUT/'evaluation'/label/'raw_complete_full.json'

def raw_receipt(q,label):
    return root_for(label,q)/'evaluation'/label/q['request_id']/'raw/receipt.json'

def verify_one_raw(q,label):
    path=raw_receipt(q,label);r=read(path)
    if (r['status']!='COMPLETE' or r['request_id']!=q['request_id'] or r['label']!=label
        or r['generator_forwards']!=50 or r['D0_forwards']!=0
        or r['raw_sha256']!=sha(path.parent/'raw.npz') or r['noise_sha256']!=q['noise_sha256']):raise ValueError('RAW identity differs')
    verify(r['files']);return path

def verify_raw(label):
    qs=requests(label)
    if label!=LABELS[2]:
        done=read(raw_completion(label))
        if done['status']!='COMPLETE' or done['outputs']!=len(qs):raise ValueError('Incomplete new RAW wave')
        verify(done['files'])
    return qs,[verify_one_raw(q,label) for q in qs]

def all_raw_barrier():
    receipts=[]
    for label in LABELS:
        _,rr=verify_raw(label);receipts.extend(rr)
    expected=sum(read(OUT/'eval_manifest.json')['requests_per_label'].values())
    if len(receipts)!=expected:raise ValueError('All new paired RAW must complete before any newD0')
    return receipts
