"""Resumable CPU-only entry. Human/AI visual judgments are never synthesized.

Existing stage outputs are verified and reused. Missing visual reviews stop only
their dependent publication stage; neither training nor generation is available.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse, json, subprocess, sys, time
from pathlib import Path
from .publish import OUT, ROOT, read, resolve, sha


def invoke(module, *args):
    subprocess.run([sys.executable,'-B','-m','r2.data_tools.bench_parts_v1.'+module,*map(str,args)],cwd=ROOT,check=True)


def reviews_ready(bundlefile):
    if not bundlefile.exists():return [str(bundlefile)]
    return [p for b in read(bundlefile)['bundles'] for p in b['reviews'] if not resolve(p).exists()]


def verify_gallery(path):
    data=read(path);assert sha(resolve(data['manifest_path']))==data['manifest_sha256']
    for p in data['pages']:assert sha(resolve(p['path']))==p['sha256']


def verify_quality_implementation(path):
    data=read(path)
    assert data['implementation_sha256']==sha(Path(__file__).with_name('quality.py')), 'Quality implementation changed: use a separate audit version, do not silently reuse'
    return data


def verify_final():
    p=OUT/'data/manifest.json'
    if not p.exists():return dict(status='INCOMPLETE',reason='Reviewed data not yet published')
    m=read(p)
    for path,digest in m['references'].items():assert sha(resolve(path))==digest
    for r in m['rows']:
        for k in ('source_case','paired','input','input_obj','quality','supervision'):
            pathkey=k+'_path' if k not in ('quality','supervision') else k+'_file'
            assert sha(resolve(r[pathkey]))==r[k+'_sha256'],(r['condition_id'],k)
        assert sha(resolve(r['review']['image_path']))==r['review']['image_sha256']
    for p,h in read(OUT/'data/verification.json')['source_hashes_unchanged'].items():assert sha(resolve(p))==h
    for b in read(OUT/'FINAL_BUNDLES.json')['bundles']:verify_quality_implementation(resolve(b['quality']))
    gallery=OUT/'gallery/final/manifest.json'
    if not gallery.exists():return dict(status='INCOMPLETE',reason='Final gallery not yet published',active_conditions=len(m['rows']),model_calls=0)
    verify_gallery(gallery)
    assert {r['condition_id'] for r in read(gallery)['pages']}=={r['condition_id'] for r in m['rows']}
    return dict(status='PASS',active_conditions=len(m['rows']),final_gallery=True,model_calls=0,training_updates=0)


def resume():
    base=OUT/'candidates/manifest.json'
    if not base.exists():invoke('construct')
    q=OUT/'quality/summary.json'
    if not q.exists():invoke('quality','--manifest',base,'--out',q.parent)
    verify_quality_implementation(q)
    g=OUT/'gallery/candidates/manifest.json'
    if not g.exists():invoke('render','--manifest',base,'--mode','candidates')
    else:verify_gallery(g)
    supplement=OUT/'candidates/supplement_3461/manifest.json'
    if not supplement.exists():invoke('construct','--supplement-3461')
    sq=OUT/'quality/supplement_3461/summary.json'
    if not sq.exists():invoke('quality','--manifest',supplement,'--out',sq.parent)
    verify_quality_implementation(sq)
    sg=OUT/'gallery/supplement_3461/manifest.json'
    if not sg.exists():invoke('render','--manifest',supplement,'--mode','candidates','--output-name','supplement_3461')
    else:verify_gallery(sg)
    missing=reviews_ready(OUT/'BASE_BUNDLES.json')
    if missing:return dict(status='BLOCKED_ACTUAL_VISUAL_REVIEW',missing=missing)
    invoke('publish','combine','--bundles',OUT/'BASE_BUNDLES.json')
    cm=OUT/'combinations/manifest.json';cq=OUT/'quality/combinations/summary.json'
    if not cq.exists():invoke('quality','--manifest',cm,'--out',cq.parent)
    verify_quality_implementation(cq)
    cg=OUT/'gallery/combinations/manifest.json'
    if not cg.exists():invoke('render','--manifest',cm,'--mode','candidates','--output-name','combinations')
    else:verify_gallery(cg)
    missing=reviews_ready(OUT/'FINAL_BUNDLES.json')
    if missing:return dict(status='BLOCKED_ACTUAL_COMBINATION_REVIEW',missing=missing)
    invoke('publish','finalize','--bundles',OUT/'FINAL_BUNDLES.json')
    fg=OUT/'gallery/final/manifest.json'
    if not fg.exists():invoke('render','--manifest',OUT/'data/manifest.json','--mode','final')
    else:verify_gallery(fg)
    return verify_final()


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');args=p.parse_args()
    started=time.time();result=resume() if args.resume else verify_final()
    result['wall_seconds']=time.time()-started
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['status'] not in ('PASS',):sys.exit(2)

if __name__=='__main__':main()
