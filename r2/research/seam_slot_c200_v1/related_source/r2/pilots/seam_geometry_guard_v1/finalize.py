"""Seal CPU experiment, preserve sources and create a small evidence package."""
from pathlib import Path
import csv,difflib,json,zipfile
from r2.pilots.seam_slot_c200_v1 import common as C
OUT=C.ROOT/'experiments/r2_seam_geometry_guard_v1'
CODE=C.ROOT/'r2/pilots/seam_geometry_guard_v1'

def verify(mapping):
    for p,h in mapping.items():assert C.sha(p)==h,p

def run():
    if (OUT/'completion.json').exists():
        from .resume import verify as verify_done
        return verify_done()
    occ=C.read(OUT/'occupancy/completion.json');normal=C.read(OUT/'normal/completion.json')
    attr=C.read(OUT/'attribution/completion.json');cap=C.read(OUT/'capacity/completion.json')
    capreg=C.read(OUT/'capacity/registration.json');pair=C.read(OUT/'capacity/connectivity_pairing_summary.json')
    independent=C.read(OUT/'capacity_independent_review.json');hist=C.read(OUT/'history_review.json')
    assert hist['attribution_independent_review']['status']=='PASS'
    assert C.read(OUT/'gallery/review_receipt.json')['paired_requests']==16
    assert cap['requests']==16 and C.read(OUT/'capacity/baseline_replay.json')['all_arrays_byte_exact']
    assert pair['totals']['CF_legal_edges_retained']==175 and pair['totals']['CF_legal_edges_lost']==0
    protected={**occ['source_hashes'],**normal['input_hashes'],**attr['protected'],**capreg['protected'],**capreg['source_files']}
    protected.update({v['path']:v['sha256'] for v in hist['source_reports'].values()})
    protected[str(C.OUT/'SEAM_SLOT_C200_REVIEW.zip')]='f68d1b07633f4ca34f8c0a5d6a8dd894631bf4d8ad08e2ee92413948493c5c7f'
    verify(protected);verify(normal['output_hashes']);verify(attr['files']);verify(cap['files']);verify(pair['inputs'])
    for rec in C.read(OUT/'capacity/score_completion.json')['receipts']:
        verify({rec['path']:rec['sha256']});verify(C.read(rec['path'])['files'])
    rows=[];snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    paths=set(capreg['source_files'])|{str(p) for p in CODE.glob('*.py')}
    for item in sorted(paths):
        p=C.resolve(item);name=str(p).replace('\\','/').replace('/mnt/e/','E__/').replace(':','_').strip('/').replace('/','__')
        target=snapshot/name;payload=p.read_bytes()
        if target.exists():assert target.read_bytes()==payload
        else:target.write_bytes(payload)
        rows.append(dict(path=str(p),sha256=C.sha(p),bytes=len(payload),snapshot=str(target)))
    C.csvwrite(OUT/'SOURCE_HASHES.csv',rows)
    diff=''.join(''.join(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(C.ROOT)).replace('\\','/'))) for p in sorted(CODE.glob('*.py')))
    (OUT/'code.diff').write_text(diff,encoding='utf-8')
    C.write(OUT/'gate_decisions.json',dict(status='COMPLETE',occupancy='NOT_STABLE: positive62/62, Null0/4 after analytic correction',
        normal_loss='REJECTED_AS_STATED: rejects50/62 valid local source interfaces',
        capacity='BOUNDED_TOPOLOGY_POSITIVE_WITH_COST: NM18to0, all175CFedges kept, FF+0.786pp, I tail remains',
        RL='NOT_RUN_GATE_NOT_MET',generator_forwards=0,head_forwards=0,D0_forwards=0,backwards=0,optimizer_updates=0))
    C.write(OUT/'resources_initial_amendment.json',dict(status='CLARIFICATION',windows_CIM='ACCESS_FAILED',
        initial_RAM_zero_fields='UNAVAILABLE_NOT_ZERO_RAM',usable_memory_source='resources_wsl_initial.json',scientific_definition_changed=False))
    ledger=[dict(stage='occupancy',status=occ['status'],seconds=occ['wall_seconds'],model_calls=0),
        dict(stage='normal',status=normal['status'],seconds=normal['seconds'],model_calls=0),
        dict(stage='attribution',status=attr['status'],seconds=attr['seconds'],new_scans=16,model_calls=0),
        dict(stage='capacity',status=cap['status'],geometry_outputs=16,quality_scans=16,baseline_replay=16,model_calls=0),
        dict(stage='visual',status='ACTUALLY_REVIEWED_AI_PROXY_NONBLIND',paired_requests=16,model_calls=0),
        dict(stage='RL',status='NOT_RUN_GATE_NOT_MET',updates=0,rollouts=0)]
    (OUT/'ledger.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in ledger),encoding='utf-8')
    files={str(p):C.sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name not in ('completion.json','package_receipt.json')}
    result=dict(experiment='R2_SEAM_GEOMETRY_GUARD_V1',status='COMPLETE_BOUNDED_CPU_GATES',RL='NOT_RUN_GATE_NOT_MET',
        generator_forwards=0,head_forwards=0,D0_forwards=0,training_updates=0,new_noise=0,new_pre_geo_scores=16,
        new_post_geometries=16,new_post_scores=16,actual_AI_PROXY_paired_reviews=16,default_changed=False,
        cleanup=False,commit_push=False,protected=protected,files=files)
    C.write(OUT/'completion.json',result)
    selected=[]
    for p in OUT.rglob('*'):
        if not p.is_file() or p.suffix=='.zip' or p.name=='package_receipt.json':continue
        rel=p.relative_to(OUT); parts=rel.parts
        if 'evaluation' in parts or 'pre_geo' in parts:continue
        if p.suffix in ('.md','.json','.jsonl','.csv','.html','.png','.diff','.py') or 'source_snapshot' in parts:selected.append(p)
    zp=OUT/'SEAM_GEOMETRY_GUARD_REVIEW.zip'
    with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for p in sorted(selected):z.write(p,str(p.relative_to(OUT)).replace('\\','/'))
    with zipfile.ZipFile(zp) as z:assert z.testzip() is None
    if zp.stat().st_size>=10_000_000:raise RuntimeError('Review archive exceeds10MB; retain and report instead of replacing silently')
    C.write(OUT/'package_receipt.json',dict(path=str(zp),bytes=zp.stat().st_size,sha256=C.sha(zp),members=len(selected),
        contents='Report, full per-output CSVs and role/edge ledgers, actual source and scorer snapshot, four reviewed comparison figures; large pair caches and meshes stay local'))
    return dict(status=result['status'],RL=result['RL'],protected_files=len(protected),artifacts=len(files),zip_bytes=zp.stat().st_size)

if __name__=='__main__':print(json.dumps(run()))
