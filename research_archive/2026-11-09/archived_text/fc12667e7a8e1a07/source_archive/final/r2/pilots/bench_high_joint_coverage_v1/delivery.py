"""CPU-only delivery summaries/archive. Never imports or invokes a model/scorer."""
import argparse, collections, csv, difflib, hashlib, json, statistics, zipfile
from pathlib import Path

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT/'experiments/r2_bench_high_joint_coverage_v1'
CODE = ROOT/'r2/pilots/bench_high_joint_coverage_v1'
GROUPS = ('OLD','FOCUS','DEV','HOLDOUT','HIGH_TRAIN','HIGH_HOLDOUT','NEW_WIDE_TRAIN')
LABELS = ('B','W')

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n' if not isinstance(value,str) else value
    if path.exists():
        assert path.read_bytes()==text.encode('utf-8'), 'Refuse overwrite: '+str(path)
    else: path.write_text(text,encoding='utf-8')
def csvwrite(path, data):
    import io
    f=io.StringIO(newline=''); w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader();w.writerows(data)
    write(path,f.getvalue())
def macro(subset,key):
    by=collections.defaultdict(list)
    for r in subset: by[r['parent_id']].append(float(r[key]))
    return statistics.fmean(statistics.fmean(v) for v in by.values())

def numeric():
    dest=OUT/'results/delivery_v1'
    metrics=rows(OUT/'results/analysis_v1/primary_parent_macro.csv')
    by={(r['report_group'],r['stage'],r['label'],r['metric']):r for r in metrics}
    selected={'CF_illegal_affected_free_fraction':'CF%', 'FF_illegal_affected_free_fraction':'FF%',
      'ConflictCov_C':'I% (PASS)', 'free_minimum_angle_lt5_fraction':'小角%',
      'C_F_has_valid_edge':'有合法接口%',
      'hinge_diag_legal_opposed_edge_C_reached_free_area_fraction':'定向可达面积%',
      'actual_free_faces':'原始槽保留面数','free_total_surface_area':'自由面面积',
      'deleted_free_faces':'GEO删面数','hinge_diag_isolated_1face_edge_components':'合法边图单面分量'}
    lines=['# 固定评价数值摘要','', '父等权；B=旧窄HIGH，W=新宽HIGH，同起点同1500步。比例显示为百分数；NA不转成0。完整原177字段另存。','']
    for group in GROUPS:
        lines += ['## '+group+' / CLEAN_FINAL_POST','','|指标|B窄池|W宽池|W−B|','|---|---:|---:|---:|']
        for key,name in selected.items():
            rr=[by[group,'CLEAN_FINAL_POST',label,key] for label in LABELS]
            vals=[None if not r['mean'] else float(r['mean'])*(100 if '%' in name else 1) for r in rr]
            fmt=lambda v:'NA' if v is None else f'{v:.3f}'
            delta=None if vals[0] is None or vals[1] is None else vals[1]-vals[0]
            lines.append('|'+name+'|'+'|'.join(fmt(v) for v in vals+[delta])+'|')
        lines.append('')
    area=rows(OUT/'results/analysis_v1/deletion_area_per_output.csv'); fates=rows(OUT/'results/analysis_v1/deletion_fate_per_slot.csv')
    retention=[];reasons=[]
    for group in GROUPS:
        for label in LABELS:
            sub=[r for r in area if (r['report_group'],r['label'],r['stage'])==(group,label,'CLEAN_FINAL_POST')]
            rec=dict(report_group=group,label=label,parents=len({r['parent_id'] for r in sub}),requests=len(sub))
            for key in ('raw_free_faces','retained_original_free_faces','deleted_original_free_faces','raw_free_area','pre_geo_free_area',
                        'deleted_raw_reference_area','deleted_pre_geo_reference_area','actual_final_free_area','deleted_raw_area_fraction'):
                rec[key]=macro(sub,key)
            for r in sub:
                r['final_over_raw_area']=float(r['actual_final_free_area'])/float(r['raw_free_area'])
                r['final_retained_slot_fraction']=float(r['retained_original_free_faces'])/float(r['raw_free_faces'])
            rec['final_over_raw_area']=macro(sub,'final_over_raw_area')
            rec['final_retained_slot_fraction']=macro(sub,'final_retained_slot_fraction')
            retention.append(rec)
            fs=[r for r in fates if (r['report_group'],r['label'],r['stage'])==(group,label,'CLEAN_FINAL_POST') and r['retained']=='False']
            counts=collections.Counter(r['GEO_reason'] for r in fs)
            for reason,count in sorted(counts.items()):
                reasons.append(dict(report_group=group,label=label,reason=reason,deleted_slots=count,request_count=len(sub),
                    raw_reference_area_sum=sum(float(r['raw_area']) for r in fs if r['GEO_reason']==reason),
                    pre_geo_reference_area_sum=sum(float(r['pre_geo_area']) for r in fs if r['GEO_reason']==reason)))
    csvwrite(dest/'area_parent_macro.csv',retention);csvwrite(dest/'deletion_reason_counts.csv',reasons)
    write(dest/'NUMERICAL_SUMMARY.md','\n'.join(lines))
    return {'status':'COMPLETE','model_calls':0,'scoring_calls':0,'rows':len(retention)}

def visual_joint():
    data=rows(OUT/'visual_review/summary_v1/actual_observations.csv'); assert len(data)==276
    for r in data:
        r['three_axis_joint']=str(r['overall_shape']=='REASONABLE' and r['C_use']=='PLAUSIBLE' and r['support_retention']=='PRESENT')
    summary=[]; transitions=[]
    for group in GROUPS:
        for stage in ('RAW','BASE_POST','CLEAN_FINAL_POST'):
            for label in LABELS:
                s=[r for r in data if (r['report_group'],r['stage'],r['label'])==(group,stage,label)]
                summary.append(dict(report_group=group,stage=stage,label=label,requests=len(s),
                    parents=len({r['parent_id'] for r in s}),joint_count=sum(r['three_axis_joint']=='True' for r in s),
                    plausible=sum(r['C_use']=='PLAUSIBLE' for r in s),reasonable=sum(r['overall_shape']=='REASONABLE' for r in s),
                    support_present=sum(r['support_retention']=='PRESENT' for r in s),
                    fragments_or_gaps_yes=sum(r['visible_fragment_or_gap']=='YES' for r in s),
                    layering_or_fold_yes=sum(r['layering_or_fold']=='YES' for r in s)))
    by={(r['label'],r['stage'],r['request_id']):r for r in data}
    for r in data:
        if r['label']!='W':continue
        for ref in ('B',):
            a=by[ref,r['stage'],r['request_id']]
            transitions.append(dict(reference=ref,candidate='W',request_id=r['request_id'],parent_id=r['parent_id'],
                report_group=r['report_group'],stage=r['stage'],reference_joint=a['three_axis_joint'],candidate_joint=r['three_axis_joint']))
    dest=OUT/'results/delivery_v1'
    csvwrite(dest/'three_axis_per_output.csv',data);csvwrite(dest/'three_axis_counts.csv',summary);csvwrite(dest/'three_axis_transitions.csv',transitions)
    write(dest/'VISUAL_SEMANTICS.md','三轴交集仅为既有实际AI_PROXY标签的描述性联合计数：REASONABLE且PLAUSIBLE且PRESENT；不是新冻结主指标，不认证无碎片、无折层、无相交或拓扑接续。五轴原标签、UNKNOWN/UNCERTAIN和逐项备注完整保留。没有从数值分数产生标签。\n')
    return {'status':'COMPLETE','actual_observations':276,'new_ratings':0}

def archive():
    def archived_relative(p):
        try:return p.relative_to(ROOT)
        except ValueError:return Path('_external')/p.relative_to(ROOT.parent)
    codefiles={}
    maps=[read(OUT/'training_registration.json')['training_code_hashes'],read(OUT/'evaluation/score_registration.json')['source_hashes']]
    for p in sorted((OUT/'evaluation/inference_registrations').glob('*.json')):maps.append(read(p)['code'])
    for mapping in maps:
        for p,h in mapping.items():
            p=Path(p)
            if p.suffix!='.py':continue
            assert sha(p)==h,'Registered source changed '+str(p)
            codefiles[p]=h
    for p in CODE.glob('*.py'):codefiles[p]=sha(p)
    for p in OUT.glob('*.py'):codefiles[p]=sha(p)
    for p,h in codefiles.items():
        target=OUT/'source_archive/final'/archived_relative(p)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==h
        else:target.write_bytes(p.read_bytes())
    manifest=[dict(source=str(p),archive=str((OUT/'source_archive/final'/archived_relative(p)).relative_to(OUT)),sha256=h,bytes=p.stat().st_size) for p,h in sorted(codefiles.items())]
    write(OUT/'source_archive/final_source_hashes.json',manifest)
    diff=''.join(''.join(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(keepends=True),fromfile='/dev/null',tofile=str(p.relative_to(ROOT)))) for p in sorted([*CODE.glob('*.py'),*OUT.glob('*.py')]))
    write(OUT/'code.diff',diff)
    return {'status':'COMPLETE','actual_code_files':len(manifest),'production_files_modified':0}

def package():
    assert read(OUT/'audits/evaluation_closeout.json')['status']=='PASS'
    assert (OUT/'REPORT.md').exists()
    files=set()
    for name in ('REPORT.md','PROTOCOL.md','RUNBOOK.md','MODELS.md','model_index.json','AUTHORIZATION.json','training_registration.json','code.diff'):
        p=OUT/name
        if p.is_file():files.add(p)
    for folder in ('data','results','training_closeout','visual_review','audits','dispatch','backend_geo_only'):
        for p in (OUT/folder).rglob('*'):
            if p.suffix.lower() in ('.json','.csv','.md') and p.stat().st_size<12_000_000:files.add(p)
    files.update((OUT/'source_archive/final').rglob('*.py'))
    files.update((OUT/'backend_geo_only/source_archive').rglob('*.py'))
    files.update((OUT/'backend_geo_only/gallery').rglob('*.png'))
    files.update((OUT/'training_closeout').rglob('*.png'))
    files.add(OUT/'source_archive/final_source_hashes.json')
    for name in ('generator_step1500.json','checkpoint_step1500.json','status.json'):
        files.add(OUT/'training/WIDE_HIGH'/name)
    panel=read(OUT/'data/eval_manifest.json')['requests']
    new=[q for q in panel if q['report_group']=='NEW_WIDE_TRAIN']
    assert len(new)==8
    for q in new:
        for stage in ('RAW','BASE_POST','CLEAN_FINAL_POST'):
            p=OUT/'gallery'/(stage+'__'+q['request_id']+'.png');assert p.is_file();files.add(p)
    write(OUT/'PACKAGE_CONTENTS.md','包含完整逐输出177字段、逐父配对、实际五轴图审及解释漂移、数据与预算、实际源码hash/diff、W权重身份、四RAW CPU对照全部显示图及源码、全部新四父八请求三阶段24页图。完整主面板138页图集、大权重、固定数组及原始调用账本保留本地；不是独立可重训发行包。没有按结果选图。\n')
    files.add(OUT/'PACKAGE_CONTENTS.md')
    target=OUT/'R2_BENCH_HIGH_JOINT_COVERAGE_REVIEW.zip'
    assert not target.exists(),'Do not overwrite delivery'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    result=dict(status='COMPLETE',path=str(target),bytes=target.stat().st_size,sha256=sha(target),files=len(files),
        gallery_pages_included=24,model_weights_included=False,
        files_sha256={str(p.relative_to(OUT)):sha(p) for p in sorted(files)})
    write(OUT/'package_receipt.json',result)
    return {k:v for k,v in result.items() if k!='files_sha256'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('numeric','visual','archive','package'));a=p.parse_args()
    print(json.dumps({'numeric':numeric,'visual':visual_joint,'archive':archive,'package':package}[a.action](),ensure_ascii=False,indent=2))
