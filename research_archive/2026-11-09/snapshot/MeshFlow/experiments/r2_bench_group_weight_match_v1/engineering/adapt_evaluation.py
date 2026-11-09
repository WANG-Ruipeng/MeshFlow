"""One-time source adaptation; no models or old artifact mutations."""
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'r2/pilots/bench_high_joint_coverage_v1'
DST=ROOT/'r2/pilots/bench_group_weight_match_v1'

def replace_function(s,name,new):
    tree=ast.parse(s);node=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
    lines=s.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+new.strip()+'\n'+''.join(lines[node.end_lineno:])

s=(SRC/'evaluation.py').read_text(encoding='utf-8')
s=s.replace('Wider HIGH evaluation: frozen narrow B38 plus new B8/W46.', 'Group-weight matched evaluation: B/W92 read-only; only M46 is new.')
s=s.replace("OUT = ROOT/'experiments/r2_bench_high_joint_coverage_v1'", "OUT = ROOT/'experiments/r2_bench_group_weight_match_v1'\nWIDE = ROOT/'experiments/r2_bench_high_joint_coverage_v1'")
s=s.replace("LABELS = ('B', 'W')", "LABELS = ('B', 'W', 'M')")
s=s.replace('NEW_LABELS = LABELS', "NEW_LABELS = ('M',)")
s=s.replace("ARMS = {'W':'WIDE_HIGH'}", "ARMS = {'M':'W_MATCHED'}")
s=s.replace("VERSION = 'bench_high_joint_coverage_evaluation_v1'", "VERSION = 'bench_group_weight_match_evaluation_v1'")
s=replace_function(s,'new_requests', "def new_requests(label):\n    if label not in LABELS:raise ValueError('Unknown label')\n    return requests(label) if label=='M' else []")
s=replace_function(s,'budget', "def budget():\n    n=len(requests())\n    return dict(requests=n,new_RAW=n,new_D0=n,generator_forwards=50*n,new_score_rows=3*n,reused_score_rows=6*n,total_score_rows=9*n)")
s=replace_function(s,'root_for', "def root_for(label,q=None):\n    if label not in LABELS:raise ValueError('Unknown label')\n    if label=='B' and (q is None or cached_b(q)):return OLD\n    return WIDE if label in ('B','W') else OUT")
s=replace_function(s,'raw_completion', "def raw_completion(label):\n    return (WIDE if label in ('B','W') else OUT)/'evaluation'/label/'raw_complete_full.json'")
s=replace_function(s,'prepare_inputs', '''
def prepare_inputs():
    """Byte-for-byte reuse, including all legacy manifest metadata. No random draws."""
    source=WIDE/'data/eval_manifest.json';raw=source.read_bytes()
    if MANIFEST.exists():
        if MANIFEST.read_bytes()!=raw:raise ValueError('Original46 manifest bytes differ')
    else:
        MANIFEST.parent.mkdir(parents=True,exist_ok=True)
        with MANIFEST.open('xb') as f:f.write(raw)
    result=read(MANIFEST)
    if len(result['requests'])!=46:raise ValueError('Original panel cardinality')
    return result
''')
s=replace_function(s,'prepare_panel', '''
def prepare_panel():
    """Authenticate 92 old outputs/276 score rows; keep the old manifest unchanged."""
    done=OUT/'audits/BW_cache_audit.json'
    if done.exists():
        r=read(done);verify(r['files']);verify(r['protected']);return r
    from .common import resources
    rp=OUT/'audits/eval_resources_before.json'
    if not rp.exists():immutable(rp,dict(resources=resources(),CPU_workers=1,threads_per_worker=1,model_calls=0,GPU_calls=0))
    original=prepare_inputs();qs=requests();fields=original['metric_fields']
    if len(fields)!=len(set(fields)) or len(fields)!=177:raise ValueError('Metric dictionary')
    frozen={str(p):sha(p) for p in (WIDE/'data/eval_manifest.json',WIDE/'results/completion.json')}
    cells=[];oldrows=read(WIDE/'results/metric_rows.json')
    original_rows={(r['label'],r['stage'],r['request_id']):r for r in oldrows}
    if len(oldrows)!=len(original_rows) or len(oldrows)!=276:raise ValueError('Original B/W score universe')
    frozen[str(WIDE/'results/metric_rows.json')]=sha(WIDE/'results/metric_rows.json')
    for q in qs:
        if not 128<=q['N']<=799 or not 2<=q['K']<q['N']:raise ValueError('Native N/K')
        for key,hkey in (('inference_npz','inference_npz_sha256'),('metadata_npz','metadata_sha256'),('source_npz','source_npz_sha256'),('noise_npz','noise_sha256')):
            p=resolve(q[key]);h=sha(p)
            if h!=q[hkey]:raise ValueError('Input SHA '+key)
            frozen[str(p)]=h
        p=resolve(q['query_descriptor']['path'])
        if sha(p)!=q['query_descriptor']['sha256']:raise ValueError('Query SHA')
        frozen[str(p)]=sha(p)
        if 'gallery_bounds' not in q or 'gallery_depth_bounds' not in q:raise ValueError('Fixed frame missing')
        for label in ('B','W'):
            for kind in ('raw','post'):
                folder=producer(label,q,kind)
                receipt=read(folder/'receipt.json')
                frozen.update(receipt['files']);frozen[str(folder/'receipt.json')]=sha(folder/'receipt.json')
            for stage in STAGES:
                row,score,folder=authenticate_score(label,q,stage)
                old=original_rows[label,stage,q['request_id']]
                if any(m not in row or row[m]!=old[m] for m in fields):raise ValueError('Original metric missing/changed')
                cells.append(dict(label=label,request_id=q['request_id'],stage=stage,source=str(folder)))
                for name in ('row.json','score.json','complete.json'):
                    p=folder/name;frozen[str(p)]=sha(p)
            complete=root_for(label,q)/'cpu'/label/q['request_id']/'complete.json'
            rr=read(complete);verify(rr['files']);frozen.update(rr['files']);frozen[str(complete)]=sha(complete)
    if len(cells)!=276:raise ValueError('Cached score count')
    immutable(OUT/'data/evaluation_plan.json',dict(schema=VERSION,status='FROZEN',labels=list(LABELS),
        new_labels=list(NEW_LABELS),visual_labels=['W','M'],budget=budget(),
        original_manifest=str(WIDE/'data/eval_manifest.json'),original_manifest_sha256=sha(MANIFEST),
        original_manifest_bytes_preserved=True,new_noise_arrays=0,new_queries=0,new_camera_or_sections=0,
        metadata_note='eval_manifest is the original W manifest byte-for-byte; current labels/budgets are only in this plan.',
        primary_contrast='M-W',secondary_quantitative_contrast='M-B',
        chain='single official Bench D0 -> GEO_C/source winding -> HINGE_OPP015 -> ALPHA -> HOLE'))
    result=dict(status='PASS',requests=46,old_reused_requests=92,score_rows_reused=276,numeric_fields=177,
        source_score_rows=cells,protected=frozen,files={str(p):sha(p) for p in (MANIFEST,OUT/'data/evaluation_plan.json')},
        model_calls=0,scoring_calls=0,GPU_calls=0,new_noise=0,new_queries=0,code_sha256=sha(__file__))
    immutable(done,result);return result
''')
s=s.replace('reused_score_rows=114','reused_score_rows=276')
s=s.replace("if label not in LABELS or (label=='B' and cached_b(q)):raise ValueError('Cached B scorer rerun forbidden')", "if label not in NEW_LABELS:raise ValueError('Cached B/W scorer rerun forbidden')")
s=s.replace("if label not in LABELS or (label=='B' and cached_b(q)):raise ValueError('Cached B cascade rerun forbidden')", "if label not in NEW_LABELS:raise ValueError('Cached B/W cascade rerun forbidden')")
s=s.replace("raise ValueError('B/W CPU labels required')", "raise ValueError('Only M CPU work authorized')")
s=s.replace("result_origin='REUSED_READ_ONLY' if label=='B' and cached_b(q) else 'NEW_REGISTERED'", "result_origin='REUSED_READ_ONLY' if label in ('B','W') else 'NEW_REGISTERED'")
s=s.replace("for ref,cand in (('B','W'),):", "for ref,cand in (('W','M'),('B','M')):")
s=replace_function(s,'label_spec', '''
def label_spec(label):
    if label not in LABELS:raise ValueError('Unknown label')
    if label in ('B','W'):
        from r2.pilots.bench_high_joint_coverage_v1.evaluation import label_spec as prior
        return prior(label)
    return dict(mode='PLAIN',arm='W_MATCHED',local_step=1500,global_step=22500,subset='full',
        schema='bench_group_weight_match_generator_v1',generator_path=str(OUT/'training/W_MATCHED/generator_step1500.pt'))
''')
s=s.replace("inherited=read(OLD/'evaluation/inference_registrations'/('B__'+kind+'.json'))['code']", "inherited=read(WIDE/'evaluation/inference_registrations'/('W__'+kind+'.json'))['code']")
s=replace_function(s,'reuse_spec', "def reuse_spec(q,label):return dict(root=str(root_for(label,q)),label=label) if label in ('B','W') else None")
s=s.replace("def sample(label,estimated_peak_mib=3500):\n    if label not in LABELS:", "def sample(label,estimated_peak_mib=3500):\n    if label not in NEW_LABELS:")
s=s.replace("def sample_run(label, estimated_peak_mib=2800):\n    spec", "def sample_run(label, estimated_peak_mib=2800):\n    if label not in NEW_LABELS:raise ValueError('Cached B/W generation forbidden')\n    spec")
s=s.replace("def d0(label,estimated_peak_mib=3500):\n    if label not in LABELS:", "def d0(label,estimated_peak_mib=3500):\n    if label not in NEW_LABELS:")
s=s.replace("        if cached_b(q):\n            prior=arrays(OLD/'evaluation/B/layouts'/q['condition_id']/'layout.npz')", "        if True:\n            prior=arrays(WIDE/'evaluation/W/layouts'/q['condition_id']/'layout.npz')")
s=s.replace('reused_old=cached_b(q)', 'reused_old=True')
s=replace_function(s,'all_raw_barrier', '''
def all_raw_barrier():
    verify(read(OUT/'audits/BW_cache_audit.json')['files'])
    receipts=[]
    for label in NEW_LABELS:
        verify_raw(label);receipts.extend(raw_receipt(q,label) for q in new_requests(label))
    if len(receipts)!=budget()['new_RAW']:raise ValueError('All46 M RAW must precede D0')
    return receipts
''')
s=s.replace('bench_high_joint_coverage_registered_new_requests','bench_group_weight_match_registered_new_requests')
s=s.replace('bench_high_joint_coverage_single_D0','bench_group_weight_match_single_D0')
ast.parse(s);(DST/'evaluation.py').write_text(s,encoding='utf-8')

s=(SRC/'report_results.py').read_text(encoding='utf-8')
s=s.replace("CONTRASTS=(('B','W'),)", "CONTRASTS=(('W','M'),('B','M'))\nVISUAL_LABELS=('W','M')\nVISUAL_CONTRASTS=(('W','M'),)")
s=s.replace("VERSION='bench_high_joint_coverage_report_v1'", "VERSION='bench_group_weight_match_report_v2'")
s=s.replace('def prepare():','def prepare_gallery():',1)
start=s.index('    protocol=dict(');end=s.index('\n\n\ndef one(',start)
old=s[start:end]
template=old[:old.index("    E.immutable(OUT/'visual_review/protocol.json'")]
template=template.replace('    protocol=dict(', '    return dict(',1)
s=s[:start]+"    return dict(status='FROZEN',sheets=3*len(E.requests()),cells=len(VISUAL_LABELS)*len(E.STAGES)*len(E.requests()),model_calls=0,ratings_generated=0)\n\n\ndef protocol_template():\n    \"\"\"Return metadata only; root owns the version2 definitions and anchors.\"\"\"\n"+template+'''

def prepare():
    result=prepare_gallery()
    definition=OUT/'visual_review/v2_definition.json'
    if not definition.exists():
        return dict(result,protocol='PENDING_ROOT_V2_DEFINITION')
    declared=E.read(definition)
    protocol=protocol_template()
    protocol.update(interpretation_version=2,definition=declared,definition_sha256=E.sha(definition),
                    no_historical_label_pooling=True,quantitative_only_labels=['B'])
    E.immutable(OUT/'visual_review/protocol.json',protocol)
    return dict(result,protocol='FROZEN_V2')
'''+s[end:]
# Gallery and actual visual code use only W/M; numeric tables retain B/W/M.
cut=s.index('def deletion_rows')
s=s[:cut].replace('E.LABELS','VISUAL_LABELS')+s[cut:]
cut=s.index('def visual():');s=s[:cut]+s[cut:].replace('E.LABELS','VISUAL_LABELS').replace('for a,b in CONTRASTS:', 'for a,b in VISUAL_CONTRASTS:')
s=s.replace('B narrow HIGH; W WIDE_HIGH. Same frozen noise, frame and sections.', 'W original wide HIGH; M group-weight matched. Same frozen noise, frame and sections.')
s=s.replace('# Bench wider HIGH pool: complete numerical results','# Bench group-weight match: complete numerical results')
s=s.replace('B is the completed narrow HIGH reference (38 cached +8 new outputs). W is the one new1500-step wide HIGH arm from the identical P0. Contrast: W-B.', 'B and W are completed read-only quantitative references. M is the new1500-step group-weight matched arm from identical P0. Primary M-W; secondary M-B.')
s=s.replace('| metric | B narrow | W wide | unit |','| metric | B narrow | W wide | M matched | unit |')
s=s.replace('| --- | ---: | ---: | --- |','| --- | ---: | ---: | ---: | --- |')
start=s.index('    drift=[];history=');end=s.index("    dest=OUT/'visual_review/summary_v1'",start)
s=s[:start]+"    # V2 paired observations only. Historical labels are never pooled or re-counted.\n"+s[end:]
s=s.replace(',historical_B_label_drift=drift','')
s=s.replace("'Do not copy historic B observations; record label drift separately.'", "'Do not copy or pool historical B/W observations. Apply root-owned V2 anchors to W/M only.'")
s=s.replace("choices=('prepare','render','tables','visual')", "choices=('prepare','prepare-gallery','render','tables','visual')")
s=s.replace("{'prepare':prepare,'render':", "{'prepare':prepare,'prepare-gallery':prepare_gallery,'render':")
ast.parse(s);(DST/'report_results.py').write_text(s,encoding='utf-8')

s=(SRC/'evaluation_closeout.py').read_text(encoding='utf-8')
s=s.replace('B_cache_audit.json','BW_cache_audit.json')
s=s.replace('new_RAW=54,new_D0=54,generator_forwards=2700','new_RAW=46,new_D0=46,generator_forwards=2300')
s=s.replace('new_score_rows=162,reused_score_rows=114,total_score_rows=276','new_score_rows=138,reused_score_rows=276,total_score_rows=414')
s=s.replace("{'B':8,'W':46}", "{'B':0,'W':0,'M':46}")
s=s.replace("root==E.OLD", "label in ('B','W')")
s=s.replace("{'REUSED':114,'NEW':162}", "{'REUSED':276,'NEW':138}")
s=s.replace("cached_B38='Read-only original arrays, queries, scores and camera/sections'", "cached_BW92='Read-only original arrays, queries, scores and camera/sections'")
# Only new M needs new call accounting. Quantitative loop still includes all labels.
s=s.replace("    for label in E.LABELS:\n        raw_complete", "    for label in E.NEW_LABELS:\n        raw_complete")
s=s.replace("len(registration['RAW_barrier'])==54", "len(registration['RAW_barrier'])==46")
s=s.replace('dict(RAW=54,generator_forward=2700,D0=54,GEO=54)', 'dict(RAW=46,generator_forward=2300,D0=46,GEO=46)')
s=s.replace('All54 RAW before any D0','All46 M RAW before any D0')
s=s.replace('==(276,162,114)', '==(414,138,276)')
s=s.replace('len(by)==len(rows)==276','len(by)==len(rows)==414')
s=s.replace("{'NEW_REGISTERED':162,'REUSED_READ_ONLY':114}", "{'NEW_REGISTERED':138,'REUSED_READ_ONLY':276}")
s=s.replace('scoring=dict(cells=276,new=162,reused=114','scoring=dict(cells=414,new=138,reused=276')
s=s.replace('original_B_cache_unchanged=True','original_BW_cache_unchanged=True')
s=s.replace("    train_rows,train_totals=_train_audit(book)", "    cache=book.read(OUT/'audits/BW_cache_audit.json');book.verify(cache['protected']);book.verify(cache['files'])\n    train_rows,train_totals=_train_audit(book)")
ast.parse(s);(DST/'evaluation_closeout.py').write_text(s,encoding='utf-8')
print('Wrote evaluation.py, report_results.py, evaluation_closeout.py; all AST pass.')
