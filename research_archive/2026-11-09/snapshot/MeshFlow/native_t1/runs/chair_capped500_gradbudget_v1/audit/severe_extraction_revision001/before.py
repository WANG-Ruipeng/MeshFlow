"""Extract already-scored parent risks and measured resource records; no scoring."""
from pathlib import Path
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict

OUT=Path(__file__).resolve().parents[1]
SOURCES={}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    SOURCES[str(p)]=digest(p)
    return json.loads(Path(p).read_text())
def read_lines(p):
    SOURCES[str(p)]=digest(p)
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def status(metric,lo,hi):
    if lo is None or hi is None:return 'INCOMPLETE'
    if metric=='ConflictCov_C':return 'FAIL' if lo>.1 else 'PASS' if hi<=.1 else 'INCOMPLETE'
    return 'FAIL' if hi<-.1 else 'PASS' if lo>=-.1 else 'INCOMPLETE'
def number(s):return None if s in ('','None','null') else float(s)
def pp(x):return 'NA' if x is None else f'{100*x:+.4f}'

source_csv=OUT/'analysis/per_parent.csv'
SOURCES[str(source_csv)]=digest(source_csv)
paired=list(csv.DictReader(source_csv.open(newline='')))
indexed={(r['panel'],r['candidate_label'],r['reference_label'],r['uid'],r['metric']):r for r in paired}
assert len(indexed)==len(paired)
gate_flags={};gate_states={};checks=0
for p in sorted((OUT/'analysis/original_joint_gates').glob('*/*.json')):
    gate=read(p)
    for side,reference in [('vs_START',gate['start_label']),('vs_matched',gate['matched_label'])]:
        key=(gate['panel'],gate['candidate_label'],reference)
        gate_states[key]={'path':str(p),'status':gate['status'],'numeric_status':gate['numeric_status'],
                          'severe_cost_caveat':gate['severe_cost_caveat']}
        for parent in gate[side]['parents']:
            for metric in ['ConflictCov_C','ExposedRetain']:
                row=indexed[key+(parent['uid'],metric)]
                for field,suffix in [('delta','delta'),('lower','delta_lower'),('upper','delta_upper')]:
                    a=number(row[field]);b=parent[metric+'_'+suffix]
                    assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-12),(p,key,parent['uid'],metric,field,a,b)
                    checks+=1
        for parent in gate[side]['severe_parent_costs']:
            gate_flags[key+(parent['uid'],)]=parent

MATCHED={'HIST_CAPPED500':'HIST_TEACHER500','L500':'T500','G500':'T500','L1000':'T1000','G1000':'T1000'}
groups=defaultdict(dict)
for r in paired:
    if r['metric'] not in ['ConflictCov_C','ExposedRetain']:continue
    if r['reference_label']!='START' and r['reference_label']!=MATCHED.get(r['candidate_label']):continue
    groups[(r['panel'],r['candidate_label'],r['reference_label'],r['uid'])][r['metric']]=r
rows=[]
for key,metrics in sorted(groups.items()):
    vals={m:{k:number(r[k]) for k in ['candidate_value','reference_value','delta','lower','upper']} for m,r in metrics.items()}
    st={m:status(m,v['lower'],v['upper']) for m,v in vals.items()}
    if all(s=='PASS' for s in st.values()):continue
    original=gate_flags.get(key)
    if key[:3] in gate_states:
        assert original is not None,('CSV severe not flagged by gate',key)
        assert st['ConflictCov_C']==original['I_cost']['status'] and st['ExposedRetain']==original['E_retention']['status']
    row=dict(panel=key[0],candidate_label=key[1],reference_label=key[2],uid=key[3],
        registered_noises=int(metrics['ConflictCov_C']['registered_noises']),
        evidence_scope='ORIGINAL_JOINT_GATE' if original else 'FROZEN_PAIRED_CSV_SUPPLEMENT',
        risk_status='CONFIRMED' if 'FAIL' in st.values() else 'POSSIBLE_OR_UNRESOLVED',
        triggering_metrics=';'.join(m for m,s in st.items() if s!='PASS'))
    for m,short in [('ConflictCov_C','I'),('ExposedRetain','E')]:
        row[short+'_status']=st[m]
        for field,value in vals[m].items():row[short+'_'+field]=value
        for field in ['delta','lower','upper']:row[short+'_'+field+'_pp']=100*vals[m][field] if vals[m][field] is not None else None
    g=gate_states.get(key[:3],{})
    row.update(joint_gate_status=g.get('status','NOT_APPLICABLE'),joint_numeric_status=g.get('numeric_status','NOT_APPLICABLE'),
               joint_severe_cost_caveat=g.get('severe_cost_caveat','NOT_APPLICABLE'),
               joint_gate_source=g.get('path',''),paired_csv_source=str(source_csv),paired_csv_sha256=SOURCES[str(source_csv)])
    rows.append(row)
assert set(gate_flags)=={(r['panel'],r['candidate_label'],r['reference_label'],r['uid']) for r in rows if r['evidence_scope']=='ORIGINAL_JOINT_GATE'}
save_csv(OUT/'severe_parent_costs.csv',rows)

audit=read(OUT/'audit/runtime_final.json')
before=read(OUT/'registration/resources_before_preparation.json')
snaps=read_lines(OUT/'resources/snapshots.jsonl')
mem=[]
for s in snaps:
    values={k:int(v) for k,v in re.findall(r'^(MemTotal|MemAvailable|MemFree):\s+(\d+) kB',s.get('memory',''),re.M)}
    if values:mem.append(dict(phase=s['phase'],**values))
stage_rows=[]
for p in sorted((OUT/'resources').glob('*_summary.json')):
    s=read(p);monitor=p.with_name(p.name.replace('_summary.json','_monitor.jsonl'))
    values=[]
    for r in read_lines(monitor):
        fields=r.get('gpu',{}).get('stdout','').strip().split(',')
        if len(fields)==5:
            name=fields[0].strip();total,used,free,util=[int(x.strip()) for x in fields[1:]]
            values.append(dict(name=name,total=total,used=used,free=free,util=util))
    intervals=[i for i in audit['GPU_occupancy']['intervals'] if i['stage']==s['stage']]
    assert values and max(x['used'] for x in values)==s['observed_device_used_peak_MiB']
    stage_rows.append(dict(stage=s['stage'],GPU_name=values[0]['name'],GPU_total_MiB=values[0]['total'],
        sampled_device_used_peak_MiB=s['observed_device_used_peak_MiB'],
        sampled_own_process_peak_MiB=s['observed_own_process_peak_MiB'],
        sampled_GPU_utilization_max_percent=max(x['util'] for x in values),sampled_free_min_MiB=min(x['free'] for x in values),
        monitor_samples=len(values),lock_intervals=len(intervals),lock_seconds=sum(i['seconds'] for i in intervals),
        scope='Device-wide sampled usage including desktop; own-process memory unavailable in WSL/WDDM',
        source_summary=str(p),source_summary_sha256=SOURCES[str(p)],source_monitor=str(monitor),source_monitor_sha256=SOURCES[str(monitor)]))
save_csv(OUT/'resource_costs.csv',stage_rows)
resources=dict(GPU_name=stage_rows[0]['GPU_name'],GPU_total_MiB=stage_rows[0]['GPU_total_MiB'],
    maximum_sampled_device_used_MiB=max(r['sampled_device_used_peak_MiB'] for r in stage_rows),
    own_process_peak='NA: WSL/WDDM did not expose numeric per-process memory',
    CPU_logical_cores=before['cpu_logical'],registered_cpu_threads_per_worker=2,
    initial_Windows_cpu_load_percent=before['cpu_load_percent'],initial_Windows_RAM_free_bytes=before['ram_free_bytes'],
    WSL_MemTotal_kB=sorted({m['MemTotal'] for m in mem}),
    sampled_WSL_MemAvailable_min_kB=min(m['MemAvailable'] for m in mem),
    sampled_WSL_MemAvailable_max_kB=max(m['MemAvailable'] for m in mem),
    GPU_lock_seconds=audit['GPU_occupancy']['closed_seconds'],GPU_lock_limit_seconds=audit['GPU_occupancy']['limit_seconds'],
    GPU_lock_intervals=audit['GPU_occupancy']['closed_intervals'],max_simultaneous_GPU_workers=1,
    timings_scope='Measured lock ownership includes model load, save, CPU waits; monitor peaks are sampled device-wide peaks, not process or allocator maxima.',
    stages=stage_rows)
(OUT/'audit/resource_costs.json').write_text(json.dumps(resources,indent=2)+'\n')

joint=[r for r in rows if r['evidence_scope']=='ORIGINAL_JOINT_GATE']
extra=[r for r in rows if r['evidence_scope']!='ORIGINAL_JOINT_GATE']
lines=['# Severe parent I/E costs (existing-result extraction)','',
    f'**{len(joint)} original joint-gate parent/comparison rows carry a severe-cost warning; these must be reported explicitly even if a numeric gate were to pass.**',
    '', 'All deltas below are candidate minus reference, in percentage points (pp). Positive I = worse conflict; negative E = worse exposed retention. Original severe thresholds are I increase >10 pp or E decrease >10 pp. Bounds are the original saved PARTIAL-aware bounds. CONFIRMED means the whole bound exceeds the adverse threshold; possible/unresolved flags remain separate.',
    '', 'OLD and FRESH each average the two registered noises within each parent (eight parents per panel). ALL, when shown as a supplementary CSV result, averages four registered noises within the same eight parents; it is not another independent parent panel.',
    '', 'The original `severe_cost_caveat` is a separate warning, not an extra undocumented numeric veto. Neither original `numeric_status` nor overall `status` incorporates it. Therefore numeric PASS alone does not establish a cost-free or acceptable promotion. Original definitions and statuses are unchanged.',
    '', '|Panel|Candidate|Reference|Parent UID|I delta [lower, upper] pp|I gate|E delta [lower, upper] pp|E gate|Risk|',
    '|---|---|---|---|---:|---|---:|---|---|']
for r in joint:
    lines.append(f"|{r['panel']}|{r['candidate_label']}|{r['reference_label']}|{r['uid']}|{pp(r['I_delta'])} [{pp(r['I_lower'])}, {pp(r['I_upper'])}]|{r['I_status']}|{pp(r['E_delta'])} [{pp(r['E_lower'])}, {pp(r['E_upper'])}]|{r['E_status']}|{r['risk_status']}|")
lines+=['',f'The accompanying [CSV](severe_parent_costs.csv) also retains {len(extra)} supplementary flagged rows already present in the frozen paired CSV (ALL and control-versus-START comparisons). Each row identifies whether its source is the original joint gate or the paired CSV. No new scores or point samples were generated.',
    '',f'Original joint-gate parent I/E values and bounds were cross-checked against the paired CSV: {checks} scalar comparisons passed (absolute difference <1e-12).',
    '', 'Resource evidence:', '',
    f"- GPU: {resources['GPU_name']}, {resources['GPU_total_MiB']} MiB total. Maximum monitored device-wide usage: {resources['maximum_sampled_device_used_MiB']} MiB. Own-process peak: NA under WSL/WDDM; device usage includes unrelated desktop processes.",
    f"- CPU: {resources['CPU_logical_cores']} logical cores; registered two threads per worker. Initial Windows CPU load {resources['initial_Windows_cpu_load_percent']}%; free RAM {resources['initial_Windows_RAM_free_bytes']/2**30:.3f} GiB. WSL total {resources['WSL_MemTotal_kB'][0]/2**20:.3f} GiB; available RAM across recorded dispatch snapshots {resources['sampled_WSL_MemAvailable_min_kB']/2**20:.3f} to {resources['sampled_WSL_MemAvailable_max_kB']/2**20:.3f} GiB. These are snapshots, not continuous CPU/RAM peak monitoring.",
    f"- GPU lock: {resources['GPU_lock_seconds']:.6f} seconds ({resources['GPU_lock_seconds']/60:.3f} minutes), {resources['GPU_lock_intervals']} closed intervals, one worker; limit {resources['GPU_lock_limit_seconds']} seconds. Includes the preserved 0-call loader failure, loading, saving and in-worker CPU waiting.",
    '- Stage-specific observed peaks and lock duration are in [resource_costs.csv](resource_costs.csv). Sampled peaks are not continuous hardware maxima; no process/allocator equivalence is claimed.',
    '', '|Stage|Device peak MiB|GPU utilization max %|Min free MiB|Lock seconds|', '|---|---:|---:|---:|---:|']
for r in stage_rows:lines.append(f"|{r['stage']}|{r['sampled_device_used_peak_MiB']}|{r['sampled_GPU_utilization_max_percent']}|{r['sampled_free_min_MiB']}|{r['lock_seconds']:.3f}|")
lines+=['','Extraction source and all input SHA256 bindings are retained under `audit/severe_parent_costs_receipt.json`. The extraction imports no scoring, model, mesh, or training module.']
(OUT/'severe_parent_costs.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
receipt=dict(status='PASS',source_path=str(Path(__file__).resolve()),source_sha256=digest(__file__),
    existing_joint_gate_scalar_crosschecks=checks,joint_flagged_rows=len(joint),supplemental_flagged_rows=len(extra),
    joint_confirmed_rows=sum(r['risk_status']=='CONFIRMED' for r in joint),joint_possible_or_unresolved_rows=sum(r['risk_status']!='CONFIRMED' for r in joint),
    parent_ids=sorted({r['uid'] for r in joint}),sources=SOURCES,
    outputs={str(OUT/p):digest(OUT/p) for p in ['severe_parent_costs.md','severe_parent_costs.csv','resource_costs.csv','audit/resource_costs.json']},
    scoring_calls=0,model_calls=0,geometry_calls=0)
(OUT/'audit/severe_parent_costs_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status='PASS',joint_flagged_rows=len(joint),supplemental_flagged_rows=len(extra),parents=receipt['parent_ids'],resources={k:v for k,v in resources.items() if k!='stages'},outputs=receipt['outputs']),indent=2))
