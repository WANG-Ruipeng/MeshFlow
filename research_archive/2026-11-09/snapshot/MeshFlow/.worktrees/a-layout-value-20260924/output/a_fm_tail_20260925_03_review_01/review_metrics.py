from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd().resolve()))
import numpy as np
from experiments.a_radial.io_utils import file_hash,write_json
from experiments.a_radial.system_probe.ours_package import compare,require,read
from experiments.a_radial.system_probe import ours_metrics as m
run=Path('output/a_fm_tail_20260925_03').resolve();out=Path(__file__).resolve().parent
result=read(run/'artifacts/metrics/RESULTS.json');rows=result['rows'];mapping={(r['seed'],r['sample_id'],r['arm']):r for r in rows}
expected={(s,i,a) for s in (10,11) for i in m.expected_ids() for a in m.ARMS}
require(len(rows)==len(mapping)==384 and set(mapping)==expected and result['metric_success_count']==384,'Metric slots')
require(not result['provenance_errors'] and not result['failed_slots'],'Metric error ledger')
ph=file_hash(run/'RUN_PROTOCOL.json');ah=file_hash(run/'ARTIFACT_PROTOCOL.json')
require(result['generation_protocol_sha256']==ph and result['artifact_protocol_sha256']==ah,'Metrics protocol')
metric_protocol=read(run/'artifacts/metrics/METRIC_PROTOCOL.json')
require(metric_protocol['euler_steps']==32 and metric_protocol['auxiliary_excluded'] and metric_protocol['points_each_direction']==8192,'Primary metric scope')
for seed in (10,11):
    folder=run/f'artifacts/seed{seed}';sm=read(folder/'samples/SAMPLES.json');samples={r['sample_id']:r for r in sm['records']};identity=read(folder/'IDENTITY.json')
    for arm in m.ARMS:
        em=read(folder/'exports'/f'{arm}_MANIFEST.json');exports={r['sample_id']:r for r in em['records']}
        require(len(exports)==64,'Export slots')
        for sid in m.expected_ids():
            row=mapping[(seed,sid,arm)];e=exports[sid];fixed=mapping[(seed,sid,'fixed')]
            compare(row,read(run/'artifacts/metrics/attempts'/f'seed{seed}'/sid/(arm+'.json')),'Metric per-slot copy')
            require(row['AE_step']==3200 and row['FM_step']==64000 and row['euler_steps']==32 and row['auxiliary_excluded'],'Metric checkpoint budget')
            compare(row['generation_identity'],identity,'Metric generation identity')
            require(row['source_nodes_sha256']==e['source_nodes_sha256']==samples[sid]['nodes_hash'],'Shared mainfield')
            require(file_hash(folder/'exports'/e['mesh_path'])==e['mesh_sha256']==row['mesh_npz_sha256'],'Actual mesh SHA')
            compare(e['exact'],row['audit']['exact'],'Exact prior ledger')
            compare(e['topology'],row['audit']['topology'],'Topology prior ledger')
            require([x['seed'] for x in row['surface_metrics']]==[b+(seed-10)*64+int(sid) for b in m.SEED_BASES],'New same-field MC seeds')
            for x in row['surface_metrics']:
                require(x['points_per_direction']==8192 and np.isfinite(x['rms_bbox_percent']),'Metric sample count/numerics')
            for k,v in m.common_gate(row['quality'],row['audit'],fixed['quality'],fixed['audit']).items():compare(v,row[k],'RelativeFixed gate')
            for k,v in m.paired_metrics(row,fixed).items():compare(v,row[k],'Paired gain')
compare(m.summary_groups(rows),read(run/'artifacts/metrics/SUMMARY.json')['groups'],'Primary metric summary')
groups=[]
for seed in ('pooled',10,11):
    for arm in m.ARMS:
        rr=[r for r in rows if r['arm']==arm and (seed=='pooled' or r['seed']==seed)];gain=np.array([r['rms_gain_percent_two_measurements'] for r in rr])
        groups.append({'seed':seed,'arm':arm,'n':len(rr),'exact_valid':sum(r['audit']['valid'] for r in rr),
            'actual512':sum(r['quality']['face_count']==512 for r in rr),'qualified':sum(r['qualified'] for r in rr),
            'improved_both':sum(r.get('improved_both',False) for r in rr),'paired_gain_median':np.median(gain,axis=0).tolist(),
            'paired_gain_mean':gain.mean(0).tolist(),'paired_gain_min':gain.min(0).tolist(),
            'quality_min_margin_median':float(np.median([r['min_angle_margin_deg'] for r in rr])),
            'quality_p05_margin_median':float(np.median([r['p05_angle_margin_deg'] for r in rr]))})
failures=[{'seed':r['seed'],'sample_id':r['sample_id'],'arm':r['arm'],'faces':r['quality']['face_count'],
    'intersection_pairs':r['audit']['selfintersection_count'],'topology_pass':r['audit']['topology_pass'],
    'topology':r['audit']['topology']} for r in rows if not r['audit']['valid'] or r['quality']['face_count']!=512]
value={'status':'PRIMARY_OUTLET_METRIC_REVIEW_PASS_DISTRIBUTION_PENDING','source_sha256':file_hash(__file__),
    'results_sha256':file_hash(run/'artifacts/metrics/RESULTS.json'),'protocol_sha256':ph,'artifact_protocol_sha256':ah,
    'rows':384,'metric_failures':0,'groups':groups,'geometric_failures_preserved':failures,
    'prior_exact_ledger_crosschecked_not_rerun':True,'new_numeric_surface_sampling':False,'GPU_or_export_executed':False,
    'not_causal_comparison_of_A_gains_on_different_8k_64k_fields':True,'shared_strict_quality_contract_still_unresolved':True}
write_json(out/'METRICS_REVIEW.json',value)
print(json.dumps({'groups':groups,'failures':failures}),flush=True)
