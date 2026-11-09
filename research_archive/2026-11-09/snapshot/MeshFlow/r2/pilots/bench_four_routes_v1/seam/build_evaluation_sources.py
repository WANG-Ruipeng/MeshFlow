"""One-time local adaptation of audited evaluation, without scientific calls."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[4];DEST=Path(__file__).parent
def fn(path,name):
    text=(ROOT/path).read_text(encoding='utf-8');node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
def replace_fn(text,name,new):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+new+'\n'+''.join(lines[node.end_lineno:])
def save(name,text):
    p=DEST/name
    if p.exists():raise RuntimeError('Refuse overwrite '+str(p))
    p.write_text(text,encoding='utf-8')

text=(ROOT/'r2/pilots/bench_group_weight_match_v1/evaluation.py').read_text(encoding='utf-8')
text=text.replace('Group-weight matched evaluation: B/W92 read-only; only M46 is new.', 'Original SEAM evaluation: B46 read-only; only S46 is new.')
text=text.replace('parents[3]','parents[4]').replace("OUT = ROOT/'experiments/r2_bench_group_weight_match_v1'","OUT = ROOT/'experiments/r2_bench_four_routes_v1/seam'")
text=text.replace("LABELS = ('B', 'W', 'M')","LABELS = ('B', 'S')").replace("NEW_LABELS = ('M',)","NEW_LABELS = ('S',)").replace("ARMS = {'M':'W_MATCHED'}","ARMS = {'S':'SEAM1500'}")
text=text.replace("label=='M'","label=='S'").replace("('B','W')","('B',)")
text=text.replace('reused_score_rows=6*n,total_score_rows=9*n','reused_score_rows=3*n,total_score_rows=6*n')
text=text.replace('reused_score_rows=276','reused_score_rows=138').replace('old_reused_requests=92,score_rows_reused=276','old_reused_requests=46,score_rows_reused=138')
text=text.replace('if len(cells)!=276:','if len(cells)!=138:')
text=text.replace("visual_labels=['W','M']","visual_labels=['B','S']")
text=text.replace("primary_contrast='M-W',secondary_quantitative_contrast='M-B'","primary_contrast='S-B',secondary_quantitative_contrast=None")
text=text.replace("for ref,cand in (('W','M'),('B','M')):","for ref,cand in (('B','S'),):")
text=text.replace('bench_group_weight_match','bench_four_routes_seam')
text=text.replace("all_native_seam_fields='NA_NOT_SEAM_REPRESENTATION'","seam_fields='B Native NA; S original seam metrics; closed C NA'")
text=text.replace("CPU.score_current(q,arrays(rawdir/'raw.npz'),dest,'RAW',False)","CPU.score_current(q,arrays(rawdir/'raw.npz'),dest,'RAW',True)")
text=text.replace("OUT/'cpu'/label/q['request_id'],False,regsha)","OUT/'cpu'/label/q['request_id'],True,regsha)")
text=text.replace("if score['seam_to_bulk']['status']!='NA_NOT_SEAM_REPRESENTATION':raise ValueError('Native seam status must remain NA')",
    "if label=='B' and score['seam_to_bulk']['status']!='NA_NOT_SEAM_REPRESENTATION':raise ValueError('Baseline Native seam status must remain NA')")
text=text.replace("if MANIFEST.read_bytes()!=raw:raise ValueError('Original46 manifest bytes differ')", "if read(MANIFEST)!=read(source):raise ValueError('Original46 request/manifest values differ')")
text=text.replace('original_manifest_bytes_preserved=True','original_manifest_values_preserved=True')
text=text.replace("metadata_note='eval_manifest is the original W manifest byte-for-byte; current labels/budgets are only in this plan.'", "metadata_note='All original46 request dictionaries and immutable arrays/query/camera/section values retained; only plan labels differ.'")
text=replace_fn(text,'label_spec', '''def label_spec(label):
    if label not in LABELS:raise ValueError('Unknown label')
    if label=='B':
        from r2.pilots.bench_high_joint_coverage_v1.evaluation import label_spec as old
        return old(label)
    return dict(mode='SEAM',arm='SEAM1500',local_step=1500,global_step=24000,subset='full',
        schema='bench_four_routes_seam_generator_v1',generator_path=str(OUT/'training/SEAM1500/generator_step1500.pt'))
''')
text=replace_fn(text,'load_head', '''def load_head(label,payload):
    if label!='S':raise ValueError('Only the new SEAM model loads a head')
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.seam.occupancy import OccupancyHead
    from r2.pilots.bench_full_seat_curriculum_v1.training_state import HEAD_STATE_SHA256,validate_head_state
    state=payload['head_state'];validate_head_state(state)
    with torch.random.fork_rng(devices=[]):head=OccupancyHead(24).eval().requires_grad_(False)
    head.load_state_dict(state,strict=True)
    if state_hash(head.state_dict())!=HEAD_STATE_SHA256:raise ValueError('Frozen historical head differs')
    return head.to('cuda'),dict(kind='FROZEN_HISTORICAL_OLD_S15000_HEAD',state_sha256=HEAD_STATE_SHA256,
        parameters=833,threshold_logit=0.,input='C_ONLY_24_FEATURES',source_labels_read=False,oracle=False)
''')
layout=fn('r2/pilots/bench_full_seat_curriculum_v1/sample.py','save_layout')
layout=layout.replace('from r2.pilots.bench_parts_convergence_v1.data import edge_features, inference_layout', 'from .runtime import edge_features, inference_layout\n    from r2.pilots.bench_seat_coverage_v1.sample import P')
text=replace_fn(text,'save_layout',layout)
text=text.replace('maximum_head_forwards=0,','maximum_head_forwards=21,').replace("representation='NATIVE_ALL_FREE'","representation='ORIGINAL_OWNER_ONLY_SEAM'")
text=text.replace('raw, trajectory, audit = S.rollout(', 'raw, trajectory, audit = rollout(')
text=text.replace('if total_head != 0:',"if total_head != reg['maximum_head_forwards']:")
roll=fn('r2/pilots/bench_control_transfer_v1/sample.py','rollout')
roll=roll.replace('    n,k=',"    import torch\n    from .runtime import native_execution,seam_forward\n    from r2.pilots.bench_control_transfer_v1 import train as T\n    n,k=",1)
text+='\n'+roll
text=replace_fn(text,'cpu_contract','''def cpu_contract():
    import numpy as np
    from .runtime import inference_layout,edge_features
    prepare_panel();checks=[]
    for q in requests():
        core,source=source_assets(q);edges,features=edge_features(core['C_vertices'],core['C_faces_original'],core['total_N'])
        for occupied in (np.zeros(len(edges),bool),np.ones(len(edges),bool)):
            layout=inference_layout(**core,occupied=occupied)
            if len(layout['seam_slots'])!=int(occupied.sum()):raise ValueError('SEAM occupancy layout differs')
        z=arrays(q['noise_npz'])['z']
        if z.shape!=(q['N']-q['K'],3,3) or z.dtype!=np.float32 or not np.isfinite(z).all():raise ValueError('Noise shape/dtype')
        inp=arrays(q['inference_npz']);query=arrays(q['query_descriptor']['path'])
        if set(inp)!={'C_vertices','C_faces_original','C_faces_model','N','K'}:raise ValueError('Non-C input')
        if len(query['C_points'])!=2048 or not np.isclose(query['C_weights'].sum(),1):raise ValueError('Query identity')
        if int(q['closed_C'])!=int(len(edges)==0):raise ValueError('Closed applicability')
        checks.append(dict(request_id=q['request_id'],N=q['N'],K=q['K'],source_winding_metadata_validated=True,
            noise_sha256=q['noise_sha256'],closed_C=not len(edges),boundary_edges=len(edges),model_input_C_only=True))
    result=dict(status='PASS',requests=len(checks),rows=checks,manifest_sha256=sha(MANIFEST),
        new_RAW_before_D0=46,model_calls=0,scoring_calls=0,new_noise=0,head_calls=0)
    immutable(OUT/'audits/eval_cpu_contract.json',result);return result
''')
# Ensure rollout is defined before the CLI's main guard executes.
pos=text.index("if __name__=='__main__':")
tail=text[pos:];rp=tail.index('\ndef rollout(')
main=tail[:rp];rollout_tail=tail[rp:]
text=text[:pos]+rollout_tail+'\n'+main
save('evaluation.py',text)

report=(ROOT/'r2/pilots/bench_group_weight_match_v1/report_results.py').read_text(encoding='utf-8')
report=report.replace("CONTRASTS=(('W','M'),('B','M'))","CONTRASTS=(('B','S'),)")
report=report.replace("VISUAL_LABELS=('W','M')","VISUAL_LABELS=('B','S')").replace("VISUAL_CONTRASTS=(('W','M'),)","VISUAL_CONTRASTS=(('B','S'),)")
report=report.replace('bench_group_weight_match_report_v2','bench_four_routes_seam_report_v2')
report=report.replace('Blue C / gray BULK / green BRIDGE; Native all-free, no SEAM','Blue C / orange SEAM / gray BULK / green BRIDGE; B Native, S owner-only SEAM')
report=report.replace('historical B/W observations','historical B/W observations').replace('to W/M only','to B/S only')
save('report_results.py',report)
print('Prepared independent sample/D0/177-field CPU evaluation and gallery; no models called.')
