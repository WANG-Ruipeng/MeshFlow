"""Register the authorized continuation without changing old protocol or evidence."""
from pathlib import Path
import difflib
from .common import *

def prepare(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():return read_json(out/'protocol.json')
    from experiments.mf_h1_local0.control_mvp48.train_io import resource_snapshot
    resources=resource_snapshot();fields=resources['gpu']['stdout'].split(',')
    total,used,free=map(float,fields[-4:-1]);reserve=max(.15*total,2048.)
    old_protocol=read_json(OLD/'protocol.json');old_manifest=read_json(OLD/'precheck/test_manifest.json')
    protocol=dict(old_protocol)
    protocol.update(schema='NATIVE_CHAIR_FINITE_SURFACE_ATTENTION_V1_IMPLEMENTATION_REPAIR_R1',
        parent_protocol_path=str(OLD/'protocol.json'),parent_protocol_sha256=digest(OLD/'protocol.json'),
        authorization='User explicitly instructed continuing the next step and not stopping for missing implementation; repair instrumentation and finish bounded audits',
        scientific_changes='None to production 192 nodes, bandwidth, relation/attention, source model, loss, dataset, thresholds or default',
        formal_training_allowed=False,new_generations_allowed=False,
        historical_calls=dict(precheck_forward=10,precheck_backward=1),
        planned_new_calls=dict(precheck_forward=76,precheck_backward=20),
        planned_cumulative_calls=dict(precheck_forward=86,precheck_backward=21),
        original_hard_limits=dict(precheck_forward=96,precheck_backward=24),
        reused_baselines='Existing eight H_CONT original-repeat output arrays, checked against same checkpoint/input/runtime and four already-planned current H calls',
        cost_repeats=dict(H_CONT=2,MOMENT_XY=1,SURFACE_XY=2),
        cost_revision='H_CONT/SURFACE preserve two complete paired measurements per fixture. MOMENT has one complete pair per fixture, descriptive only, to retain original total backward limit after the old lost measurement.',
        stopping='Finish finite implementation/cost audit and independently registered CPU reference; do not bypass numerical gates or start training')
    protocol['real_gate']=dict(old_protocol['real_gate'],planned_forwards=76,planned_backwards=20,
        schedule=dict(relation_zero_forwards=8,cost_warmup_forwards=12,cost_measured_forwards=20,cost_backwards=20,
            nonzero_permutation_forwards=24,padding_forwards=12),historical_H_zero_forwards_reused=8)
    protocol['cost_gate']=dict(old_protocol['cost_gate'],initial_total_mib=total,initial_used_mib=used,initial_free_mib=free,
        reserve_mib=reserve,available_process_budget_mib=free-reserve,
        measured_forward_backward_pairs_per_task_arm=dict(H_CONT=2,MOMENT_XY=1,SURFACE_XY=2))
    protected={str(OLD/p):h for p,h in read_json(OLD/'delivery_hashes.json').items()}
    protected[str(OLD/'delivery_hashes.json')]=digest(OLD/'delivery_hashes.json')
    protected.update({str(ROOT/p):h for p,h in read_json(OLD/'registration/source_hashes_before.json').items()})
    protected.update({str(ROOT/p):h for p,h in read_json(OLD/'registration/additional_historical_scorer_hashes.json').items()})
    protected.update({str(ROOT/p):h for p,h in read_json(OLD/'registration/implementation_hashes.json').items()})
    freeze_json(out/'registration/protected_hashes.json',protected)
    freeze_json(out/'registration/resources_initial.json',resources)
    freeze_json(out/'registration/geometry_fixtures.json',read_json(OLD/'registration/geometry_fixtures.json'))
    protocol['geometry']=dict(protocol['geometry'],fixtures_path=str(out/'registration/geometry_fixtures.json'),
        fixtures_sha256=digest(out/'registration/geometry_fixtures.json'))
    freeze_json(out/'registration/old_test_manifest.json',old_manifest)
    freeze_json(out/'protocol.json',protocol)
    (out/'protocol.sha256').write_text(digest(out/'protocol.json')+'  protocol.json\n',encoding='utf8')
    return protocol

def verify(out=OUT):
    out=Path(out)
    assert digest(out/'protocol.json')==(out/'protocol.sha256').read_text().split()[0]
    old=read_json(out/'registration/protected_hashes.json')
    changed=[p for p,h in old.items() if not Path(p).is_file() or digest(p)!=h]
    if changed:raise ValueError('Read-only evidence changed: '+str(changed))
    path=out/'registration/implementation_hashes.json'
    if path.exists():
        for p,h in read_json(path).items():assert digest(ROOT/p)==h,'Frozen revision implementation differs: '+p
    return dict(status='PASS',protected_files=len(old))

def freeze_code(out=OUT):
    files=sorted(p for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('reference'))
    freeze_json(Path(out)/'registration/implementation_hashes.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    diff=[]
    for p in files:
        before=ROOT/'experiments/mf_h1_local0/chair_surface_attention_v1'/p.name
        diff.extend(difflib.unified_diff(before.read_text(encoding='utf8').splitlines(True) if before.exists() else [],
            p.read_text(encoding='utf8').splitlines(True),fromfile=str(before.relative_to(ROOT)) if before.exists() else '/dev/null',
            tofile=p.relative_to(ROOT).as_posix()))
    (Path(out)/'code.diff').write_text(''.join(diff),encoding='utf8')

if __name__=='__main__':print(prepare()['schema'])
