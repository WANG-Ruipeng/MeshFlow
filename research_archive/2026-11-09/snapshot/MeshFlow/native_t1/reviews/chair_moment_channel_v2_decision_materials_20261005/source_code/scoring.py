"""CPU-only V2 score adapter; old scoring modules and globals remain unchanged."""
from pathlib import Path
import argparse
import copy
import importlib.util
from collections import Counter
from experiments.mf_h1_local0.chair_surface_attention_v1 import scoring as original

ROOT=Path(__file__).resolve().parents[3]
CHAIR=original.CHAIR
HYBRID=original.HYBRID
SURFACE=ROOT/'native_t1/runs/chair_surface_attention_v1'
OUT=ROOT/'native_t1/runs/chair_moment_channel_v2'
LABELS=('START','H_CONT_END','MOMENT_XY_END')
ARMS=('H_CONT','MOMENT_XY')
START_SHA=original.START_SHA
read_json,write_json,write_csv,sha=original.read_json,original.write_json,original.write_csv,original.sha
records=original.records
_spec=importlib.util.spec_from_file_location('experiments.mf_h1_local0.chair_surface_attention_v1._moment_v2_scoring',original.__file__)
helper=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(helper)
helper.LABELS=LABELS;helper.ARMS=ARMS
DEFINITION=dict(original.DEFINITION,experiment='NATIVE_CHAIR_MOMENT_CHANNEL_LEARNING_V2',
    main_comparisons=['MOMENT_XY_END-H_CONT_END','H_CONT_END-START','MOMENT_XY_END-START'],
    strict_validation_columns='4 parents x 2 noises x 3 target columns =24; 12 per noise; not independent parents',
    validation_curves='4 parents x 2 noises =8; 4 per noise',
    I_P_diagnostic='NOT_RUN_BY_PROTOCOL',GT_angle_abs_gap='Descriptive abs(predicted small-angle fraction - same GT fraction), not a mesh-quality certificate')
DEFINITION.pop('trajectory_quadrature',None);helper.DEFINITION=DEFINITION


def missing_width(label,registered,reason):
    tables={k:[] for k in ('width_per_output','width_per_curve','cross_matrix','target_column_audit')}
    groups={(r['uid'],r['noise_index']):r for r in registered if r['table']=='B'}
    for (uid,noise),rec in sorted(groups.items()):
        for candidate in ('MODEL','No-response','Post-scale'):
            base=dict(checkpoint_label=label,uid=uid,noise_index=noise,split=rec['split'],candidate=candidate,status='BLOCKED',reason=reason)
            tables['width_per_curve'].append(dict(base))
            for alpha in (.9,1.,1.1):
                tables['width_per_output'].append(dict(base,alpha=alpha))
                tables['target_column_audit'].append(dict(base,target_beta=alpha,A_column=None,column_strict_win=None,diagonal_RMS_L0=None))
                for beta in (.9,1.,1.1):tables['cross_matrix'].append(dict(base,alpha=alpha,target_beta=beta,E_RMS_L0=None))
    return tables


def _reuse_width_cache(out):
    oldfolder=SURFACE/'evaluation/START/scoring'
    summary=read_json(oldfolder/'summary.json')
    oldsource=str(Path(original.old.__file__).relative_to(ROOT))
    if summary['source_adapter_sha256'][oldsource]!=sha(original.old.__file__):raise ValueError('Historical width scoring source changed')
    folder=Path(out)/'evaluation/START/scoring/width_groups'
    for src in sorted((oldfolder/'width_groups').glob('*.json')):
        dst=folder/src.name;saved=read_json(src)
        if dst.exists():
            if read_json(dst)!=saved:raise ValueError('START width cache differs')
        else:write_json(dst,saved)
    # The inherited scorer checks every cached raw-array/input/query identity.


def run_checkpoint(out,checkpoint_label,records,generated_rows,source_out=CHAIR,start_score_root=HYBRID):
    if checkpoint_label not in LABELS:raise ValueError('Unregistered V2 checkpoint label')
    out=Path(out).resolve();source_out=Path(source_out).resolve()
    if out in (source_out,HYBRID.resolve(),SURFACE.resolve()):raise ValueError('Output must be isolated')
    if checkpoint_label=='START':_reuse_width_cache(out)
    summary=helper.run_checkpoint(out,checkpoint_label,records,generated_rows,source_out,start_score_root)
    folder=out/'evaluation'/checkpoint_label/'scoring'
    rows=read_json(folder/'per_output.json')
    for row in rows:
        if row['status']=='PASS':row['small_angle_abs_gap_to_same_GT']=abs(row['small_angle_excess_over_same_GT'])
    write_json(folder/'per_output.json',rows);write_csv(folder/'per_output.csv',rows)
    defaults=missing_width(checkpoint_label,records,'Incomplete original registered alpha triplet')
    key_fields={'width_per_output':('uid','noise_index','candidate','alpha'),
        'width_per_curve':('uid','noise_index','candidate'),
        'cross_matrix':('uid','noise_index','candidate','alpha','target_beta'),
        'target_column_audit':('uid','noise_index','candidate','target_beta')}
    for name,fields in key_fields.items():
        values=read_json(folder/(name+'.json'));index={tuple(v[k] for k in fields):v for v in values}
        if len(index)!=len(values):raise ValueError('Duplicate registered width row')
        full=[index.get(tuple(v[k] for k in fields),v) for v in defaults[name]]
        write_json(folder/(name+'.json'),full);write_csv(folder/(name+'.csv'),full);summary['row_counts'][name]=len(full)
    summary.update(definition=DEFINITION,availability=dict(Counter(r['status'] for r in rows)),
        source_adapter_sha256={str(Path(original.__file__).relative_to(ROOT)):sha(original.__file__),
            str(Path(original.old.__file__).relative_to(ROOT)):sha(original.old.__file__)})
    write_json(folder/'summary.json',summary);return summary


def cpu_tests(out):
    result=helper.cpu_tests(out)
    result.update(experiment='NATIVE_CHAIR_MOMENT_CHANNEL_LEARNING_V2',labels=list(LABELS),old_module_globals_unchanged=original.LABELS==('START','H_CONT','MOMENT_XY','SURFACE_XY'))
    if not result['old_module_globals_unchanged']:raise AssertionError('Protected module globals were changed')
    write_json(Path(out)/'tests/v2_scoring_cpu_tests.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--label',choices=LABELS)
    p.add_argument('--source',default=str(CHAIR));p.add_argument('--generated');p.add_argument('--records');p.add_argument('--cpu-tests',action='store_true')
    a=p.parse_args()
    if a.cpu_tests:print(cpu_tests(a.out)['status'])
    else:
        generated=read_json(a.generated);generated=generated if isinstance(generated,list) else generated['rows']
        registered=read_json(a.records) if a.records else records(a.source)
        if isinstance(registered,dict):registered=registered.get('records',registered.get('rows'))
        print(run_checkpoint(a.out,a.label,registered,generated,a.source)['status'])
