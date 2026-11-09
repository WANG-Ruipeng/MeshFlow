"""Correct exposure review metadata without editing the frozen renderer.

The original renderer body and every image operation remain unchanged. Its
prepare() hook is replaced only while invoking render() through this adapter.
"""
from __future__ import annotations
import argparse,datetime,json,types
from . import report_results as R

E=R.E
SOURCE=E.ROOT/'experiments/r2_bench_group_weight_match_v1/visual_review/v2_definition.json'


def prepare():
    manifest=E.read(R.GALLERY/'manifest.json');E.verify(manifest['protected'])
    assert manifest['labels']==['M','EXPOSURE'] and manifest['cells']==276
    path=E.OUT/'visual_review/v2_definition.json'
    if not path.exists():
        definition=E.read(SOURCE)
        for key in ('B','before_new_M_sampling'):definition.pop(key,None)
        definition.update(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            main_comparison=['M','EXPOSURE'],
            labels_description={'M':'Frozen W_MATCHED1500 Native endpoint',
                                'EXPOSURE':'Same Native recipe after 3000 further updates'},
            inherited_definition_path=str(SOURCE),inherited_definition_sha256=E.sha(SOURCE),
            anchor_scope='Historical interpretation examples only; zero current-experiment observations',
            current_experiment_ratings_copied=0)
        definition['rules']=[s.replace('W/M page','M/EXPOSURE page') for s in definition['rules']]
        E.immutable(path,definition)
    definition=E.read(path)
    assert definition['main_comparison']==['M','EXPOSURE'] and definition['inherited_definition_sha256']==E.sha(SOURCE)
    protocol=R.protocol_template()
    assert protocol['labels']==['M','EXPOSURE']
    protocol.update(interpretation_version=2,definition=definition,definition_sha256=E.sha(path),
                    no_historical_label_pooling=True,quantitative_only_labels=[])
    E.immutable(E.OUT/'visual_review/protocol.json',protocol)
    E.immutable(E.OUT/'visual_review/metadata_correction.json',dict(
        status='APPLIED_METADATA_ONLY',frozen_report_path=R.__file__,frozen_report_sha256=E.sha(R.__file__),
        gallery_manifest_sha256=E.sha(R.GALLERY/'manifest.json'),
        old_prepare_field={'quantitative_only_labels':['B']},corrected_field={'quantitative_only_labels':[]},
        actual_labels=['M','EXPOSURE'],
        method='Call frozen protocol_template directly; do not edit report_results.py or gallery manifest',
        render_adapter='Same original render body, corrected prepare hook and index HTML title only',
        protocol_sha256=E.sha(E.OUT/'visual_review/protocol.json'),adapter_sha256=E.sha(__file__),
        model_calls=0,scoring_calls=0,images_created=0,ratings_copied=0))
    return dict(status='FROZEN_V2',labels=['M','EXPOSURE'],sheets=138,cells=276,
                quantitative_only_labels=[],model_calls=0,scoring_calls=0,ratings_generated=0)


def corrected_text_file(path,value):
    if E.resolve(path)==R.GALLERY/'index.html':
        value=value.replace('Bench group-weight match: W original / M matched',
                            'Bench additional exposure: M frozen / EXPOSURE +3000')
        value=value.replace('B is quantitative-only.','Both displayed models receive a fresh complete V2 review.')
    return R.text_file(path,value)


def render(workers):
    prepare()
    namespace=dict(vars(R),prepare=prepare,text_file=corrected_text_file)
    function=types.FunctionType(R.render.__code__,namespace,R.render.__name__,R.render.__defaults__,R.render.__closure__)
    function.__kwdefaults__=R.render.__kwdefaults__
    return function(workers)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--render',action='store_true');p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();result=render(a.workers) if a.render else prepare()
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','images')},ensure_ascii=False))
