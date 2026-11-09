"""Summarize only 252 actual image-review cells; never read metric scores."""
from __future__ import annotations
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.bench_three_mechanisms_v1 import common as C

REVIEW=C.OUT/'visual_review'
CONTRASTS=(('PLAIN200','PATCH200'),('PLAIN200','LOOP200'),('PLAIN267','LOOP200'),('RL_ZERO','RL16'),
           ('EXPOSURE','PLAIN200'),('EXPOSURE','PLAIN267'),('EXPOSURE','PATCH200'),('EXPOSURE','LOOP200'),('EXPOSURE','RL_ZERO'),('EXPOSURE','RL16'))


def helpers():
    p=C.PRIOR/'control/summarize_visual.py'
    spec=importlib.util.spec_from_file_location('_independent_visual_table_helpers',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.CONTRASTS=CONTRASTS
    return module


def main():
    protocol=C.read(REVIEW/'protocol.json');assignments=C.read(REVIEW/'assignments.json')['assignments']
    ph=C.sha(REVIEW/'protocol.json');axes=protocol['axes'];h=helpers();rows=[];protected={}
    pending=[]
    for a in assignments:
        p=REVIEW/'completed'/(a['assignment_id']+'.json')
        if not p.exists():pending.append(str(p));continue
        doc=C.read(p)
        if doc.get('status')!='COMPLETE':pending.append(str(p));continue
    if pending:return dict(status='PENDING',missing_or_unfinished=pending,score_files_read=0,ratings_generated=0)
    expected={(r['request_id'],r['label'],r['stage']):r for r in protocol['expected_units']}
    if len(expected)!=252:raise ValueError('Frozen252 identity count differs')
    frozen=datetime.fromisoformat(protocol['frozen_at_utc']);seen=set();reviewers={};images=set()
    protected[str(REVIEW/'protocol.json')]=ph
    protected[str(REVIEW/'assignments.json')]=C.sha(REVIEW/'assignments.json')
    for pathkey,hashkey in [('criteria_source_path','criteria_source_sha256'),('axes_source_path','axes_source_sha256')]:
        p=C.resolve(protocol[pathkey]);assert C.sha(p)==protocol[hashkey];protected[str(p)]=C.sha(p)
    manifest=C.OUT/'data/eval_manifest.json'
    if C.sha(manifest)!=protocol['manifest_sha256']:raise ValueError('Frozen panel changed')
    protected[str(manifest)]=C.sha(manifest)
    for a in assignments:
        p=REVIEW/'completed'/(a['assignment_id']+'.json');doc=C.read(p)
        if doc['assignment_id']!=a['assignment_id'] or doc['protocol_sha256']!=ph:raise ValueError('Review assignment/protocol mismatch')
        if doc['reviewer_kind']!='AI_PROXY' or doc['blind'] is not False:raise ValueError('AI_PROXY nonblind scope required')
        if len(doc['rows'])!=84:raise ValueError('Each actual shard requires84 cells')
        protected[str(p)]=C.sha(p)
        for r in doc['rows']:
            key=(r['request_id'],r['label'],r['stage'])
            if key not in expected or key in seen:raise ValueError('Duplicate/unregistered reviewed cell')
            unit=expected[key]
            if unit['review_assignment']!=a['assignment_id'] or r['page_id']!=unit['page_id']:raise ValueError('Wrong review shard/page')
            if r['actually_viewed'] is not True or r['views_and_sections_checked'] is not True or r['tool']!='view_image':raise ValueError('Actual image/sections viewing required')
            if not isinstance(r['view_event_id'],str) or not r['view_event_id'].strip():raise ValueError('Actual view event absent')
            dt=datetime.fromisoformat(r['viewed_at_utc'].replace('Z','+00:00'))
            if dt.tzinfo is None or dt.utcoffset().total_seconds()!=0 or not frozen<=dt<=datetime.now(timezone.utc):raise ValueError('Actual UTC view time invalid')
            reviewer=r.get('reviewer') or doc.get('reviewer')
            if not reviewer:raise ValueError('Actual reviewer missing')
            if r['request_id'] in reviewers and reviewers[r['request_id']]!=reviewer:raise ValueError('Same request split across reviewers')
            reviewers[r['request_id']]=reviewer
            if any(r.get(axis) not in values for axis,values in axes.items()):raise ValueError('Missing/invalid visual axis')
            if not isinstance(r.get('note'),str) or not r['note'].strip():raise ValueError('Concrete observed evidence required')
            image=C.OUT/'gallery'/r['page_id']/'comparison.png';receipt=image.with_suffix('.json');ir=C.read(receipt)
            if r['image_sha256']!=C.sha(image) or ir['image_sha256']!=C.sha(image) or r['image_receipt_sha256']!=C.sha(receipt):raise ValueError('Viewed image changed')
            if [x['label'] for x in ir['identity']['cells']]!=list(C.LABELS):raise ValueError('Image model order differs')
            if ir['identity']['request_id']!=r['request_id'] or ir['identity']['stage']!=r['stage']:raise ValueError('Image request/stage differs')
            for artifact in (image,receipt):protected[str(artifact)]=C.sha(artifact)
            for path,digest in zip(r.get('crop_paths',[]),r.get('crop_sha256',[])):
                if C.sha(path)!=digest:raise ValueError('Viewed crop changed')
                protected[str(C.resolve(path))]=digest
            images.add(r['page_id']);seen.add(key)
            rows.append({**unit,**{axis:r[axis] for axis in axes},h.JOINT:h.joint(r),
                'note':r['note'],'reviewer':reviewer,'reviewer_kind':'AI_PROXY','blind':False,
                'actually_viewed':True,'views_and_sections_checked':True,'tool':'view_image',
                'view_event_id':r['view_event_id'],'viewed_at_utc':r['viewed_at_utc'],
                'image_path':str(image),'image_sha256':C.sha(image),'image_receipt_sha256':C.sha(receipt)})
    if seen!=set(expected) or len(images)!=36:raise ValueError('Incomplete fixed cells/pages')
    rows.sort(key=lambda r:(r['request_index'],protocol['stages'].index(r['stage']),protocol['labels'].index(r['label'])))
    tables=h.build_tables(rows,protocol)
    historical=C.OLD/'visual_review/summary_v1/actual_observations.csv'
    import csv
    with historical.open(encoding='utf-8-sig',newline='') as f:old=list(csv.DictReader(f))
    index={(r['request_id'],r['stage']):r for r in old if r['label']=='EXPOSURE'}
    changes=[]
    for r in rows:
        if r['label']!='EXPOSURE':continue
        prior=index[r['request_id'],r['stage']]
        for axis in [*axes,h.JOINT]:
            ov=h.joint(prior) if axis==h.JOINT else prior[axis]
            changes.append(dict(request_id=r['request_id'],parent_id=r['parent_id'],stage=r['stage'],axis=axis,
                historical_label=ov,current_label=r[axis],changed=ov!=r[axis],historical_overwritten=False))
    tables['baseline_review_differences.csv']=changes
    protected[str(historical)]=C.sha(historical);protected[str(Path(__file__))]=C.sha(__file__)
    dest=REVIEW/'summary_v1';dest.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():
        data=h.csv_bytes(table);p=dest/name
        if p.exists() and p.read_bytes()!=data:raise ValueError('No completed review overwrite')
        if not p.exists():p.write_bytes(data)
    C.verify(protected)
    result=dict(status='COMPLETE',cells=252,pages=36,requests=12,parents=6,reviewer_kind='AI_PROXY',blind=False,
        joint_definition=h.JOINT_NOTE,score_files_read=0,labels_inferred_from_metrics=0,ratings_generated=0,
        historical_labels_overwritten=0,comparison_pairs=[list(x) for x in CONTRASTS],protected=protected,
        files={str(dest/name):C.sha(dest/name) for name in tables})
    C.write(dest/'completion.json',result)
    return {k:v for k,v in result.items() if k not in ('protected','files')}


if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))
