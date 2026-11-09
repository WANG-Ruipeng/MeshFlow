"""CPU-only descriptive aggregation of actual saved AI_PROXY review labels.

No scoring, model import, image interpretation, new labels, significance testing,
or source-GT matching. Missing/invalid records are explicit and never imputed.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from . import common as C

VERSION = 'full_seat_actual_review_summary_v1'
MAIN = ('FULL_S0000','FULL_S1000','FULL_S3000','FULL_S6000')
REFERENCES = ('SMALL_C200','CREF_C200')
STEPS = dict(zip(MAIN,(0,1000,3000,6000)))
STAGES = ('RAW','POST')
VALUES = ('YES','NO','UNCERTAIN')
FIELDS = ('C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer',
          'whole_body_complete','support_missing','fragments_or_folded_layers','visible_join')
SOURCE = C.OUT/'evaluation/review/AI_PROXY_reviews.csv'
REF_SOURCE = C.OUT/'evaluation/reviews/reference_AI_PROXY_inherited.csv'
DEST = C.OUT/'evaluation/review/summary'


def csvread(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def csvwrite(path, rows, fields=('status',)):
    if rows: return C.csvwrite(path,rows)
    text=io.StringIO(newline=''); csv.DictWriter(text,fieldnames=fields).writeheader()
    return C._write_bytes(path,text.getvalue().encode('utf-8-sig'))


def capture_reviews(path, labels, requests, sources, image_checks, inherited=False):
    exists=path.is_file(); rows=csvread(path) if exists else []
    if exists: sources[str(path)]=C.sha(path)
    expected={(label,rid,stage) for label in labels for rid in requests for stage in STAGES}
    identities=[(r.get('label'),r.get('request_id'),r.get('stage')) for r in rows]
    duplicates={key for key,n in Counter(identities).items() if n>1}
    accepted, audit, preserved = {}, [], []
    image_cache={}
    def check_image(path_value, expected_sha, row_errors, kind):
        if not path_value or not expected_sha:
            row_errors.append(kind+':MISSING_PATH_OR_SHA256'); return
        p=C.resolve(path_value)
        if not p.is_file():
            row_errors.append(kind+':MISSING_IMAGE'); return
        key=str(p)
        if key not in image_cache:
            image_cache[key]=C.sha(p)
            image_checks.append(dict(path=key,sha256=image_cache[key],check='ACTUAL_FILE_SHA256',kind=kind,source_csv=str(path)))
        actual=image_cache[key]
        sources[key]=actual
        if actual!=expected_sha: row_errors.append(kind+':SHA256_MISMATCH')
    for line,(row,key) in enumerate(zip(rows,identities),2):
        errors=[]; q=requests.get(row.get('request_id'))
        if key not in expected: errors.append('UNREGISTERED_IDENTITY')
        if key in duplicates: errors.append('DUPLICATE_IDENTITY_ALL_COPIES_EXCLUDED')
        if q:
            if row.get('parent_id')!=q['parent_id']: errors.append('PARENT_ID_DIFFERS')
            try:
                if int(row.get('candidate',''))!=int(q['candidate']): errors.append('NOISE_CANDIDATE_DIFFERS')
            except ValueError: errors.append('NOISE_CANDIDATE_MISSING_OR_INVALID')
            for name in ('N','K'):
                if row.get(name):
                    try:
                        if int(row[name])!=int(q[name]): errors.append(name+'_DIFFERS')
                    except ValueError: errors.append(name+'_INVALID')
            if row.get('split') and row['split']!=q['split']: errors.append('TRAIN_DEV_SPLIT_DIFFERS')
        for field in FIELDS:
            if row.get(field) not in VALUES: errors.append(field+':INVALID_OR_MISSING_LABEL')
        if row.get('reviewer_kind')!='AI_PROXY': errors.append('NOT_AI_PROXY')
        if not str(row.get('reviewer','')).strip(): errors.append('NO_REVIEWER_IDENTITY')
        if row.get('review_status') not in ('REVIEWED','COMPLETE'): errors.append('REVIEW_NOT_COMPLETED')
        if str(row.get('blinded','')).lower() not in ('false','0'): errors.append('NONBLIND_STATUS_MISSING_OR_DIFFERENT')
        if not str(row.get('notes','')).strip(): errors.append('OBSERVATION_NOTES_MISSING')
        if inherited:
            if row.get('inheritance_status')!='VERIFIED_HISTORICAL_REVIEW_INHERITANCE': errors.append('INHERITANCE_NOT_VERIFIED')
            if str(row.get('new_visual_review','')).lower() not in ('false','0'): errors.append('INHERITED_MARKED_NEW_REVIEW')
            check_image(row.get('image_path'),row.get('image_sha256'),errors,'INHERITED_ORIGINAL_IMAGE')
            check_image(row.get('depth_image_path'),row.get('depth_image_sha256'),errors,'INHERITED_DEPTH_IMAGE')
        else:
            check_image(row.get('image_path'),row.get('image_sha256'),errors,'ACTUAL_DEPTH_IMAGE')
            check_image(row.get('section_image_path'),row.get('section_image_sha256'),errors,'ACTUAL_FINITE_SECTION_IMAGE')
        status='VALID_INHERITED' if inherited and not errors else 'VALID_ACTUAL' if not errors else 'INVALID'
        audit.append(dict(source_csv=str(path),source_csv_line=line,label=row.get('label'),request_id=row.get('request_id'),stage=row.get('stage'),status=status,reasons=errors))
        preserved.append(dict(row,summary_validation_status=status,summary_validation_reasons=errors,summary_evidence_kind='INHERITED' if inherited else 'ACTUAL_NEW_REVIEW'))
        if not errors: accepted[key]=row
    missing=sorted(expected-set(accepted))
    invalid=sum(r['status']=='INVALID' for r in audit)
    status='INVALID_INPUTS' if invalid else 'COMPLETE' if set(accepted)==expected and len(rows)==len(expected) else 'PARTIAL'
    details=dict(status=status,source_path=str(path),source_exists=exists,observed_csv_rows=len(rows),validated_rows=len(accepted),expected_rows=len(expected),invalid_rows=invalid,
                 missing_identities=[dict(label=a,request_id=b,stage=c) for a,b,c in missing],evidence_kind='INHERITED' if inherited else 'ACTUAL_NEW_REVIEW')
    return accepted,preserved,audit,details


def counted(rows, expected):
    counts=Counter(rows)
    return dict(YES=counts['YES'],NO=counts['NO'],UNCERTAIN=counts['UNCERTAIN'],observed=len(rows),expected=expected,missing=expected-len(rows),
                YES_fraction_observed=counts['YES']/len(rows) if rows else None,
                NO_fraction_observed=counts['NO']/len(rows) if rows else None,
                UNCERTAIN_fraction_observed=counts['UNCERTAIN']/len(rows) if rows else None,
                status='COMPLETE' if len(rows)==expected else 'PARTIAL')


def label_counts(available, labels, requests, inherited=False):
    parents=list(dict.fromkeys(q['parent_id'] for q in requests.values())); parent_rows=[]; strata_rows=[]
    for label in labels:
        for stage in STAGES:
            for parent in parents:
                qs=[q for q in requests.values() if q['parent_id']==parent]
                for field in FIELDS:
                    values=[available[label,q['request_id'],stage][field] for q in qs if (label,q['request_id'],stage) in available]
                    parent_rows.append(dict(label=label,local_step=STEPS.get(label),parent_id=parent,stratum=qs[0]['split'],stage=stage,field=field,
                        evidence_kind='INHERITED' if inherited else 'ACTUAL_NEW_REVIEW',**counted(values,len(qs))))
            for stratum in ('TRAIN','DEV','ALL'):
                qs=[q for q in requests.values() if stratum=='ALL' or q['split']==stratum]
                for field in FIELDS:
                    values=[available[label,q['request_id'],stage][field] for q in qs if (label,q['request_id'],stage) in available]
                    strata_rows.append(dict(label=label,local_step=STEPS.get(label),stratum=stratum,stage=stage,field=field,
                        expected_parents=len({q['parent_id'] for q in qs}),noises_per_parent=4,
                        evidence_kind='INHERITED' if inherited else 'ACTUAL_NEW_REVIEW',**counted(values,len(qs))))
    return parent_rows,strata_rows


def paired(available, requests):
    rows=[]
    for label in MAIN[1:]:
        for q in requests.values():
            for stage in STAGES:
                a=available.get((MAIN[0],q['request_id'],stage)); b=available.get((label,q['request_id'],stage))
                for field in FIELDS:
                    rows.append(dict(label=label,local_step=STEPS[label],baseline=MAIN[0],request_id=q['request_id'],parent_id=q['parent_id'],candidate=q['candidate'],stratum=q['split'],stage=stage,field=field,
                        baseline_value=a[field] if a else None,current_value=b[field] if b else None,
                        transition=(a[field]+'->'+b[field]) if a and b else None,baseline_observed=a is not None,current_observed=b is not None,
                        changed=a[field]!=b[field] if a and b else None,status='COMPLETE' if a and b else 'PARTIAL',
                        baseline_reviewer=a['reviewer'] if a else None,current_reviewer=b['reviewer'] if b else None,
                        evidence_kind='ACTUAL_NEW_REVIEW'))
    groups=defaultdict(list); parent_groups=defaultdict(list)
    for r in rows:
        parent_groups[(r['label'],r['stage'],r['field'],r['parent_id'])].append(r)
        for stratum in (r['stratum'],'ALL'): groups[(r['label'],r['stage'],r['field'],stratum)].append(r)
    def tables(grouped,parent=False):
        output=[]
        for key,values in grouped.items():
            label,stage,field,identity=key
            usable=[r for r in values if r['status']=='COMPLETE']
            counts=Counter((r['baseline_value'],r['current_value']) for r in usable)
            for a in VALUES:
                for b in VALUES:
                    row=dict(label=label,local_step=STEPS[label],baseline=MAIN[0],stage=stage,field=field,baseline_value=a,current_value=b,transition=a+'->'+b,
                        count=counts[a,b],observed_pairs=len(usable),expected_pairs=len(values),missing_pairs=len(values)-len(usable),
                        fraction_observed=counts[a,b]/len(usable) if usable else None,status='COMPLETE' if len(usable)==len(values) else 'PARTIAL')
                    if parent: row.update(parent_id=identity,stratum=values[0]['stratum'])
                    else: row['stratum']=identity
                    output.append(row)
        return output
    return rows,tables(parent_groups,True),tables(groups)


def trajectories(available, requests):
    output=[]
    for q in requests.values():
        for stage in STAGES:
            row={key:q[key] for key in ('request_id','parent_id','candidate','split','N','K')}; row['stage']=stage
            observed=0
            for label in MAIN:
                source=available.get((label,q['request_id'],stage)); observed+=source is not None
                for field in (*FIELDS,'notes','reviewer','image_path','image_sha256','section_image_path','section_image_sha256'):
                    row[label+'__'+field]=source.get(field) if source else None
                row[label+'__review_present']=source is not None
            row.update(observed_checkpoints=observed,expected_checkpoints=4,status='COMPLETE' if observed==4 else 'PARTIAL')
            output.append(row)
    return output


def run():
    registration_path=C.OUT/'registration.json'
    reg=C.read(registration_path); requests={q['request_id']:q for q in reg['requests']}
    if len(requests)!=24 or len({q['parent_id'] for q in requests.values()})!=6 or tuple(reg['evaluation_labels'])!=MAIN:
        raise ValueError('Expected frozen four-checkpoint/six-parent/24-request panel')
    for parent in {q['parent_id'] for q in requests.values()}:
        qs=[q for q in requests.values() if q['parent_id']==parent]
        if sorted(int(q['candidate']) for q in qs)!=[0,1,2,3] or len({q['split'] for q in qs})!=1:
            raise ValueError('Fixed per-parent noise or TRAIN/DEV identity differs')
    sources={str(registration_path):C.sha(registration_path),str(C.CODE/'review_summary.py'):C.sha(C.CODE/'review_summary.py')}; checks=[]
    actual,main_rows,main_audit,main_info=capture_reviews(SOURCE,MAIN,requests,sources,checks)
    ref,ref_rows,ref_audit,ref_info=capture_reviews(REF_SOURCE,REFERENCES,requests,sources,checks,True)
    signature=hashlib.sha256(json.dumps(dict(sources=sources,main_exists=SOURCE.exists(),ref_exists=REF_SOURCE.exists()),sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
    out=DEST/('snapshot_'+signature); final=out/'summary.json'
    if final.is_file():
        value=C.read(final)
        for p,h in value['output_hashes'].items():
            if C.sha(p)!=h: raise ValueError('Saved review summary changed: '+p)
        C.write(DEST/'latest.json',dict(path=str(final),sha256=C.sha(final),status=value['status']),replace=True)
        return value
    parents,strata=label_counts(actual,MAIN,requests)
    refparents,refstrata=label_counts(ref,REFERENCES,requests,True)
    transitions,parent_transitions,strata_transitions=paired(actual,requests)
    tables={'main_review_rows.csv':main_rows,'main_validation.csv':main_audit,
            'main_missing_identities.csv':main_info['missing_identities'],
            'main_label_counts_per_parent.csv':parents,'main_label_counts_strata.csv':strata,
            'main_step0_paired_transitions.csv':transitions,'main_transitions_per_parent.csv':parent_transitions,
            'main_transitions_strata.csv':strata_transitions,'main_request_checkpoint_observations.csv':trajectories(actual,requests),
            'reference_INHERITED_review_rows.csv':ref_rows,'reference_INHERITED_validation.csv':ref_audit,
            'reference_INHERITED_missing_identities.csv':ref_info['missing_identities'],
            'reference_INHERITED_label_counts_per_parent.csv':refparents,'reference_INHERITED_label_counts_strata.csv':refstrata,
            'actual_image_hash_checks.csv':checks}
    for name,rows in tables.items(): csvwrite(out/name,rows)
    # Current source bytes must still match the captured immutable snapshot.
    for p,h in sources.items():
        if C.sha(p)!=h: raise ValueError('Review input changed during summary: '+p)
    summary=dict(status=main_info['status'],version=VERSION,main=main_info,reference_INHERITED=ref_info,
        reference_not_in_main_counts=True,main_labels=list(MAIN),main_steps=[0,1000,3000,6000],stages=list(STAGES),label_values=list(VALUES),judgment_fields=list(FIELDS),
        rows_per_table={name:len(rows) for name,rows in tables.items()},source_hashes=sources,unique_checked_image_files=len({r['path'] for r in checks}),
        source_registration_sha256=sources[str(registration_path)],
        limitations=['Descriptive fixed-panel AI_PROXY counts, not human ratings or independent replicated trials.',
            'Three-state labels remain distinct. No absent review is filled as NO or UNCERTAIN.',
            'Transition matrices preserve all nine YES/NO/UNCERTAIN combinations; no criterion-independent better/worse ordering is assumed.',
            'TRAIN, DEV and ALL are separate. Four fixed noises within each studied parent do not establish new-parent generalization.',
            'Raw observations and notes are copied; this script does not inspect metric scores, assign labels, revise reviewers, or add image reviews.',
            'A verified image hash binds an existing observation to a file; it does not independently certify the observation.'],
        new_image_reviews=0,new_labels=0,new_quality_scans=0,new_model_calls=0,additional_updates=0,
        output_hashes={str(out/name):C.sha(out/name) for name in tables})
    C.write(final,summary); C.write(DEST/'latest.json',dict(path=str(final),sha256=C.sha(final),status=summary['status']),replace=True)
    return summary


if __name__=='__main__':
    result=run()
    print(json.dumps(dict(status=result['status'],main_validated=result['main']['validated_rows'],main_expected=192,
        main_invalid=result['main']['invalid_rows'],inherited_validated=result['reference_INHERITED']['validated_rows'],inherited_status=result['reference_INHERITED']['status'],
        unique_checked_image_files=result['unique_checked_image_files'],new_image_reviews=0,new_labels=0),sort_keys=True))