"""Describe existing condition-quality labels weighted by real exposure counts.

Uses only three MINI CSV tables. No original logs, mesh arrays or scorers.
Every output count/ratio is DERIVED; it is not gradient or causal attribution.
Run from any extracted directory: python protocol_and_code/summarize_quality_exposure.py --out /path/to/new-results
Optionally pass --input-dir when the CSVs are elsewhere; no fixed package basename is required.
"""
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
import argparse
import csv
import datetime
import hashlib
import io
import json

DEFAULT_INPUT=Path(__file__).resolve().parents[1]
TABLES=('conditions.csv','source_quality_and_interfaces.csv','exposure_by_condition.csv')
COUNTS=('actual_exposures','clean_exposures','width_exposures','cumulative_free_scalars',
        'clean_free_scalars','width_free_scalars')
ARMS=('USER_SMALL','USER_EXPANDED_67')
ROLES=('COMMON_SMALL32','EXP_ONLY35')


def sha(blob):return hashlib.sha256(blob).hexdigest()


def load(name, data_root):
    path=data_root/name
    if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Symlink forbidden')
    blob=path.read_bytes()
    reader=csv.DictReader(io.StringIO(blob.decode('utf-8-sig'),newline=''))
    return list(reader),dict(file=name,sha256=sha(blob),bytes=len(blob))


def table(rows,name):
    result={}
    for row in rows:
        key=(row['parent_uid'],row['condition_id'])
        if key in result:raise ValueError('Duplicate condition identity in '+name+': '+str(key))
        result[key]=row
    return result


def ratio(a,b):return str(Decimal(a)/Decimal(b)) if b else 'NA'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=DEFAULT_INPUT,
                        help='Directory containing the three MINI CSV inputs (default: parent of protocol_and_code)')
    parser.add_argument('--out',type=Path,required=True,
                        help='Independent output directory; existing result filenames are never overwritten')
    args=parser.parse_args()
    data_root=args.input_dir.expanduser().absolute()
    folder=args.out.expanduser().absolute()
    for path in (data_root,folder):
        if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Symlink forbidden')
    data={};sources=[]
    for name in TABLES:
        data[name],meta=load(name,data_root);meta['rows']=len(data[name]);sources.append(meta)
    conditions=table(data['conditions.csv'],'conditions')
    quality=table(data['source_quality_and_interfaces.csv'],'source quality')
    exposure=data['exposure_by_condition.csv'];seen=set();all_groups={};joined=[];missing=[]
    cohort_den=defaultdict(lambda:dict.fromkeys(COUNTS,0))
    arm_den=defaultdict(lambda:dict.fromkeys(COUNTS,0))
    counts=defaultdict(lambda:dict.fromkeys(COUNTS,0));cond_ids=defaultdict(set);parents=defaultdict(set)
    metadata=defaultdict(lambda:{'origins':set(),'rules':set()})
    for row in exposure:
        arm,parent,cid,role=(row[k] for k in ('arm','parent_uid','condition_id','parent_role'))
        key=(parent,cid);ek=(arm,parent,cid)
        if ek in seen:raise ValueError('Duplicate real exposure group')
        seen.add(ek)
        if arm not in ARMS or role not in ROLES:raise ValueError('Unexpected arm/role in training exposure')
        if arm=='USER_SMALL' and role!='COMMON_SMALL32':raise ValueError('Expanded-only object in SMALL')
        if key not in conditions:
            missing.append(dict(arm=arm,parent_uid=parent,condition_id=cid,field='conditions identity',reason='NOT_PRESENT'))
        else:
            c=conditions[key]
            if c.get('parent_role')!=role:raise ValueError('Condition/exposure parent role differs')
            for field in ('N','K'):
                if c.get(field) not in ('',None,'NA','NULL') and int(c[field])!=int(row[field]):
                    raise ValueError('Condition/exposure N or K differs')
        q=quality.get(key)
        label=q.get('source_quality') if q else None
        if label in (None,'','NA','NULL'):
            missing.append(dict(arm=arm,parent_uid=parent,condition_id=cid,field='source_quality',reason='NOT_RECORDED'))
            label='NA_UNLABELED'
        if q is not None and q.get('parent_role') not in (None,'',role):
            raise ValueError('Quality/exposure role differs')
        g=(arm,role,label);nums={name:int(row[name]) for name in COUNTS}
        if nums['actual_exposures']!=nums['clean_exposures']+nums['width_exposures']:
            raise ValueError('Actual exposure clean/width counts differ')
        if nums['cumulative_free_scalars']!=nums['clean_free_scalars']+nums['width_free_scalars']:
            raise ValueError('Actual free scalar clean/width counts differ')
        for name in COUNTS:
            counts[g][name]+=nums[name];cohort_den[(arm,role)][name]+=nums[name];arm_den[arm][name]+=nums[name]
        cond_ids[g].add(cid);parents[g].add(parent)
        metadata[g]['origins'].add(q.get('source_quality_label_origin','NA_NOT_RECORDED') if q else 'NA_NOT_RECORDED')
        metadata[g]['rules'].add(q.get('source_quality_parent_rule','NA_NOT_RECORDED') if q else 'NA_NOT_RECORDED')
        joined.append(dict(arm=arm,parent_uid=parent,condition_id=cid,parent_role=role,source_quality=label,
                           actual_exposures=nums['actual_exposures']))
    rows=[]
    for arm,role,label in sorted(counts,key=lambda k:(ARMS.index(k[0]),ROLES.index(k[1]),k[2])):
        g=(arm,role,label);den=cohort_den[(arm,role)];total=arm_den[arm];v=counts[g]
        rows.append(dict(arm=arm,parent_role=role,source_quality=label,condition_count=len(cond_ids[g]),
            parent_count_nonadditive=len(parents[g]),**v,
            cohort_total_exposures=den['actual_exposures'],arm_total_exposures=total['actual_exposures'],
            cohort_total_free_scalars=den['cumulative_free_scalars'],arm_total_free_scalars=total['cumulative_free_scalars'],
            exposure_fraction_within_cohort=ratio(v['actual_exposures'],den['actual_exposures']),
            exposure_fraction_of_arm=ratio(v['actual_exposures'],total['actual_exposures']),
            clean_fraction_within_cohort_clean=ratio(v['clean_exposures'],den['clean_exposures']),
            width_fraction_within_cohort_width=ratio(v['width_exposures'],den['width_exposures']),
            free_scalar_fraction_within_cohort=ratio(v['cumulative_free_scalars'],den['cumulative_free_scalars']),
            free_scalar_fraction_of_arm=ratio(v['cumulative_free_scalars'],total['cumulative_free_scalars']),
            source_quality_label_origin=' | '.join(sorted(metadata[g]['origins'])),
            source_quality_parent_rule=' | '.join(sorted(metadata[g]['rules'])),
            status='DERIVED_EXISTING_LABEL_EXPOSURE_ONLY',
            interpretation='Frequency/free-scalar description, not gradient contribution or causal evidence'))
    for arm in ARMS:
        if arm_den[arm]['actual_exposures']!=48000:raise ValueError('Unexpected actual per-arm exposure total')
    if len(seen)!=285:raise ValueError('Expected all 285 actual arm-condition groups')
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/'training_quality_exposure.csv';provenance=folder/'exposure_quality_provenance.json'
    if target.exists() or provenance.exists():raise FileExistsError('Preserve existing derived output')
    fields=list(rows[0]);buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=fields)
    writer.writeheader();writer.writerows(rows);blob=buffer.getvalue().encode('utf-8')
    for meta in sources:
        if sha((data_root/meta['file']).read_bytes())!=meta['sha256']:raise ValueError('Package input changed during read')
    target.write_bytes(blob)
    report=dict(schema='mini_quality_exposure_v1',status='DERIVED_COMPLETE' if not missing else 'DERIVED_WITH_MISSING_LABELS',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=sources,
        output=dict(file='training_quality_exposure.csv',sha256=sha(blob),rows=len(rows),bytes=len(blob),fields=fields),
        script='protocol_and_code/summarize_quality_exposure.py',script_sha256=sha(Path(__file__).read_bytes()),
        derivation='Join parent_uid+condition_id; sum already-extracted actual exposure counts and free scalars by arm, cohort and original source_quality label; decimal ratios over explicit observed denominators.',
        inherited_label_scope='Original condition manifest FF-risk label; NO_DETECTED_PAIR_PROBLEM is not a clean-mesh certificate; UNKNOWN and missing labels remain separate.',
        parent_count_caveat='A parent can have conditions in multiple source-quality labels; parent_count_nonadditive must not be summed across labels.',
        input_directory=str(data_root),output_directory=str(folder),portable=True,
        reproduce='python protocol_and_code/summarize_quality_exposure.py --out <new-output-directory>',
        only_package_csv_inputs=True,original_logs_reread=0,model_calls=0,new_scoring_calls=0,old_experiment_writes=0,
        missing=missing,checks=dict(joined_exposure_groups=len(seen),actual_exposures_by_arm={a:arm_den[a]['actual_exposures'] for a in ARMS},
                                   all_condition_identities_matched=not any(r['field']=='conditions identity' for r in missing),
                                   original_labels_preserved=True,no_missing_or_UNKNOWN_dropped=True),
        field_precision='Integer sums exact; ratios Decimal default precision 28; all values DERIVED rather than original measurements')
    provenance.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':report['status'],'output':report['output'],'checks':report['checks'],
                      'illegal_label_exposure':[{'arm':r['arm'],'role':r['parent_role'],'actual_exposures':r['actual_exposures'],
                      'cohort_exposures':r['cohort_total_exposures'],'fraction':r['exposure_fraction_within_cohort']} for r in rows if r['source_quality']=='ILLEGAL']},indent=2))


if __name__=='__main__':main()
