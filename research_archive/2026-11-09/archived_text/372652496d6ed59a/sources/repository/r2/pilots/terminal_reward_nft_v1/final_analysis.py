"""Complete-only descriptive final comparison, after anonymous actual review.

No model, fresh geometry sample, GT reconstruction or reward calibration.
All 108 outputs and their 216 original RAW/POST scores must be present. The
primary historical joint event is RAW adoption YES and POST valid CF edge>0;
it does not certify full seams, body attachment, manifoldness or editability.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter,defaultdict
import csv
import io
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np
from . import data as D

VERSION='terminal_reward_nft_final_analysis_v1'
EV=D.OUT/'evaluation'
DEST=EV/'analysis'
LABELS=('BASE','NFT','SELF_FM')
REVIEW_FIELDS=('RAW_adoption','POST_adoption','POST_body_complete')
ALLOWED_REVIEW={'YES','NO','UNKNOWN','UNCERTAIN','NA'}
CONTRASTS=(('NFT','BASE'),('SELF_FM','BASE'),('NFT','SELF_FM'))
CONTINUOUS=('CF_illegal_fraction','FF_illegal_fraction','I_lower','I_upper','free_retained_fraction',
            'legal_CF_edges','legal_CF_length','free_near_degenerate_fraction','free_small_angle_fraction',
            'reached_free_area_fraction','free_surface_area')


def _model_free():
    if 'torch' in sys.modules:raise RuntimeError('Final descriptive analysis must be torch-free')


def _digest_files(files):
    for p,h in files.items():
        if D.sha(p)!=h:raise ValueError('Final input/source changed: '+p)


def _csv(path,rows):
    rows=D.clean(rows)
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    content=buffer.getvalue().encode('utf-8-sig');path=D.checked(path)
    if path.exists():
        if path.read_bytes()!=content:raise ValueError('Refuse changed completed CSV: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(content)


def readiness(reviews_path=None):
    """Read-only availability check; never writes an incomplete success report."""
    reviews_path=D.resolve(reviews_path or EV/'blind_gallery/reviews_actual.json')
    required=[D.OUT/'inputs/manifest.json',EV/'private/label_mapping.json',EV/'blind_gallery/manifest.json',reviews_path]
    if (D.OUT/'inputs/manifest.json').exists():
        m=D.read(D.OUT/'inputs/manifest.json')
        required.extend(EV/label/'scores'/q['request_id']/stage/'score.json'
                        for label in LABELS for q in m['evaluation_requests'] for stage in ('RAW','POST'))
    missing=[str(p) for p in required if not p.exists()]
    return dict(status='READY_FOR_VALIDATION' if not missing else 'INCOMPLETE',missing=missing,
                missing_count=len(missing),reviews_path=str(reviews_path),complete_summary_written=False)


def _review_rows(path):
    if str(path).lower().endswith('.csv'):
        with D.resolve(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    value=D.read(path)
    return value['rows'] if isinstance(value,dict) else value


def _is_true(value):
    return value is True or isinstance(value,str) and value.strip().lower()=='true'


def load_reviews(m,path,protected):
    map_path=EV/'private/label_mapping.json';gallery_path=EV/'blind_gallery/manifest.json'
    mapping=D.read(map_path);gallery=D.read(gallery_path)
    if mapping['status']!='FROZEN' or mapping['manifest_sha256']!=D.sha(D.OUT/'inputs/manifest.json'):
        raise ValueError('Anonymous mapping source differs')
    if gallery['status']!='COMPLETE':raise ValueError('Final gallery incomplete')
    qids={q['request_id'] for q in m['evaluation_requests']}
    maps={r['request_id']:r['aliases'] for r in mapping['rows']}
    images={r['request_id']:r for r in gallery['images']}
    if len(maps)!=36 or set(maps)!=qids or len(images)!=36 or set(images)!=qids:
        raise ValueError('Final36 gallery/mapping request coverage differs')
    for p in (map_path,gallery_path,D.resolve(path)):protected[str(p)]=D.sha(p)
    rows=_review_rows(path)
    if len(rows)!=108:raise ValueError('All108 actual anonymous model/output reviews required')
    result={}
    for row in rows:
        rid,alias=row['request_id'],row['label']
        if rid not in maps or alias not in ('A','B','C'):raise ValueError('Unexpected anonymous review key')
        label=maps[rid][alias];key=(label,rid)
        if key in result:raise ValueError('Duplicate final actual review')
        if row.get('review_status')!='REVIEWED':raise ValueError('Unreviewed final output; do not summarize as success')
        if row.get('reviewer_kind') not in ('AI_PROXY','HUMAN') or not str(row.get('reviewer','')).strip():
            raise ValueError('Actual reviewer identity missing')
        if not _is_true(row.get('blinded')):raise ValueError('Final model-label blinded review must be explicitly identified')
        if any(row.get(k) not in ALLOWED_REVIEW for k in REVIEW_FIELDS):raise ValueError('Unrecognized actual review label')
        im=images[rid]
        if row.get('image_sha256')!=im['image_sha256']:raise ValueError('Review refers to a different final image')
        if D.sha(im['image_path'])!=im['image_sha256']:raise ValueError('Reviewed image changed')
        protected[im['image_path']]=im['image_sha256']
        result[key]=dict(row,anonymous_label=alias,label=label)
    if set(result)!={(label,rid) for label in LABELS for rid in qids}:
        raise ValueError('Final model/output reviews incomplete')
    return result


def _number(value):
    if value is None:return None
    f=float(value)
    return f if math.isfinite(f) else None


def extract_score(score,q,label,stage):
    identity=score['identity']
    for key,expected in (('request_id',q['request_id']),('condition_id',q['condition_id']),('label',label),('stage',stage),('N',q['N']),('K',q['K'])):
        if identity.get(key)!=expected:raise ValueError('Final score identity mismatch: '+key)
    if score['status']!='PASS' or score['stage']!=stage or score['C_preserved_exact'] is not True:
        raise ValueError('Final score or hard C preservation failed; not a complete comparison')
    metrics,topology=score['metrics'],score['topology']
    original=q['N']-q['K'];actual=int(metrics['actual_free_faces'])
    edges=int(topology['C_F_valid_edge_count'])
    values=dict(CF_illegal_fraction=_number(metrics['CF_illegal_affected_free_fraction']),
        FF_illegal_fraction=_number(metrics['FF_illegal_affected_free_fraction']),
        I_lower=_number(metrics['ConflictCov_C']),I_upper=_number(metrics.get('ConflictCov_C_upper')),
        I_status=metrics.get('ConflictCov_C_status','UNKNOWN'),
        free_retained_fraction=float(actual/original),actual_free_faces=actual,original_free_faces=original,
        deleted_free_faces=int(metrics['deleted_free_faces']),legal_CF_edges=edges,
        has_legal_CF=int(edges>0),legal_CF_length=float(topology['C_F_valid_edge_total_length']),
        free_near_degenerate_fraction=_number(metrics.get('free_near_degenerate_fraction')),
        free_small_angle_fraction=_number(metrics.get('free_minimum_angle_lt5_fraction')),
        reached_free_area_fraction=_number(topology.get('reached_free_area_fraction')),
        free_surface_area=_number(metrics.get('free_total_surface_area')),
        CF_unresolved_pairs=int(metrics.get('CF_unresolved_or_degenerate_pair_count',0)),
        FF_unresolved_pairs=int(metrics.get('FF_unresolved_or_degenerate_pair_count',0)),
        topology_status=topology['status'],source_C_connectivity_status=score.get('source_C_connectivity_status','UNKNOWN'))
    values['unavailable_numeric_fields']=[k for k in CONTINUOUS if values[k] is None]
    return values


def _truth(label):
    return True if label=='YES' else False if label=='NO' else None


def _and(values):
    if any(v is False for v in values):return False
    if any(v is None for v in values):return None
    return True


def load_outputs(m,reviews,protected):
    outputs=[];stages=[];meshes={};definitions=None
    selected={r['parent_id'] for r in m['train_conditions']}
    for label in LABELS:
        for q in m['evaluation_requests']:
            rid=q['request_id'];review=reviews[label,rid]
            row={k:q[k] for k in ('request_id','parent_id','condition_id','candidate','split','N','K','evaluation_noise_partition')}
            row.update(label=label,pilot_training_parent=q['parent_id'] in selected,
                anonymous_label=review['anonymous_label'],reviewer_kind=review['reviewer_kind'],reviewer=review['reviewer'],
                review_image_sha256=review['image_sha256'],review_notes=review.get('notes',''))
            for key in REVIEW_FIELDS:row[key]=review[key]
            row.update(RAW_adoption_known=_truth(review['RAW_adoption']),
                       POST_adoption_known=_truth(review['POST_adoption']),
                       POST_body_complete_known=_truth(review['POST_body_complete']))
            for stage in ('RAW','POST'):
                path=EV/label/'scores'/rid/stage/'score.json';score=D.read(path)
                protected[str(path)]=D.sha(path);_digest_files(score['files']);_digest_files(score['identity']['input_hashes'])
                protected.update(score['files']);protected.update(score['identity']['input_hashes'])
                if definitions is None:definitions=score['identity']['definition_hashes']
                elif definitions!=score['identity']['definition_hashes']:raise ValueError('Numerical scorer definitions differ across final outputs')
                values=extract_score(score,q,label,stage)
                row.update({stage+'_'+key:value for key,value in values.items()})
                stages.append(dict(label=label,stage=stage,**{k:row[k] for k in ('request_id','parent_id','condition_id','candidate','split','evaluation_noise_partition')},**values))
                meshpath=score['scoring_geometry_path'];mesh=D.arrays(meshpath)
                triangles=mesh.get('output')
                if triangles is None:triangles=mesh['vertices'][mesh['faces']]
                triangles=np.asarray(triangles)
                if triangles.dtype!=np.float32 or triangles.ndim!=3 or triangles.shape[1:]!=(3,3):
                    raise ValueError('Diversity requires existing unmodified FP32 triangles')
                meshes[label,rid,stage]=np.ascontiguousarray(triangles[q['K']:])
            row['joint_RAW_adoption_POST_legal_CF']=_and([row['RAW_adoption_known'],bool(row['POST_has_legal_CF'])])
            row['joint_both_adoptions_body_and_CF']=_and([row['RAW_adoption_known'],row['POST_adoption_known'],
                                                        row['POST_body_complete_known'],bool(row['POST_has_legal_CF'])])
            outputs.append(row)
    if len(outputs)!=108 or len(stages)!=216:raise ValueError('Final score coverage incomplete')
    return outputs,stages,meshes,definitions


def _stats(values):
    finite=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return dict(count=len(values),finite=len(finite),unavailable=len(values)-len(finite),
        mean=float(np.mean(finite)) if finite else None,median=float(np.median(finite)) if finite else None,
        minimum=min(finite) if finite else None,maximum=max(finite) if finite else None)


def _event(rows,key):
    n=len(rows);yes=sum(r[key] is True for r in rows);no=sum(r[key] is False for r in rows);unknown=n-yes-no
    return dict(total=n,YES=yes,NO=no,UNKNOWN=unknown,observed_positive_fraction_all=yes/n if n else None,
        possible_positive_fraction_if_unknown_yes=(yes+unknown)/n if n else None)


def summarize(outputs):
    result=[];parent_rows=[]
    for label in LABELS:
        for noise in ('ALL','OLD','FRESH'):
            for split in ('ALL','TRAIN','DEV'):
                rows=[r for r in outputs if r['label']==label and (noise=='ALL' or r['evaluation_noise_partition']==noise)
                      and (split=='ALL' or r['split']==split)]
                if not rows:continue
                parents=sorted({r['parent_id'] for r in rows})
                item=dict(label=label,noise_partition=noise,split=split,outputs=len(rows),parents=len(parents),events={},metrics={})
                for key in ('RAW_adoption_known','POST_adoption_known','POST_body_complete_known','joint_RAW_adoption_POST_legal_CF','joint_both_adoptions_body_and_CF'):
                    item['events'][key]=_event(rows,key)
                item['POST_interface_outputs']=sum(r['POST_has_legal_CF'] for r in rows)
                for stage in ('RAW','POST'):
                    item[stage+'_I_status_counts']=dict(Counter(r[stage+'_I_status'] for r in rows))
                    for metric in CONTINUOUS:
                        key=stage+'_'+metric
                        parent_means=[]
                        for parent in parents:
                            values=[r[key] for r in rows if r['parent_id']==parent]
                            parent_means.append(_stats(values)['mean'])
                        item['metrics'][key]=dict(output_distribution=_stats([r[key] for r in rows]),
                                                  equal_parent_distribution=_stats(parent_means))
                result.append(item)
                if split=='ALL':
                    for parent in parents:
                        subset=[r for r in rows if r['parent_id']==parent]
                        p=dict(label=label,parent_id=parent,split=subset[0]['split'],noise_partition=noise,outputs=len(subset),
                               RAW_adoption=_event(subset,'RAW_adoption_known'),joint=_event(subset,'joint_RAW_adoption_POST_legal_CF'),
                               POST_interface_outputs=sum(r['POST_has_legal_CF'] for r in subset))
                        for stage in ('RAW','POST'):
                            for metric in CONTINUOUS:p[stage+'_'+metric]=_stats([r[stage+'_'+metric] for r in subset])['mean']
                        parent_rows.append(p)
    return result,parent_rows


def paired(outputs):
    index={(r['label'],r['request_id']):r for r in outputs};rids=list(dict.fromkeys(r['request_id'] for r in outputs));rows=[]
    for candidate,reference in CONTRASTS:
        for rid in rids:
            a,b=index[candidate,rid],index[reference,rid]
            row=dict(candidate_model=candidate,reference_model=reference,request_id=rid,parent_id=a['parent_id'],split=a['split'],
                     noise_partition=a['evaluation_noise_partition'])
            for key in ('RAW_adoption_known','joint_RAW_adoption_POST_legal_CF','joint_both_adoptions_body_and_CF'):
                row[key+'_candidate']=a[key];row[key+'_reference']=b[key]
                row[key+'_delta']=int(a[key])-int(b[key]) if a[key] is not None and b[key] is not None else None
            row['POST_has_legal_CF_delta']=a['POST_has_legal_CF']-b['POST_has_legal_CF']
            for stage in ('RAW','POST'):
                for metric in CONTINUOUS:
                    key=stage+'_'+metric
                    row[key+'_delta']=a[key]-b[key] if a[key] is not None and b[key] is not None else None
            rows.append(row)
    summaries=[]
    for candidate,reference in CONTRASTS:
        for noise in ('ALL','OLD','FRESH'):
            for split in ('ALL','TRAIN','DEV'):
                subset=[r for r in rows if r['candidate_model']==candidate and r['reference_model']==reference
                        and (noise=='ALL' or r['noise_partition']==noise) and (split=='ALL' or r['split']==split)]
                if not subset:continue
                row=dict(candidate_model=candidate,reference_model=reference,noise_partition=noise,split=split,paired_outputs=len(subset),
                         joint_gained=sum(r['joint_RAW_adoption_POST_legal_CF_delta']==1 for r in subset),
                         joint_lost=sum(r['joint_RAW_adoption_POST_legal_CF_delta']==-1 for r in subset),
                         joint_unknown_pairs=sum(r['joint_RAW_adoption_POST_legal_CF_delta'] is None for r in subset),differences={})
                for key in subset[0]:
                    if key.endswith('_delta'):
                        parents=sorted({r['parent_id'] for r in subset})
                        pm=[_stats([r[key] for r in subset if r['parent_id']==p])['mean'] for p in parents]
                        row['differences'][key]=dict(output_distribution=_stats([r[key] for r in subset]),equal_parent_distribution=_stats(pm))
                summaries.append(row)
    return rows,summaries


def diversity(outputs,meshes):
    """Descriptive corner-cloud dispersion; independent version, no new points."""
    from scipy.spatial import cKDTree
    groups=defaultdict(list)
    for row in outputs:groups[row['label'],row['condition_id']].append(row)
    pairs=[];summary=[]
    for (label,cid),rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:r['candidate'])
        for stage in ('RAW','POST'):
            for noise in ('ALL','OLD','FRESH'):
                selected=[r for r in rows if noise=='ALL' or r['evaluation_noise_partition']==noise]
                local=[]
                for a,b in itertools.combinations(selected,2):
                    first=meshes[label,a['request_id'],stage].reshape(-1,3).astype(np.float64)
                    second=meshes[label,b['request_id'],stage].reshape(-1,3).astype(np.float64)
                    if len(first) and len(second):
                        da=cKDTree(second).query(first,workers=1)[0];db=cKDTree(first).query(second,workers=1)[0]
                        value=float(np.sqrt(.5*(np.mean(da**2)+np.mean(db**2)))/D.L);status='FINITE'
                    else:value=None;status='NA_EMPTY_FREE_CORNERS'
                    pair=dict(label=label,condition_id=cid,parent_id=a['parent_id'],split=a['split'],stage=stage,noise_partition=noise,
                        request_a=a['request_id'],request_b=b['request_id'],corner_count_a=len(first),corner_count_b=len(second),
                        symmetric_corner_RMS_over_L=value,status=status)
                    pairs.append(pair);local.append(pair)
                hashes=[D.array_sha(meshes[label,r['request_id'],stage]) for r in selected]
                summary.append(dict(label=label,condition_id=cid,parent_id=rows[0]['parent_id'],split=rows[0]['split'],stage=stage,
                    noise_partition=noise,outputs=len(selected),pairs=len(local),byte_distinct_free_arrays=len(set(hashes)),
                    dispersion=_stats([r['symmetric_corner_RMS_over_L'] for r in local]),
                    all_outputs_used=True,no_quality_filter=True))
    return pairs,summary


def analyze(reviews_path=None):
    _model_free();ready=readiness(reviews_path)
    if ready['status']!='READY_FOR_VALIDATION':
        raise RuntimeError('Final comparison incomplete; no success report written: '+str(ready['missing_count'])+' missing files')
    path=D.resolve(ready['reviews_path']);m=D.read(D.OUT/'inputs/manifest.json');protected={str(D.OUT/'inputs/manifest.json'):D.sha(D.OUT/'inputs/manifest.json')}
    reviews=load_reviews(m,path,protected)
    outputs,stage_rows,meshes,definitions=load_outputs(m,reviews,protected)
    summary,parent_rows=summarize(outputs);pair_rows,pair_summary=paired(outputs)
    diversity_pairs,diversity_summary=diversity(outputs,meshes)
    _model_free();_digest_files(protected)
    tables={'per_output.csv':outputs,'per_output_stage.csv':stage_rows,'per_parent.csv':parent_rows,
            'paired_per_request.csv':pair_rows,'diversity_pairs.csv':diversity_pairs}
    for name,rows in tables.items():_csv(DEST/name,rows)
    D.write(DEST/'per_output.json',outputs);D.write(DEST/'paired_summary.json',pair_summary)
    D.write(DEST/'diversity_summary.json',dict(version='terminal_nft_existing_free_corner_dispersion_v1',rows=diversity_summary,
        definition='sqrt(0.5*(mean nearest_corner_distance(A->B)^2 + mean nearest_corner_distance(B->A)^2))/fixed L; existing free corners only, duplicates retained',
        limitations=['Corner density and triangulation dependent; not an area sampled surface distance or semantic diversity score.',
                     'No GT, feature network, new random queries, or quality filtering. More dispersion does not establish useful variation.',
                     'Exact free-array hashes are ordering-sensitive duplication diagnostics; distinct hashes do not guarantee distinct shapes.',
                     'FRESH has only2 noises per condition; descriptive evidence, not independent population inference.']))
    result=dict(status='COMPLETE',version=VERSION,outputs=108,RAW_scores=108,POST_scores=108,actual_reviews=108,
        distinct_requests=36,parent_conditions=6,models=list(LABELS),groups=summary,
        main_joint_definition='RAW actual adoption YES AND POST C_F_valid_edge_count>0; UNKNOWN adoption retained unless no interface makes joint false',
        stricter_joint_definition='Both RAW/POST adoption YES, POST body complete YES, and at least one valid POST CF edge',
        aggregation='Outputs retained; metrics first mean within parent/noise partition then equal parents. OLD/FRESH and TRAIN/DEV shown separately. UNKNOWN review has its own count and positive-rate bounds.',
        limitations=['Three TRAIN and three DEV evaluation parents are repeatedly observed; FRESH means new noise, not new parents.',
                     'A legal edge alone does not certify a complete seam, semantic body attachment, manifoldness, or editability.',
                     'CF illegal, FF illegal, ConflictCov_C lower/upper and status, deletion, and visual adoption remain separate.',
                     'Human reviews and AI_PROXY remain explicitly identified; anonymized model labels do not erase previous reviewer familiarity.',
                     'No statistical superiority, automatic model promotion or default switch is inferred by this script.'],
        definitions=definitions,protected_hashes=protected,model_calls=0,new_geometry_samples=0,
        metrics_recomputed=0,descriptive_corner_pair_distances=len(diversity_pairs),torch_imported=False,
        analysis_source_sha256=D.sha(__file__),tables={name:dict(path=str(DEST/name),sha256=D.sha(DEST/name),rows=len(rows)) for name,rows in tables.items()})
    D.write(DEST/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reviews');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:
        value=readiness(args.reviews);print(json.dumps(value,ensure_ascii=False));return
    r=analyze(args.reviews)
    print(json.dumps({k:r[k] for k in ('status','outputs','actual_reviews','model_calls','descriptive_corner_pair_distances')}),flush=True)


if __name__=='__main__':main()
