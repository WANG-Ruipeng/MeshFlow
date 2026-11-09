"""Read-only historical context for fixed parent/noise FULL-patch evaluation.

Does not rescore, rerender, re-review, load models or write old artifacts.
"""
from __future__ import annotations
import collections, csv, json, time
from pathlib import Path
import numpy as np
from . import common as C

PRIOR=C.ROOT/'experiments/r2_c_reference_frame_v1'
GALLERY=PRIOR/'evaluation/gallery'
HOUT=C.OUT/'historical_context'
FIELDS=('C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer','whole_body_complete','support_missing','fragments_or_folded_layers','visible_join')

def rows(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def array_eq(a,b):return C.same(np.asarray(a),np.asarray(b))
def area_center(tri):
    x=np.asarray(tri,np.float32);cross=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0])
    area=np.sqrt(np.sum(cross*cross,axis=1,dtype=np.float32))*np.float32(.5)
    return np.sum(area[:,None]*np.mean(x,axis=1,dtype=np.float32),axis=0,dtype=np.float32)/np.sum(area,dtype=np.float32)
def describe(source,query,N):
    tri=np.asarray(source['C_original'],np.float64);ids=source['C_original_source_vertex_ids']
    edges=collections.Counter()
    vertices={}
    for row,t in zip(ids,tri):
        for i in range(3):
            vertices[int(row[i])]=t[i]
            edges[tuple(sorted((int(row[i]),int(row[(i+1)%3]))))]+=1
    boundary=[x for x,n in edges.items() if n==1]
    bvertices=sorted({v for e in boundary for v in e});cntr=area_center(source['C'])
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2
    w=query['C_weights'];interior=query['C_border_distance']>=.01*C.L
    return dict(K=len(tri),N=N,free_count=N-len(tri),area_model=float(area),
        CREF_FP32_center=cntr.tolist(),boundary_vertex_count=len(bvertices),boundary_edge_count=len(boundary),
        boundary_length_model=float(sum(np.linalg.norm(vertices[a]-vertices[b]) for a,b in boundary)),
        boundary_source_vertex_ids=bvertices,source_face_ids=source['source_face_ids'].tolist(),
        query_count=len(w),query_weight_denominator=float(w.sum()),query_interior_weight_fraction=float(w@interior/w.sum()))

def main():
    reg=C.load_registration();regsha=C.sha(C.OUT/'registration.json');protected={}
    def pin(p,h=None):
        got=C.sha(p)
        if h is not None and got!=h:raise ValueError('Historical source changed '+str(p))
        protected[str(C.resolve(p))]=got;return got
    pin(C.OUT/'registration.json')
    oldreg=C.read(PRIOR/'evaluation/registration.json');pin(PRIOR/'evaluation/registration.json')
    old_requests={q['request_id']:q for q in oldreg['requests']}
    oldqueries={q['condition_id']:q for q in oldreg['queries']};newqueries={q['condition_id']:q for q in reg['queries']}
    scorepath=PRIOR/'evaluation/per_output.csv';pin(scorepath);scores={(x['label'],x['request_id'],x['stage']):x for x in rows(scorepath)}
    keypath=PRIOR/'evaluation/blinding/model_key.json';key=C.read(keypath)['assignments'];pin(keypath)
    completion=C.read(GALLERY/'review_completion.json');pin(GALLERY/'review_completion.json')
    originals={}
    for name in ('73b','3f','dev','familiar_full'):
        p=GALLERY/f'review_{name}.csv';filehash=pin(p,completion['review_files'][str(p)])
        for source_row in rows(p):
            row=dict(source_row);row['label']=key[row['condition_id']][row['anonymous_label']]
            identity=(row['label'],row['request_id'],row['stage'])
            if identity in originals:raise ValueError('Duplicate original actual review')
            originals[identity]=(row,p,filehash)
    unblindpath=GALLERY/'AI_PROXY_review_unblinded.csv';pin(unblindpath)
    decoded={(x['label'],x['request_id'],x['stage']):x for x in rows(unblindpath)}
    historical_scores=[];historical_reviews=[];context=[];condition_changes={};images={}
    old_complete=PRIOR/'evaluation/score_completion.json';pin(old_complete)
    for q in reg['requests']:
        oq=old_requests[q['original_partial_request_id']]
        if (q['parent_id'],q['candidate'],q['N'])!=(oq['parent_id'],oq['candidate'],oq['N']):raise ValueError('Parent/noise/N pairing mismatch')
        if q['full_Gaussian_sha256']!=oq['full_Gaussian_sha256']:raise ValueError('Full Gaussian pairing mismatch')
        newqid=q['condition_id'];oldqid=oq['condition_id'];changed=q['action']=='NEW'
        if q['parent_id'] not in condition_changes:
            pin(q['condition_source'],q['condition_source_sha256']);pin(oq['condition_source'],oq['condition_source_sha256'])
            source_new=C.arrays(q['condition_source']);source_old=C.arrays(oq['condition_source'])
            nq,oldq=newqueries[newqid],oldqueries[oldqid]
            pin(nq['path'],nq['sha256']);pin(oldq['path'],oldq['sha256'])
            an=C.arrays(nq['path']);ao=C.arrays(oldq['path'])
            ndesc=describe(source_new,an,q['N']);odesc=describe(source_old,ao,oq['N'])
            if not set(odesc['source_face_ids'])<=set(ndesc['source_face_ids']):raise ValueError('Not a nested source C change')
            if not changed:
                if not array_eq(source_new['C'],source_old['C']) or nq['sha256']!=oldq['sha256']:raise ValueError('Unchanged FULL cache condition differs')
            change=dict(parent_id=q['parent_id'],old_condition_id=oldqid,new_condition_id=newqid,
                input_changed=changed,comparison_type='HALF_TO_FULL_MULTIFACTOR' if changed else 'IDENTICAL_FULL_REUSED_MESH_CONTROL',
                old_condition_type=oq['condition_type'],new_condition_type=q['condition_type'],new_patch_shape=q.get('patch_shape'),
                old_query_path=oldq['path'],old_query_sha256=oldq['sha256'],new_query_path=nq['path'],new_query_sha256=nq['sha256'],
                same_query_bytes=oldq['sha256']==nq['sha256'],same_full_N_Gaussian_per_noise=True,
                old_source_path=oq['condition_source'],old_source_sha256=oq['condition_source_sha256'],
                new_source_path=q['condition_source'],new_source_sha256=q['condition_source_sha256'],
                CREF_center_shift_model=float(np.linalg.norm(np.asarray(ndesc['CREF_FP32_center'])-np.asarray(odesc['CREF_FP32_center']))),
                area_ratio_NEW_over_OLD=ndesc['area_model']/odesc['area_model'],
                interpretation='K, area, boundary, free count and CREF origin change jointly; I/AVMR denominators and measured domains change' if changed else 'Same geometry/output controls: changed visual labels indicate review/view drift, not an input benefit')
            for prefix,desc in [('old',odesc),('new',ndesc)]:
                for k,v in desc.items():change[prefix+'__'+k]=v
            condition_changes[q['parent_id']]=change
        change=condition_changes[q['parent_id']]
        for arm in reg['labels']:
            for stage in ('RAW','POST'):
                identity=(arm,oq['request_id'],stage)
                score=dict(scores[identity]);pin(score['score_path'],score['score_sha256'])
                review,reviewfile,reviewhash=originals[identity]
                if review!=decoded[identity]:raise ValueError('Anonymous-key unbinding differs from original saved analysis')
                if review['review_status']!='REVIEWED' or any(review[k] not in ('YES','NO','UNCERTAIN') for k in FIELDS):raise ValueError('Missing actual original review')
                if review['image_sha256']!=review['reviewed_image_sha256']:raise ValueError('Unbound actual reviewed image')
                pin(review['image_path'],review['image_sha256']);images[review['image_path']]=review['image_sha256']
                historical_scores.append(score)
                historical_reviews.append(dict(review,source_actual_review_path=str(reviewfile),source_actual_review_sha256=reviewhash,
                    old_unblinding_key_path=str(keypath),old_unblinding_key_sha256=protected[str(keypath)],
                    new_request_id=q['request_id'],new_condition_id=q['condition_id'],historical_read_only=True))
                row=dict(parent_id=q['parent_id'],candidate=q['candidate'],label=arm,stage=stage,
                    old_request_id=oq['request_id'],new_request_id=q['request_id'],old_condition_id=oldqid,new_condition_id=newqid,
                    old_condition_type=oq['condition_type'],new_condition_type=q['condition_type'],input_changed=changed,
                    input_comparison_type=change['comparison_type'],old_K=oq['K'],new_K=q['K'],N=q['N'],old_free_N_minus_K=q['N']-oq['K'],new_free_N_minus_K=q['N']-q['K'],
                    old_C_area_model=change['old__area_model'],new_C_area_model=change['new__area_model'],CREF_center_shift_model=change['CREF_center_shift_model'],
                    old_boundary_vertex_count=change['old__boundary_vertex_count'],new_boundary_vertex_count=change['new__boundary_vertex_count'],
                    old_query_sha256=change['old_query_sha256'],new_query_sha256=change['new_query_sha256'],
                    old_score_path=score['score_path'],old_score_sha256=score['score_sha256'],
                    old_image_path=review['image_path'],old_image_sha256=review['image_sha256'],
                    old_review_path=str(reviewfile),old_review_sha256=reviewhash,old_reviewer=review['reviewer'],old_model_names_hidden=review['model_names_hidden'],
                    old_prior_familiarity=review['prior_familiarity'],old_notes=review['notes'],
                    review_comparison_limit='Prior names-hidden AI_PROXY with disclosed familiarity versus current nonblind AI_PROXY; views and possibly reviewer differ. No net causal gain from label changes alone.',
                    numeric_comparison_limit='I, A-VMR/PMR and boundary distances change measurement domain; do not treat old/new fractions as a pure improvement. CF/FF free counts also change.' if changed else 'Strict identical condition and mesh reuse control; scores should remain numerically identical under the same definitions.')
                for name in FIELDS:row['old_review__'+name]=review[name]
                for name in ('ConflictCov_C','ConflictCov_C_cross','ConflictCov_C_dup','CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','C_F_valid_edge_count','C_F_has_valid_edge','A_VMR__raw_tau_1e-6','PMR__raw_tau_1e-6','A_VMR__L_tau_0.005','PMR__L_tau_0.005','actual_free_faces','boundary_vertex_denominator','C_query_weight_denominator'):
                    row['old_score__'+name]=score[name]
                context.append(row)
    if not (len(context)==len(historical_scores)==len(historical_reviews)==96 and len(condition_changes)==6 and len(images)==24):raise ValueError('Historical budget mismatch')
    if len({(x['label'],x['old_request_id'],x['stage']) for x in context})!=96:raise ValueError('Historical duplicate identities')
    C.csvwrite(HOUT/'historical_per_output.csv',historical_scores)
    C.csvwrite(HOUT/'historical_actual_reviews.csv',historical_reviews)
    C.csvwrite(HOUT/'historical_context_per_output.csv',context)
    C.csvwrite(HOUT/'condition_changes.csv',list(condition_changes.values()))
    C.csvwrite(HOUT/'historical_images.csv',[dict(path=p,sha256=h) for p,h in images.items()])
    # Existing observations only. No new full-versus-half numeric delta or causal label.
    oldcounts=[]
    for item in condition_changes.values():
        for arm in reg['labels']:
            for stage in ('RAW','POST'):
                rr=[x for x in context if (x['parent_id'],x['label'],x['stage'])==(item['parent_id'],arm,stage)]
                cnt=collections.Counter(x['old_review__C_adoption'] for x in rr)
                oldcounts.append(dict(parent_id=item['parent_id'],old_condition_id=item['old_condition_id'],new_condition_id=item['new_condition_id'],label=arm,stage=stage,
                    input_changed=item['input_changed'],old_observations=len(rr),**{k:cnt[k] for k in ('YES','NO','UNCERTAIN')}))
    C.csvwrite(HOUT/'historical_adoption_counts.csv',oldcounts)
    for path,h in protected.items():
        if C.sha(path)!=h:raise ValueError('Read-only source changed '+path)
    C.write(HOUT/'provenance.json',dict(status='COMPLETE_READ_ONLY_HISTORICAL_CONTEXT',new_registration_sha256=regsha,
        source_hashes=protected,all_sources_unchanged=True,code_sha256=C.sha(__file__),
        old_score_rows=96,old_actual_review_rows=96,unique_old_request_pages=24,conditions=6,
        changed_HALF_to_FULL_conditions=4,unchanged_FULL_controls=2,
        old_blinding_policy=completion['notes'],old_review_status='Copied actual original reviewed rows, decoded with frozen per-condition key; no new re-review',
        model_calls=0,training_updates=0,new_generation=0,old_quality_rescores=0,new_images=0,new_visual_reviews=0,
        scope='Historical descriptive context only. Models/noise identity paired; four input surfaces and their metric domains differ. Two unchanged full controls can expose review drift.',
        outputs={str(p):C.sha(p) for p in HOUT.glob('*.csv')}))
    print(json.dumps(dict(status='COMPLETE',old_score_rows=96,old_review_rows=96,condition_changes=6,old_images=24)),flush=True)

if __name__=='__main__':main()

