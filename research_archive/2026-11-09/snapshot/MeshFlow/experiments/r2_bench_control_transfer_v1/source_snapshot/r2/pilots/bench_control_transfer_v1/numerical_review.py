"""Independent aggregation/graph bookkeeping on already saved Bench evidence.

No model imports, geometry classifiers, distance queries, or new metric scans.
"""
from __future__ import annotations
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'experiments/r2_bench_control_transfer_v1'
EV=RUN/'evaluation'
OUT=RUN/'numerical_review'


def read(path):return json.loads(Path(path).read_text('utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csvread(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
def number(v):
    if v in (None,'','None'):return None
    if str(v) in ('True','False'):return float(str(v)=='True')
    return float(v)
def flag(v):
    assert str(v) in ('True','False'), v
    return str(v)=='True'
def savecsv(path,rows):
    if path.exists():raise FileExistsError(str(path))
    with path.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def close(x,y):assert abs(float(x)-float(y))<1e-8*max(1.,abs(float(y))), (x,y)


def main():
    rows=csvread(EV/'per_output.csv');reg=read(EV/'score_registration.json')
    summary=read(EV/'summary_completion.json')
    assert len(rows)==96 and summary['numeric_field_count']==124
    qq={q['request_id']:q for q in reg['requests']}
    lookup={(r['label'],r['stage'],r['request_id']):r for r in rows}
    assert len(lookup)==96
    inputs={str(EV/'per_output.csv'):sha(EV/'per_output.csv'),
        str(EV/'score_registration.json'):sha(EV/'score_registration.json')}
    groups={
        'TRAIN':[k for k,q in qq.items() if q['split']=='TRAIN'],
        'DEV':[k for k,q in qq.items() if q['split']=='DEV'],
        'DEV_OFFICIAL_TEST':[k for k,q in qq.items() if q['split']=='DEV' and q['official_split']=='test'],
        'DEV_PRETRAIN_TRAIN':[k for k,q in qq.items() if q['split']=='DEV' and q['official_split']=='train'],
        'ALL':list(qq),
    }
    metrics=['C_F_has_valid_edge','C_F_valid_edge_count','C_F_valid_edge_total_length',
        'reached_free_area_fraction','FF_illegal_affected_free_fraction','CF_illegal_affected_free_fraction',
        'ConflictCov_C','free_minimum_angle_lt5_fraction','actual_free_faces','deleted_free_faces',
        'free_total_surface_area','EXACT_COORD_face_components','EXACT_COORD_nonmanifold_edges']
    aggregate=[];parentpairs=[]
    for group,ids in groups.items():
        parentids=sorted({qq[i]['parent_id'] for i in ids})
        for stage in ('RAW','POST'):
            for metric in metrics:
                x=np.array([number(lookup['PLAIN',stage,i][metric]) for i in ids],float)
                y=np.array([number(lookup['SEAM',stage,i][metric]) for i in ids],float)
                assert np.isfinite(x).all() and np.isfinite(y).all()
                parentd=[]
                for parent in parentids:
                    ii=[i for i in ids if qq[i]['parent_id']==parent];assert len(ii)==2
                    xv=[number(lookup['PLAIN',stage,i][metric]) for i in ii]
                    yv=[number(lookup['SEAM',stage,i][metric]) for i in ii]
                    d=float(np.mean(yv)-np.mean(xv));parentd.append(d)
                    parentpairs.append(dict(group=group,stage=stage,parent_id=parent,metric=metric,
                        requests=2,PLAIN=float(np.mean(xv)),SEAM=float(np.mean(yv)),SEAM_minus_PLAIN=d))
                aggregate.append(dict(group=group,stage=stage,metric=metric,requests=len(ids),parents=len(parentids),
                    PLAIN_mean=float(x.mean()),SEAM_mean=float(y.mean()),SEAM_minus_PLAIN=float((y-x).mean()),
                    PLAIN_sum=float(x.sum()),SEAM_sum=float(y.sum()),
                    request_delta_negative=int((y<x).sum()),request_delta_zero=int((y==x).sum()),request_delta_positive=int((y>x).sum()),
                    parent_delta_negative=sum(d<0 for d in parentd),parent_delta_zero=sum(d==0 for d in parentd),parent_delta_positive=sum(d>0 for d in parentd)))
    graph=[]
    for rid,q in sorted(qq.items()):
        for stage in ('RAW','POST'):
            folder=EV/'scores/SEAM'/rid/stage
            edgepath=folder/'valid_edges.csv';facepath=folder/'seam_bulk_faces.csv';jp=folder/'seam_to_bulk.json'
            saved=read(jp);ee=csvread(edgepath);ff=csvread(facepath)
            meshpath=EV/'SEAM'/rid/stage.lower()/(stage.lower()+'.npz');a=npz(meshpath)
            k=q['K'];faces=a['faces'];rawids=a['raw_slot_ids'];fullseam=a['raw_seam_slot_mask'];mask=fullseam[rawids]
            assert not mask[:k].any() and len(mask)==len(faces)
            assert np.array_equal(rawids[:k],np.arange(k)) and len(set(rawids.tolist()))==len(rawids)
            if stage=='POST':
                assert np.array_equal(a['retained_seam_mask'],mask)
                assert np.array_equal(rawids,a['source_face_ids'])
            book=collections.defaultdict(list)
            for fi,tri in enumerate(faces):
                for j in range(3):book[tuple(sorted((int(tri[j]),int(tri[(j+1)%3]))))].append(fi)
            seenedge=set();seeds=set();adj=collections.defaultdict(set);cross=[];invalidcross=[]
            for e in ee:
                edge=(int(e['vertex_u']),int(e['vertex_v']));incident=json.loads(e['incident_face_ids'])
                assert edge in book and sorted(book[edge])==sorted(incident)
                assert edge not in seenedge;seenedge.add(edge)
                if len(incident)==2 and min(incident)>=k and bool(mask[incident[0]])!=bool(mask[incident[1]]):
                    (cross if flag(e['valid_edge']) else invalidcross).append(e)
                if not flag(e['valid_edge']):continue
                assert len(incident)==2
                i,j=incident
                if min(i,j)<k:
                    if mask[max(i,j)]:seeds.add(max(i,j))
                else:adj[i].add(j);adj[j].add(i)
            # The retained original graph intentionally omits C-only edges.
            expected_edges={edge for edge,incident in book.items() if any(i>=k for i in incident)}
            assert seenedge==expected_edges
            reached=set(seeds);todo=list(seeds)
            while todo:
                i=todo.pop()
                for j in adj[i]-reached:assert j>=k;reached.add(j);todo.append(j)
            bulk=set(range(k,len(faces)))-set(np.flatnonzero(mask));reachedbulk=bulk & reached
            facerows={int(f['output_face_id']):f for f in ff}
            assert set(facerows)==set(range(k,len(faces)))
            for i,f in facerows.items():
                assert int(f['raw_slot_id'])==int(rawids[i])
                assert f['role']==('SEAM' if mask[i] else 'BULK')
                assert flag(f['legal_CF_seed'])==(i in seeds)
                assert flag(f['reached_free_only'])==(i in reached)
            denom=sum(float(facerows[i]['area_model2']) for i in bulk)
            num=sum(float(facerows[i]['area_model2']) for i in reachedbulk)
            assert denom>0 and saved['status']=='PASS'
            close(denom,saved['retained_bulk_area']);close(num,saved['free_only_reached_bulk_area'])
            close(num/denom,saved['free_only_reached_bulk_area_fraction'])
            assert len(cross)==saved['legal_seam_bulk_edge_count']
            row=lookup['SEAM',stage,rid]
            close(num/denom,number(row['free_only_reached_bulk_area_fraction']))
            graph.append(dict(request_id=rid,split=q['split'],stage=stage,raw_active_seam_faces=int(fullseam.sum()),
                retained_seam_faces=int(mask.sum()),retained_bulk_faces=len(bulk),legal_seed_seam_faces=len(seeds),
                actual_indexed_edges=len(book),saved_free_incident_edges=len(expected_edges),legal_seam_bulk_edges=len(cross),illegal_or_unknown_shared_seam_bulk_edges=len(invalidcross),
                reached_bulk_faces=len(reachedbulk),reached_bulk_area=num,retained_bulk_area=denom,
                reached_bulk_area_fraction=num/denom,role_masks_match_raw_source_ids=True,
                saved_edges_match_actual_index_incidence=True,all_saved_BFS_flags_match=True))
            for p in (edgepath,facepath,jp,meshpath):inputs[str(p)]=sha(p)
    deletions=[];cleanup=[]
    for arm in ('PLAIN','SEAM'):
        for rid,q in sorted(qq.items()):
            path=EV/arm/rid/'post/geometry.json';g=read(path);a=npz(EV/arm/rid/'post/post.npz')
            rawmask=a['raw_seam_slot_mask'];records=g['face_records'];counts=collections.Counter()
            assert len(records)==q['N']
            for f in records:
                i=f['source_face_id'];assert i<q['N']
                if not f['kept']:
                    assert i>=q['K'] and not f['known']
                    role='SEAM' if rawmask[i] else 'BULK';counts[f['reason']]+=1
                    deletions.append(dict(label=arm,request_id=rid,parent_id=q['parent_id'],split=q['split'],
                        official_split=q['official_split'],raw_slot_id=i,role=role,reason=f['reason']))
                assert bool(a['source_face_kept'][i])==f['kept']
                assert str(a['source_face_reason'][i])==f['reason']
            assert dict(counts)==g['stats']['deleted_by_reason']
            row=lookup[arm,'POST',rid]
            assert sum(counts.values())==int(number(row['deleted_free_faces']))
            assert int(number(row['actual_free_faces']))+sum(counts.values())==q['N']-q['K']
            cleanup.append(dict(label=arm,request_id=rid,split=q['split'],requested_free=q['N']-q['K'],
                retained_free=int(number(row['actual_free_faces'])),deleted_free=sum(counts.values()),
                exact_repeated_vertex=counts['exact_repeated_vertex'],duplicate_free=counts['duplicate_free'],
                duplicate_known=counts['duplicate_known'],exact_zero_area=counts['exact_zero_area'],
                other=sum(v for reason,v in counts.items() if reason not in ('exact_repeated_vertex','duplicate_free','duplicate_known','exact_zero_area'))))
            inputs[str(path)]=sha(path)
    deletion_summary=[]
    for group,ids in groups.items():
        for arm in ('PLAIN','SEAM'):
            cc=[c for c in cleanup if c['label']==arm and c['request_id'] in ids]
            dd=[d for d in deletions if d['label']==arm and d['request_id'] in ids]
            counts=collections.Counter((d['role'],d['reason']) for d in dd)
            for (role,reason),count in sorted(counts.items()):
                deletion_summary.append(dict(group=group,label=arm,requests=len(ids),role=role,reason=reason,
                    faces=count,mean_per_output=count/len(ids),fraction_of_deleted=count/len(dd),
                    requested_free_total=sum(c['requested_free'] for c in cc),retained_free_total=sum(c['retained_free'] for c in cc),
                    deleted_free_total=len(dd)))
    # Compare independent key aggregation to already completed paired table.
    official=csvread(EV/'paired_complete/paired_group_summary.csv')
    table={(r['group'],r['stage'],r['metric']):r for r in official}
    for r in aggregate:
        z=table[r['group'],r['stage'],r['metric']]
        close(r['PLAIN_mean'],number(z['baseline_parent_macro_mean']))
        close(r['SEAM_mean'],number(z['candidate_parent_macro_mean']))
        close(r['SEAM_minus_PLAIN'],number(z['paired_delta_parent_macro_mean']))
    OUT.mkdir(exist_ok=True)
    for name,data in (('key_paired_summary',aggregate),('key_per_parent',parentpairs),('saved_graph_verification',graph),
        ('cleanup_per_output',cleanup),('cleanup_reason_summary',deletion_summary)):
        savecsv(OUT/(name+'.csv'),data)
    indexed={(r['group'],r['stage'],r['metric']):r for r in aggregate}
    def value(group,metric):return indexed[group,'POST',metric]
    def pair(group,metric,pct=False):
        r=value(group,metric);scale=100 if pct else 1
        return f"{r['PLAIN_mean']*scale:.3f} → {r['SEAM_mean']*scale:.3f}"
    def winloss(group,metric):
        r=value(group,metric)
        return f"{r['parent_delta_negative']}/{r['parent_delta_zero']}/{r['parent_delta_positive']}"
    lines=['# Independent Bench numerical review','',
        'This review only reads saved per-output tables, indexed meshes, legal-edge classifications and GEO face-cleanup records. It adds no model calls, distance/intersection scans, noise, queries or metric definitions.',
        '', 'The numerical tables agree with independent paired recomputation. SEAM establishes C-side indexed interfaces, but these outputs do not show a legal shared-edge path from seam into bulk. FF and C conflict coverage increase, while the retained-face small-angle fraction decreases. This is a bounded interface result, not an overall quality upgrade.',
        '', '| POST group | Requests | C/F coverage PLAIN→SEAM | FF % | I % | Small-angle % | Retained free faces |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group,ids in groups.items():
        rr=value(group,'C_F_has_valid_edge')
        lines.append(f"| {group} | {len(ids)} | {int(rr['PLAIN_sum'])}/{len(ids)} → {int(rr['SEAM_sum'])}/{len(ids)} | {pair(group,'FF_illegal_affected_free_fraction',True)} | {pair(group,'ConflictCov_C',True)} | {pair(group,'free_minimum_angle_lt5_fraction',True)} | {pair(group,'actual_free_faces')} |")
    lines += ['', '## Graph zero: verified, not a missing or inapplicable field','',
        f"Independently reconstructed adjacency from all {len(graph)} saved SEAM RAW/POST edge tables. Every edge incidence matches the saved indexed faces; every retained role matches raw_seam_slot_mask[raw_slot_ids]; source IDs and retained masks agree. All per-face seed/reach flags and positive bulk-area denominators match saved records.",
        f"POST: {sum(g['reached_bulk_faces']>0 for g in graph if g['stage']=='POST')}/24 outputs reach any bulk face; legal seam/bulk edges total {sum(g['legal_seam_bulk_edges'] for g in graph if g['stage']=='POST')}. Non-legal indexed seam/bulk two-face edges total {sum(g['illegal_or_unknown_shared_seam_bulk_edges'] for g in graph if g['stage']=='POST')}. RAW: legal seam/bulk edges total {sum(g['legal_seam_bulk_edges'] for g in graph if g['stage']=='RAW')}. Thus zero reach is a genuine zero under the existing legal indexed-edge graph, not NA coerced to zero. PLAIN seam-specific quantities remain NA.",
        '', 'This graph does not rule out near contact, shared vertices without edges, or visual continuity. C/F seed faces are not by themselves evidence that the bulk is incorporated. Traversal through C is deliberately excluded in the unchanged definition.',
        '', '## Cleanup loss accounting','',
        '| POST group/arm | Requested free mean | Retained free mean | Deleted mean | Deletion reasons (total faces) |',
        '|---|---:|---:|---:|---|']
    for group,ids in groups.items():
        for arm in ('PLAIN','SEAM'):
            rr=[r for r in cleanup if r['label']==arm and r['request_id'] in ids]
            counts=collections.Counter(d['reason'] for d in deletions if d['label']==arm and d['request_id'] in ids)
            lines.append(f"| {group}/{arm} | {np.mean([r['requested_free'] for r in rr]):.3f} | {np.mean([r['retained_free'] for r in rr]):.3f} | {np.mean([r['deleted_free'] for r in rr]):.3f} | "+'; '.join(f'{k}: {v}' for k,v in sorted(counts.items()))+' |')
    lines += ['', 'The reason exact_repeated_vertex means a triangle has repeated vertex indices after the existing GEO consolidation. It is not a measurement of its original RAW area, nor proof that D0 alone created the problem. duplicate_free means repeated retained free-face vertex triples. These records establish why cleanup deleted faces; without a separate intervention they do not identify how much was caused by generator geometry, D0 displacement, or consolidation. Lower small-angle fraction on the surviving faces must be read alongside the much smaller surviving set. Retained surface area and face counts remain in the paired tables.',
        '', '## Per-parent paired direction','',
        'Counts below are lower / equal / higher SEAM than PLAIN after averaging the two fixed noises within each parent. For FF, CF, I and small-angle fraction lower is favorable; no significance or generalization claim is attached to these small counts.',
        '', '| Group | Parents | FF | CF | I | Small-angle |', '|---|---:|---:|---:|---:|---:|']
    for group,ids in groups.items():
        lines.append(f"| {group} | {len({qq[i]['parent_id'] for i in ids})} | {winloss(group,'FF_illegal_affected_free_fraction')} | {winloss(group,'CF_illegal_affected_free_fraction')} | {winloss(group,'ConflictCov_C')} | {winloss(group,'free_minimum_angle_lt5_fraction')} |")
    lines += ['', 'DEV contains eight condition-training-held-out parents: seven official-test parents (14 outputs), and one official-training parent (two outputs). These are reported separately. Official split metadata is not an independent audit of every possible upstream pretraining source.',
        '', 'No unique-source reconstruction ranking is introduced. Saved-head/source-owner agreement remains a separate exploratory diagnostic. Actual shape completeness, floating fragments and use of the supplied patch require the separately recorded image review; this numerical review does not invent those labels.']
    report='\n'.join(lines)+'\n'
    (OUT/'NUMERICAL_REVIEW.md').write_text(report,encoding='utf-8')
    for path,digest in inputs.items():assert sha(path)==digest
    assert 'torch' not in sys.modules
    receipt=dict(status='PASS',per_output_rows=96,registered_metrics=124,key_paired_cells_verified=len(aggregate),
        saved_graphs_verified=len(graph),cleanup_outputs_verified=len(cleanup),deleted_face_records=len(deletions),
        model_forwards=0,new_metric_scans=0,new_queries=0,inputs=inputs,source_sha256=sha(__file__),
        review_engineering_note='Initial independent guard incorrectly expected the original free-incidence edge CSV to include pure C/C edges; corrected expectation to match unchanged graph scope before any review output was written. No scoring/data/code change.',
        files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(report)


if __name__=='__main__':main()
