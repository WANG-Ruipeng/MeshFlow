"""Consolidate completed, reviewed CPU target variants without altering old assets."""
from . import common as c
from collections import Counter
import csv
import io
import html


def text_once(path,text):
    path=c.resolve(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text('utf-8-sig')!=text:raise RuntimeError('Immutable text differs: '+str(path))
    else:path.write_text(text,encoding='utf-8-sig',newline='\n')


def main():
    src=c.read(c.SOURCE);protocol=c.read(c.OUT/'protocol.json')
    v1={r['parent_id']:r for r in c.read(c.OUT/'numeric_results.json')['rows']}
    v2={r['parent_id']:r for r in c.read(c.OUT/'engineering_v2/numeric_results.json')['rows']}
    parents=[];rows=[];accepted=[];reviews=[]
    for s in src['train']:
        pid=s['parent_id'];r=v2.get(pid,v1[pid]);old=c.arrays(s['case_path'])
        if c.sha(s['case_path'])!=s['case_sha256']:raise RuntimeError('Old case changed: '+pid)
        folder=c.resolve(r.get('proposal_path',c.OUT/'parents'/pid/'proposal.npz')).parent
        out=dict(parent_id=pid,condition_id=s['condition_id'],replay_group=s['replay_group'],
            N_original=s['N'],K=s['K'],N_proposed=r['N_candidate'],
            proposal_revision='engineering_v2' if pid in v2 else 'v1',
            proposal_attempts=(2 if pid in v2 else int('proposal_path' in r)),
            numeric_status=r['status'],status=r['status'],reasons=';'.join(r['reasons']),
            visual_status='NA',C_exact='NA',interface_exact='NA',area_ratio='NA',
            surface_probe_max_pct_L='NA',bbox_shift_pct_L='NA',
            new_CF_pairs='NA',new_FF_pairs='NA',component_identities_preserved='NA')
        variants=[dict(variant_id='NATURAL',kind='NATURAL',N=s['N'],K=s['K'],
            case_path=s['case_path'],case_sha256=s['case_sha256'],
            target_version_id=str(old['target_version_id'].item()),
            source_admission='Inherited frozen original pool; not a new all-clean certification.')]
        if 'quality_path' in r:
            q=c.read(r['quality_path']);checks=q['checks'];f=checks['SAMPLED_SURFACE_AND_BBOX']
            out.update(C_exact=checks['C_AND_SOURCE_CF_INTERFACE_UNCHANGED']['pass_'],
                interface_exact=checks['C_AND_SOURCE_CF_INTERFACE_UNCHANGED']['pass_'],
                area_ratio=checks['TOTAL_AREA_RATIO']['ratio'],
                surface_probe_max_pct_L=100*max(f['source_to_new_probe_max_L'],f['new_to_source_probe_max_L']),
                bbox_shift_pct_L=100*f['bbox_max_coordinate_shift_L'],
                component_identities_preserved=all(x['topology_one_to_one'] for x in checks['COMPONENT_PRESERVATION']['per_component']),
                new_CF_pairs=q['finite_pairs']['by_partition']['CF']['new_count'],
                new_FF_pairs=q['finite_pairs']['by_partition']['FF']['new_count'])
        if r['status']=='NUMERIC_PASS_PENDING_VISUAL':
            reviewpath=c.OUT/'visual_review'/(pid+'.json')
            if not reviewpath.exists():raise RuntimeError('Actual visual review missing: '+pid)
            review=c.read(reviewpath)
            for im in review['viewed_images']:
                if c.sha(im['path'])!=im['sha256']:raise RuntimeError('Reviewed image changed')
            out['visual_status']=review['status'];reviews.append(review)
            if review['status']=='PASS':
                a=c.arrays(r['case_path'])
                if c.sha(r['case_path'])!=r['case_sha256']:raise RuntimeError('New case changed')
                for key in ('C','C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','C_source_corner_permutations'):
                    if old[key].dtype!=a[key].dtype or old[key].shape!=a[key].shape or old[key].tobytes()!=a[key].tobytes():
                        raise RuntimeError('C identity changed: '+pid+':'+key)
                out['status']='ACCEPTED_LOW';accepted.append(out)
                variants.append(dict(variant_id='LOW',kind='LOW',N=r['N_candidate'],K=s['K'],
                    case_path=r['case_path'],case_sha256=r['case_sha256'],
                    target_version_id=str(a['target_version_id'].item()),
                    quality_path=r['quality_path'],quality_sha256=r['quality_sha256'],
                    actual_visual_review=str(reviewpath),actual_visual_review_sha256=c.sha(reviewpath)))
            else:
                out['status']='BLOCKED_VISUAL';out['reasons']=str(review['rejection_reason'])
        parents.append(dict(parent_id=pid,condition_id=s['condition_id'],split='TRAIN',
            replay_group=s['replay_group'],variants=variants))
        rows.append(out)
    if c.sha(c.SOURCE)!=protocol['source_manifest_sha256']:raise RuntimeError('Source manifest changed')
    manifest=dict(schema='BENCH_FIXED_C_MULTILOD_POOL_V1',parents=parents,
        source_manifest=str(c.SOURCE),source_manifest_sha256=c.sha(c.SOURCE),
        original_parents=len(parents),added_independent_parents=0,
        original_physical_conditions=len(parents),added_physical_conditions=0,
        added_geometry_versions=len(accepted),total_versions=sum(len(p['variants']) for p in parents),
        scheduler='Preserve original OLD/FOCUS/HIGH parent exposure sequence; select NATURAL/LOW only after the parent. Do not flatten variants into a uniform list.',
        conditions='C, K, source C vertex IDs and C winding unchanged; free geometry/count vary.',
        deployment_inputs='C and real total N. Owner/alias/source geometry remain supervision-only metadata.',
        training_status='DATA_READY; CPU variant adapter only. Generator integration, paired OT/noise stream and budget must be frozen before training.',
        model_forwards=0,optimizer_updates=0)
    c.write(c.OUT/'data/manifest.json',manifest)
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text_once(c.OUT/'data/per_parent.csv',buf.getvalue())
    counts=dict(Counter(x['status'] for x in rows))
    c.write(c.OUT/'data/admission_summary.json',dict(statuses=counts,original_hashes_verified=len(parents),
        original_parents=len(parents),added_versions=len(accepted),model_forwards=0,model_updates=0,
        max_accepted_probe_distance_pct_L=max(x['surface_probe_max_pct_L'] for x in accepted),
        min_accepted_area_ratio=min(x['area_ratio'] for x in accepted),
        max_accepted_area_ratio=max(x['area_ratio'] for x in accepted)))
    index='<!doctype html><meta charset="utf-8"><title>Bench accepted multi-LOD targets</title><style>body{font:16px sans-serif}img{width:100%;max-width:1600px}</style><h1>Actual reviewed target augmentation</h1><p>Blue: unchanged C. Gray: source target free faces. These are data, not generated results. Reviews: AI_PROXY, nonblind.</p>'
    for row in accepted:
        review=next(x for x in reviews if x['parent_id']==row['parent_id'])
        image=c.resolve(review['viewed_images'][0]['path']).relative_to(c.OUT).as_posix()
        index+=f'<section><h2>{html.escape(row["parent_id"])} N{row["N_original"]} → {row["N_proposed"]}</h2><p>{html.escape(str(review["observations"]))}</p><img loading="lazy" src="{image}"></section>'
    text_once(c.OUT/'REVIEW.html',index)
    c.event('CPU_DATA_ADMITTED',counts=counts,added_versions=len(accepted),model_forwards=0,model_updates=0)
    print(c.json.dumps(dict(counts=counts,added_versions=len(accepted),total_versions=manifest['total_versions']),ensure_ascii=False))


if __name__=='__main__':main()
