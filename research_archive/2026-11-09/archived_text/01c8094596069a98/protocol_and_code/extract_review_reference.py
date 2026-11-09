"""Read saved CSV/JSON only. No images, models, arrays or scoring imports."""
import argparse, csv, hashlib, json
from pathlib import Path

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text('utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rr,fields=None):
    p.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for r in rr for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rr)
def key(r):return r['request_id'],r['label'],r['stage']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    source=Path(a.source).resolve();out=Path(a.out).resolve()
    assert source!=out and not out.is_relative_to(source)
    manifest=read(source/'delivery/R2_USER_DATA_SCALING_32_67_REVIEW/PACKAGE_MANIFEST.json')
    known={x['archive_path']:x['sha256'] for x in manifest['files']}
    sources={}
    def track(rel):
        p=source/rel
        h=sha(p);prior=known.get(rel,'NA_NOT_IN_PRIOR_PACKAGE_MANIFEST')
        sources[rel]=dict(path=str(p),existing_sha256=prior,this_read_sha256=h,hash_scope='SMALL_CSV_OR_JSON_BYTES_READ_THIS_RUN')
        if len(prior)==64: assert h==prior,(rel,'source changed since prior package')
        return p
    original=rows(track('evaluation/AI_PROXY_visual_review.csv'))
    lookup={};handoff=read(track('review_work/chunk_0_handoff.json'))
    for chunk in range(4):
        rr=rows(track(f'review_work/chunk_{chunk}.csv'));receipt=read(track(f'review_work/chunk_{chunk}_receipt.json'))
        for i,r in enumerate(rr):
            session={1:'/root/scaling_training',2:'/root/scaling_evaluation',3:'/root/scaling_assets'}.get(chunk,'/root')
            first=receipt['first_recorded_utc'];last=receipt['updated_utc']
            if chunk==0:
                if i<120:first,last=handoff['root_first_recorded_utc'],handoff['root_last_recorded_utc']
                else:session=handoff['assignee'];first,last=handoff['assignee_started_utc'],handoff['assignee_finished_utc']
            assert key(r) not in lookup
            lookup[key(r)]=(r,chunk,session,first,last)
    gallery=read(track('evaluation/gallery/manifest.json'))
    images={(x['identity']['request_id'],x['identity']['stage']):x for x in gallery['receipts']}
    exported=[]
    for r in original:
        old,chunk,session,first,last=lookup[key(r)]
        assert r==old,'frozen review differs from chunk'
        page=images[r['request_id'],r['stage']]
        assert r['reviewed_image_sha256']==page['image_sha256']
        exported.append(dict(r,parent_uid=r['parent_id'],model_label=r['label'],noise_id=r['candidate'],
            condition_type=r['region_type']+'_'+r['tier'],reviewer_assignment=session,original_chunk=chunk,
            review_timestamp='NA_NOT_RECORDED_PER_ROW',review_interval_start_utc=first,review_interval_end_utc=last,
            review_time_scope='SAVED_CHUNK_OR_SERIAL_HANDOFF_INTERVAL_NOT_PER_ROW',
            image_page_path=page['image_path'],image_hash_status='EXISTING_RECORDED_NOT_IMAGE_REREAD',
            source_review_csv=str(source/'evaluation/AI_PROXY_visual_review.csv'),
            source_review_csv_sha256=sources['evaluation/AI_PROXY_visual_review.csv']['this_read_sha256'],
            source_record_key='|'.join(key(r)),original_values_modified='False'))
    assert len(exported)==840
    write(out/'visual_review_840.csv',exported)
    refs=rows(track('report_tables/absolute_parent_equal.csv'))
    refs=[r for r in refs if r['stratum']=='ALL' and r['stratum_value']=='ALL']
    fields=['panel','stage','label','metric','parent_equal_mean','parent_median','registered_parents','finite_parents',
            'numerically_complete_parents','detector_uncertain_parents','status','direction']
    write(out/'protocol_and_code/original_aggregate_reference.csv',[{k:r[k] for k in fields} for r in refs])
    dirs=rows(track('report_tables/metric_directions.csv'))
    write(out/'protocol_and_code/registered_metric_fields.csv',dirs)
    track('review_work/REVIEW_RULES.md')
    (out/'protocol_and_code/original_visual_review_rules.md').write_bytes((source/'review_work/REVIEW_RULES.md').read_bytes())
    provenance=dict(outputs={
        'visual_review_840.csv':dict(rows=840,derived=False,transformation='Original columns and notes exact strings; append existing page/chunk/assignment references, aliases, and explicit per-row-time NA',sources=list(sources)[:12]),
        'protocol_and_code/original_aggregate_reference.csv':dict(rows=len(refs),derived=False,transformation='Filter stratum=ALL,value=ALL; retain listed source columns without numeric parsing',sources=['report_tables/absolute_parent_equal.csv']),
        'protocol_and_code/registered_metric_fields.csv':dict(rows=len(dirs),derived=False,transformation='CSV copy; no value changes',sources=['report_tables/metric_directions.csv']),
        'protocol_and_code/original_visual_review_rules.md':dict(rows=None,derived=False,transformation='Byte-identical copy',sources=['review_work/REVIEW_RULES.md'])},sources=sources)
    (out/'_work').mkdir(exist_ok=True)
    (out/'_work/review_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'_work/review_missing.json').write_text(json.dumps([dict(field='review_timestamp',status='NOT_RECORDED_PER_ROW',scope='840 original observations',available='Chunk/handoff intervals preserved; image bytes not reread')],indent=2),encoding='utf-8')
    print(json.dumps(dict(review_rows=len(exported),reference_rows=len(refs),registered_metrics=len(dirs),model_calls=0,image_reads=0)))

if __name__=='__main__':main()
