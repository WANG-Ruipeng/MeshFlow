"""Lossless merge of saved initial visual notes; stdlib only, no models or scoring."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import csv,hashlib,json,re,time
RUN=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1')
REPO=Path('/mnt/e/MeshFlow-Control')
OUT=RUN/'audit/visual_review/main_complete'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False)
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def csv_write(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:canon(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def main():
    tick=time.perf_counter();frozen=load(RUN/'registrations/source_hashes.json')
    assert {p:sha(REPO/p) for p in frozen}==frozen
    before_path=OUT/'pre_unblind_verification.json';before=load(before_path)
    assert before['status']=='PASS' and before['outputs']==384 and before['requests']==64 and before['original_pngs']==128
    for rel,h in before['source_hashes'].items():assert sha(RUN/rel)==h
    for rel,h in before['image_hashes'].items():assert sha(RUN/rel)==h
    assignment_path=RUN/'registrations/visual_review_assignment.json';assignment=load(assignment_path)
    mapping_path=RUN/'phases/main/gallery/private_method_mapping.json';mapping=load(mapping_path)
    assert set(mapping.values())=={'Method '+str(i) for i in range(1,7)}
    labels={v:k for k,v in mapping.items()}
    gallery_path=RUN/'phases/main/gallery/gallery.json';gallery=load(gallery_path)
    assert gallery['status']=='COMPLETE' and gallery['page_count']==128 and gallery['all_registered_outputs_shown']
    gallery_pages={(p['request_id'],p['mode']):p for p in gallery['pages']};assert len(gallery_pages)==128
    csvnames={'root':'per_output.csv','schedule40_data':'visual_review_per_output.csv','schedule40_train':'observations.csv','review_unified_recipe_code':'all_observations.csv'}
    supportnames={'root':['review_receipt.json'],'schedule40_data':['COMPLETION.json','VISUAL_REVIEW_SUMMARY.md'],'schedule40_train':['observations.json','review_manifest.json','image_hash_checks.json','VISUAL_REVIEW.md'],'review_unified_recipe_code':['coverage.json','condition_noise_comparisons.json']}
    rows=[];documents=[];supports=[];seen=set();checks=[];sources=dict(before['source_hashes'])
    for group in assignment['groups']:
        reviewer=group['reviewer'];folder=RUN/'audit/visual_review'/reviewer;csvpath=folder/csvnames[reviewer]
        with csvpath.open(newline='',encoding='utf-8-sig') as f:csvrows=list(csv.DictReader(f))
        assert len(csvrows)==96
        lookup={(r['request_id'],r.get('displayed_method_alias') or r.get('display_alias') or r['method_alias']):r for r in csvrows}
        for name in supportnames[reviewer]:
            p=folder/name
            supports.append({'path':str(p.relative_to(RUN)),'sha256':sha(p),'original_document':load(p) if p.suffix=='.json' else p.read_text(encoding='utf-8-sig')})
            sources[str(p.relative_to(RUN))]=sha(p)
        for rid in group['request_ids']:
            p=folder/(rid+'.json');doc=load(p);observations=doc.get('notes',doc.get('rows',doc.get('observations')))
            assert len(observations)==6
            metadata={k:v for k,v in doc.items() if k not in ('notes','rows','observations')}
            parent=next(s for s in group['parent_ids'] if s in rid)
            ratio=.4 if '__C40__' in rid else .2;noise=int(rid.rsplit('__n',1)[1]);assert noise in (0,1,6,7)
            pages={}
            for mode in ('colored','neutral'):
                page=gallery_pages[(rid,mode)];png=RUN/'phases/main/gallery'/mode/(rid+'.png');rel=str(png.relative_to(RUN));actual=sha(png)
                assert actual==page['sha256']==before['image_hashes'][rel]
                pages[mode]={'path':rel,'sha256':actual}
                checks.append({'request_id':rid,'reviewer':reviewer,'mode':mode,'path':rel,'expected_sha256':page['sha256'],'actual_sha256':actual,'status':'PASS','view_evidence_source':str(p.relative_to(RUN))})
            documents.append({'path':str(p.relative_to(RUN)),'sha256':sha(p),'original_document':doc})
            for original in observations:
                alias=original.get('displayed_method_alias') or original.get('display_alias') or original['method_alias']
                assert re.fullmatch(r'Method [1-6]',alias)
                key=(rid,alias);assert key not in seen;seen.add(key);cr=lookup[key]
                for k,v in original.items():
                    if k in cr:assert str(v)==cr[k],(reviewer,rid,alias,k)
                row={'request_id':rid,'parent_id':parent,'ratio':ratio,'noise_index':noise,'reviewer':reviewer,'display_alias':alias,'method_alias':alias,'model_label':labels[alias],
                     'observation_text':original.get('observation',''),
                     'organization_note':original.get('overall_chair_organization',original.get('organization',original.get('chair_organization',''))),
                     'missing_or_collapse_note':original.get('missing_or_collapsed_regions',original.get('missing_or_collapse','')),
                     'isolated_or_duplicate_note':original.get('isolated_or_duplicate_surfaces',''),
                     'C_incorporation_note':original.get('C_incorporation',original.get('C_visual_incorporation','')),
                     'uncertainty_note':original.get('uncertainty',original.get('limitation','')),
                     'common_observation':original.get('common_observation',''),'condition_note':doc.get('condition_note',''),'within_request_comparison':doc.get('within_request_comparison',''),
                     'colored_png_path':pages['colored']['path'],'colored_png_sha256':pages['colored']['sha256'],'neutral_png_path':pages['neutral']['path'],'neutral_png_sha256':pages['neutral']['sha256'],
                     'initial_notes_saved_before_unblinding':True,'initial_colored_and_neutral_actual_view_evidence':True,'all_five_views_reported_reviewed':True,'human_rating':False,
                     'initial_source_json':str(p.relative_to(RUN)),'initial_source_json_sha256':sha(p),'initial_source_csv':str(csvpath.relative_to(RUN)),'initial_source_csv_sha256':sha(csvpath),
                     'original_observation_fields':original,'original_csv_fields':cr,'original_request_metadata':metadata}
                for k,v in original.items():row['initial_'+k]=v
                rows.append(row)
    assert len(rows)==384 and len(seen)==384 and len(documents)==64 and len(checks)==128
    assert len({(p['request_id'],p['mode']) for p in checks})==128
    assert len(Counter(r['model_label'] for r in rows))==6 and all(v==64 for v in Counter(r['model_label'] for r in rows).values())
    assert Counter(r['reviewer'] for r in rows)=={g['reviewer']:96 for g in assignment['groups']}
    assert Counter(r['ratio'] for r in rows)=={.2:192,.4:192}
    rows.sort(key=lambda r:(r['parent_id'],r['ratio'],r['noise_index'],int(r['display_alias'].split()[-1])))
    summaries=load(OUT/'parent_summary_source.json');assert len(summaries)==16
    assert {(s['parent_id'],s['ratio']) for s in summaries}=={(r['parent_id'],r['ratio']) for r in rows}
    for s in summaries:
        subset=[r for r in rows if r['parent_id']==s['parent_id'] and r['ratio']==s['ratio']]
        assert len(subset)==24 and {r['noise_index'] for r in subset}=={0,1,6,7}
        s['request_ids']=sorted({r['request_id'] for r in subset})
        s['initial_notes_sources']=sorted({r['initial_source_json'] for r in subset})
        s['summary_scope']='Descriptive synthesis of saved initial reviewer notes; no new image rating, scalar score or global rank.'
    payload={'schema':'complete_main_initial_visual_review_merge_v1','status':'COMPLETE','created_utc':datetime.now(timezone.utc).isoformat(),
             'initial_review_scope':'Four reviewers actually inspected assigned original pages; merging agent did not newly re-review all pages.',
             'unblinding':'Initial notes and image hashes verified; pre_unblind_verification.json saved before mapping lookup. All original text is preserved verbatim.',
             'mapping':mapping,'rows':rows,'original_request_documents':documents,'original_reviewer_support_documents':supports,
             'human_study':False,'new_scores':False,'new_geometry_metrics':False,'secondary_rechecks_in_initial_coverage':False}
    csv_write('per_output.csv',rows);dump('per_output.json',payload)
    csv_write('per_parent_noise_summary.csv',summaries);dump('per_parent_noise_summary.json',summaries)
    noise_notes=[]
    for s in summaries:
        subset=[r for r in rows if r['parent_id']==s['parent_id'] and r['ratio']==s['ratio']]
        noise_notes.append({'parent_id':s['parent_id'],'ratio':s['ratio'],'reviewer':s['reviewer'],'methods':[
            {'display_alias':a,'model_label':labels[a],'noise_observations':[{'noise_index':r['noise_index'],'request_id':r['request_id'],'original_observation':r['original_observation_fields']} for r in subset if r['display_alias']==a]} for a in sorted(labels)]})
    dump('per_parent_noise_original_notes.json',noise_notes);dump('image_hash_verification.json',checks)
    text=['# Main initial visual review / 主阶段初始视觉审阅','',
          '完整覆盖：8父 × 2比例 × 4噪声 × 6方法 = 384唯一输出；64请求、128原PNG（彩色/中性）。四位审阅者每人实际查看分配的32图，每图有四个实心视角及线框。合并者复核记录与哈希，不声称重新独自查看全部128图。','',
          'Complete coverage: 8 parents × 2 ratios × 4 noises × 6 methods = 384 unique outputs; 64 requests and 128 original colored/neutral PNGs. Each reviewer actually inspected its assigned 32 pages and five views per page. The merger verifies records/hashes, not a new personal inspection of all 128 pages.','',
          '初始记录先保存，再在本次合并中解盲。Method 1–6是规范显示行名；原文A–F保持原样，仅对应行1–6。JSON及CSV的initial_* / original_*保留全部原字段和措辞。未设分数、综合排名或显著性。开发背景已保留，本材料不是独立盲法人类研究。','',
          'Initial notes precede mapping access. Displayed Method 1–6 is canonical; original A–F text is retained as positional spelling only. JSON and initial_*/original_* CSV fields preserve every original field and text. No scores, composite ranks or significance claims were added. Prior implementation context is disclosed; this is not an independent blinded human study.','',
          '| Display | Model |','|---|---|']
    text.extend(f'| {a} | {labels[a]} |' for a in sorted(labels))
    text.extend(['','Files / 文件：','',
                 '- per_output.csv / .json: all 384 records; JSON also includes complete original request/support documents.',
                 '- per_parent_noise_summary.csv / .json and PARENT_NOISE_SUMMARY.md: all 16 parent/ratio descriptions.',
                 '- per_parent_noise_original_notes.json: all four noises for each method/condition, with original text.',
                 '- coverage.json, pre_unblind_verification.json, image_hash_verification.json, source_hashes.json: provenance and coverage.','',
                 '排除：root/initial_notes.json只含初始5请求，不重复计入；secondary复查不计初始覆盖。schedule40_train/review_manifest.json保留旧分配状态，以实际逐请求记录及observations.json COMPLETE认证完成。gallery旧PENDING不改写，由本目录独立收据记录完成。','',
                 'Excluded: root/initial_notes.json is an obsolete five-request partial; secondary rechecks are not counted. The train review_manifest retains its old assignment status, while completed request records and observations.json establish completion. Original gallery PENDING flags remain unchanged; this separate receipt records actual review completion.','',
                 '限制：图片不能认证接缝、无相交/重复、精确C保持或语义用途。C自带尖片/缺口不自动归因于生成器。开放背板或不同椅型不按GT重建误差判坏。所有噪声和不利尾部保留。','',
                 'Limits: pictures do not certify attachment, absence of intersections/duplicates, exact C retention or semantic use. Pre-existing input spikes/gaps are not automatically generator defects. Open backs and different chair styles are not penalized by source reconstruction. All noises and unfavorable tails remain included.'])
    with (OUT/'README.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    lines=['# Per-parent, across-noise descriptions / 逐父跨噪声描述','','Derived only from saved initial notes. Displayed method labels remain unchanged; see README for mapping. No global rank or scalar score.','']
    for s in sorted(summaries,key=lambda x:(x['parent_id'],x['ratio'])):
        lines.extend([f"## {s['parent_id']} — C{round(s['ratio']*100)}",'',s['summary_zh'],'',s['summary_en'],'',f"Reviewer: {s['reviewer']}; noises: n0, n1, n6, n7.",''])
    with (OUT/'PARENT_NOISE_SUMMARY.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    assert {p:sha(REPO/p) for p in frozen}==frozen
    for rel,h in before['source_hashes'].items():assert sha(RUN/rel)==h
    for rel,h in before['image_hashes'].items():assert sha(RUN/rel)==h
    for p in (assignment_path,mapping_path,gallery_path,OUT/'parent_summary_source.json',Path(__file__),before_path):sources[str(p.relative_to(RUN))]=sha(p)
    dump('source_hashes.json',{'input_and_merge_source_hashes':sources,'frozen_repository_sources_count':len(frozen),'frozen_repository_before_after_status':'PASS','frozen_repository_sources':frozen})
    products={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.is_file()}
    coverage={'schema':'complete_main_initial_visual_coverage_v1','status':'COMPLETE','created_utc':datetime.now(timezone.utc).isoformat(),
              'expected_and_actual':{'parents':8,'ratios':2,'noise_indices':[0,1,6,7],'methods':6,'unique_outputs':384,'requests':64,'original_pngs':128},
              'by_reviewer':dict(Counter(r['reviewer'] for r in rows)),'by_model':dict(Counter(r['model_label'] for r in rows)),'by_ratio':dict(Counter(str(r['ratio']) for r in rows)),
              'all_128_original_png_hashes':'PASS','all_384_initial_csv_vs_request_notes':'PASS',
              'actual_view_evidence':'All four reviewers report original colored/neutral pages and five views in completed per-request/coverage evidence.',
              'pre_unblind_receipt_sha256':sha(before_path),'mapping_sha256':sha(mapping_path),'source_text_changed':False,'source_images_changed':False,
              'frozen_sources_verified':len(frozen),'frozen_source_status':'PASS','root_partial_counted':False,'secondary_rechecks_counted':False,
              'human_study':False,'new_scores':0,'model_calls':0,'GPU_dispatches':0,'new_OT':0,'new_noise':0,
              'warnings':['Root obsolete five-request initial_notes.json excluded.','Train assignment manifest status predates complete request/observations evidence.','Original gallery PENDING flags retained; this receipt independently records completed review.','Original A-F text preserved; canonical columns use displayed Method 1-6.'],
              'products':products,'wall_seconds':time.perf_counter()-tick}
    dump('coverage.json',coverage)
    print(json.dumps({'status':'COMPLETE','outputs':384,'requests':64,'original_pngs':128,'frozen_sources':len(frozen),'csv_sha256':sha(OUT/'per_output.csv'),'json_sha256':sha(OUT/'per_output.json'),'coverage_sha256':sha(OUT/'coverage.json'),'wall_seconds':coverage['wall_seconds']}))
if __name__=='__main__':main()
