"""Static local gallery navigation. Metadata-only; no rendering or scoring.

AI review receipts are independent evidence. A historical PENDING string in a
gallery manifest is retained as provenance, never interpreted as proof that the
images have not been viewed. Exact image paths/hashes associate review receipts
with galleries; unrelated or only label-related reviews do not clear PENDING.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import quote
from . import assets

OUT=assets.OUT


def _all_strings(value):
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for v in value.values():yield from _all_strings(v)
    elif isinstance(value,list):
        for v in value:yield from _all_strings(v)


def _reviews(out):
    values=[]
    paths=set((out/'audit').glob('*.json'))|set((out/'visuals/reviews').glob('*.json'))|set((out/'decisions').glob('*visual*.json'))
    for path in sorted(paths):
        data=assets.read_json(path);rows=data.get('rows',[])
        marker=data.get('reviewer_type')=='AI_VISUAL_REVIEW' or data.get('review_type')=='AI_VISUAL_REVIEW' or \
            any(r.get('review_type')=='AI_VISUAL_REVIEW' for r in rows if isinstance(r,dict))
        if not marker:continue
        actual=[r for r in rows if isinstance(r,dict) and r.get('actually_viewed') is True]
        labels=sorted({str(r[k]) for r in rows if isinstance(r,dict) for k in ('candidate_label','matched_label') if r.get(k)})
        for key in ('candidate_label','matched_label','comparison_reference'):
            if data.get(key) and data[key] not in labels:labels.append(data[key])
        values.append(dict(path=str(path),sha256=assets.digest(path),status=data.get('status','RECORDED'),
            reviewer_type='AI_VISUAL_REVIEW',scope=data.get('scope',data.get('review_scope','See receipt')),
            labels=labels,explicit_actually_viewed_request_ids=sorted({r['request_id'] for r in actual if r.get('request_id')}),
            explicit_actually_viewed_parents=sorted({r['uid'] for r in actual if r.get('uid')}),
            human_usability=data.get('human_usability','NOT_RUN'),_strings=set(_all_strings(data))))
    return values


def _steps(labels):
    rows=[]
    for label in labels:
        if label=='START':rows.append(dict(label=label,feedback_step=0,conditional_total_step=5000))
        else:
            match=re.search(r'_(\d+)$',label);step=int(match.group(1)) if match else None
            rows.append(dict(label=label,feedback_step=step,conditional_total_step=5000+step if step is not None else None))
    return rows


def _matches(review,manifest,path,out):
    images={str(assets.resolve(x['path'])):x.get('sha256') for x in manifest.get('parent_pages',[]) if x.get('path')}
    if manifest.get('contact_sheet'):images[str(assets.resolve(manifest['contact_sheet']))]=manifest.get('contact_sheet_sha256')
    matched=[]
    for value in review['_strings']:
        if value in set(v for v in images.values() if v):matched.append(dict(kind='EXACT_RECORDED_IMAGE_SHA256',value=value));continue
        if any(value.lower().endswith(suffix) for suffix in ('.png','.html','.json')):
            try:
                p=assets.resolve(value,out)
                if str(p) in images or p==path or p==assets.resolve(manifest['index']):
                    matched.append(dict(kind='EXPLICIT_GALLERY_ARTIFACT_PATH',value=value))
            except (OSError,ValueError):pass
    return matched


def run(out=OUT):
    out=assets.resolve(out);folder=out/'visuals';folder.mkdir(parents=True,exist_ok=True)
    reviews=_reviews(out);galleries=[];unfinished=[];sources={}
    for path in sorted(folder.glob('*/manifest.json')):
        manifest=assets.read_json(path)
        if manifest.get('status')!='COMPLETE':
            unfinished.append(dict(path=str(path),status=manifest.get('status'),labels=manifest.get('labels',[])));continue
        pages=manifest.get('parent_pages',[]);cameras=manifest.get('cameras',{})
        views=sorted(next(iter(cameras.values()),{}))
        evidence=[]
        for review in reviews:
            matches=_matches(review,manifest,path,out)
            if matches:evidence.append({**{k:v for k,v in review.items() if k!='_strings'},'matching_evidence':matches})
        related=[{k:v for k,v in r.items() if k!='_strings'} for r in reviews
                 if set(r['labels'])&set(manifest['labels']) and not any(e['path']==r['path'] for e in evidence)]
        ids=sorted({rid for r in evidence for rid in r['explicit_actually_viewed_request_ids']})
        row=dict(id=path.parent.name,labels=manifest['labels'],steps=_steps(manifest['labels']),split=manifest.get('split'),
            manifest_path=str(path),manifest_sha256=assets.digest(path),gallery_status=manifest['status'],
            requests=manifest.get('requests'),parents=manifest.get('parents'),all16=manifest.get('requests')==16 and manifest.get('parents')==8,
            view_names=views,four_views_plus_wire=bool(len([v for v in views if v!='wire'])==4 and 'wire' in views),
            overlay_status=manifest.get('feedback_overlay_status'),overlay_panels=manifest.get('overlay_panels'),
            expected_overlay_panels=manifest.get('expected_overlay_panels'),
            contact_sheet=manifest['contact_sheet'],contact_sheet_sha256=manifest.get('contact_sheet_sha256'),
            index=manifest['index'],parent_pages=[dict(path=x['path'],sha256=x.get('sha256'),exists=assets.resolve(x['path']).exists()) for x in pages],
            human_usability='NOT_RUN',original_manifest_AI_field=manifest.get('AI_VISUAL_REVIEW'),
            AI_review_evidence_status='MATCHED_REVIEW_EVIDENCE' if evidence else 'PENDING',
            AI_explicit_request_coverage_count=len(ids),AI_explicit_request_ids=ids,AI_review_receipts=evidence,
            related_other_gallery_reviews=related,
            review_caveat='Receipt scope is authoritative; evidence of a contact-sheet or partial-page review is not automatically a complete four-view/overlay review.')
        galleries.append(row);sources[str(path)]=row['manifest_sha256']
    representatives=[]
    for path in sorted((folder/'representatives').glob('*/selection_manifest.json')):
        data=assets.read_json(path)
        representatives.append(dict(path=str(path),sha256=assets.digest(path),status=data.get('status'),
            candidate_label=data.get('candidate_label'),matched_label=data.get('matched_label'),
            index=data.get('index'),paired_csv=data.get('paired_csv'),registered_pairs=data.get('registered_pairs'),
            selection_is_descriptive=True,human_usability='NOT_RUN'))
        sources[str(path)]=assets.digest(path)
    for review in reviews:sources[review['path']]=review['sha256']
    tables=[]
    for path in sorted((out/'evaluation').glob('*/DEV/per_output_metrics.csv')):
        label=path.parents[1].name
        tables.append(dict(label=label,pool='DEV',per_output=str(path),per_parent=str(path.with_name('per_parent_metrics.csv')),
            per_noise=str(path.with_name('per_noise_metrics.csv')),summary=str(path.with_name('summary.json'))))
    report=out/'CONTROL_FEEDBACK_AUTONOMOUS_REPORT.md'
    value=dict(version='chair_feedback_local_view_catalog_v1',status='SNAPSHOT',
        created_utc=datetime.now(timezone.utc).isoformat(),completed_galleries=galleries,unfinished_galleries=unfinished,
        representative_selections=representatives,all_AI_review_receipts=[{k:v for k,v in r.items() if k!='_strings'} for r in reviews],
        per_checkpoint_tables=tables,report=dict(path=str(report),exists=report.exists()),source_files=sources,
        human_usability='NOT_RUN',new_model_calls=0,new_render_calls=0,new_scoring_calls=0,
        script_sha256=assets.digest(__file__),interpretation='Navigation only; this page neither performs visual review nor changes existing review judgments or research gates.')
    catalog=folder/'gallery_catalog.json';catalog.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    index=folder/'INDEX.html'
    def link(path,label):
        if not path:return '<span class="muted">NA</span>'
        p=assets.resolve(path,out)
        if not p.exists():return '<span class="muted">'+html.escape(label)+' — PENDING</span>'
        relative=os.path.relpath(p,folder).replace(os.sep,'/')
        return '<a href="'+html.escape(quote(relative,safe='/'),quote=True)+'">'+html.escape(label)+'</a>'
    esc=lambda x:html.escape(str(x))
    body=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Chair control · 本地图库索引</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1500px;margin:30px auto;padding:0 20px;color:#243247;background:#f7f9fc}h1{margin-bottom:8px}h2{margin-top:30px}a{color:#12617a;text-underline-offset:3px}table{width:100%;border-collapse:collapse;background:white}td,th{vertical-align:top;text-align:left;border-bottom:1px solid #d9e0e8;padding:12px}th{background:#eaf0f5}.muted,small{color:#607085}.labels{font:13px/1.65 ui-monospace,monospace;overflow-wrap:anywhere}details{margin:8px 0}.tag{display:inline-block;background:#e2edf2;padding:2px 7px;border-radius:3px;font-size:13px}ul{padding-left:20px}nav{background:white;border:1px solid #d9e0e8;padding:14px}</style></head><body>',
        '<h1>Chair control 本地图库</h1><p>完整已生成图库的短入口。这里只读取清单并链接现有文件，没有新渲染、评分或模型调用。</p>',
        '<nav>'+link(report,'最终报告')+' · '+link(out/'manifests/DEV.json','固定 DEV 16 例登记')+' · '+
        link(folder/'gallery_catalog.json','机器可读图库目录')+' · '+link(out/'derived/attribution/DEV/latest.json','逐父归因索引')+'</nav>',
        '<p><strong>Human usability：NOT_RUN。</strong>AI 审阅记录单独列示；旧图库中写着 PENDING，不代表后来没有看图。联系表审阅、部分页面审阅和完整多视角审阅按各收据的实际范围区分。</p>',
        '<h2>已完成图库</h2><table><tr><th>模型与步数</th><th>固定范围</th><th>打开图集</th><th>独立 AI 审阅证据</th></tr>']
    for row in sorted(galleries,key=lambda x:(-len(x['labels']),x['id'])):
        labels='<br>'.join(esc(s['label'])+'<br><small>新增步数 '+esc(s['feedback_step'])+'；累计 '+esc(s['conditional_total_step'])+'</small>' for s in row['steps'])
        scope=esc(row['requests'])+' 例 / '+esc(row['parents'])+' 父 / 2 固定噪声<br>'+esc(', '.join(row['view_names']))+'<br>S/I/E overlay：'+esc(row['overlay_status'])+' ('+esc(row['overlay_panels'])+'/'+esc(row['expected_overlay_panels'])+')'
        opening=link(row['contact_sheet'],'所有 16 例联系表')+'<br>'+link(row['index'],'完整多视角图库')+'<br>'+link(row['manifest_path'],'来源清单')
        opening+='<details><summary>逐父页面 / 反馈点叠图</summary>'+'<br>'.join(link(p['path'],Path(p['path']).stem) for p in row['parent_pages'])+'</details>'
        review='<span class="tag">'+esc(row['AI_review_evidence_status'])+'</span>'
        if row['AI_review_receipts']:
            review+='<ul>'
            for r in row['AI_review_receipts']:
                review+='<li>'+link(r['path'],Path(r['path']).name)+'<br><small>收据状态：'+esc(r['status'])+'；'+esc(r['scope'])+'</small></li>'
            review+='</ul>'
        else:review+='<p class="muted">尚无与此图库图像路径或哈希匹配的收据。</p>'
        if row['related_other_gallery_reviews']:
            review+='<details><summary>相关模型的其他图库审阅（不等于此图已审）</summary>'+'<br>'.join(link(r['path'],Path(r['path']).name) for r in row['related_other_gallery_reviews'])+'</details>'
        body.append('<tr><td class="labels">'+labels+'</td><td>'+scope+'</td><td>'+opening+'</td><td>'+review+'</td></tr>')
    body.append('</table><h2>代表图选择</h2><p>代表图是描述性选例，不能替代完整 16 例图集或原成功门槛。</p>')
    if representatives:
        body.append('<ul>')
        for r in representatives:body.append('<li>'+esc(r['candidate_label'])+' vs '+esc(r['matched_label'])+'：'+link(r['index'],'代表图入口')+' · '+link(r['paired_csv'],'全部配对 CSV')+' · '+link(r['path'],'选择依据')+'</li>')
        body.append('</ul>')
    else:body.append('<p class="muted">当前没有已存在的正式代表图 selection_manifest；本导航不会自行选例。</p>')
    body.append('<h2>逐例与汇总数据</h2><table><tr><th>Checkpoint</th><th>逐例</th><th>逐父</th><th>逐噪声 / 摘要</th></tr>')
    for r in tables:body.append('<tr><td class="labels">'+esc(r['label'])+'</td><td>'+link(r['per_output'],'16 例 CSV')+'</td><td>'+link(r['per_parent'],'8 父 CSV')+'</td><td>'+link(r['per_noise'],'2 噪声 CSV')+' · '+link(r['summary'],'JSON 摘要')+'</td></tr>')
    body.append('</table><h2>所有独立 AI 审阅记录</h2><ul>')
    for r in reviews:body.append('<li>'+link(r['path'],Path(r['path']).name)+' — '+esc(r['status'])+'<br><small>'+esc(r['scope'])+'</small></li>')
    body.append('</ul><p class="muted">快照 '+esc(value['created_utc'])+'。链接指向本地现有文件，无服务器、外网脚本或图像重采样。完成最终分析后可重新生成此目录。</p></body></html>')
    index.write_text('\n'.join(body),encoding='utf-8')
    print(json.dumps(dict(status='SNAPSHOT_COMPLETE',galleries=len(galleries),review_receipts=len(reviews),
        representative_selections=len(representatives),index=str(index),catalog=str(catalog),model_calls=0,render_calls=0,score_calls=0)))
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(OUT));args=parser.parse_args();run(args.out)
