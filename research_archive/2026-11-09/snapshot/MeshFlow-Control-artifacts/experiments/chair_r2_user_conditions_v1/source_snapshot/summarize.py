"""Tabulate and browse actual exported source-face conditions, CPU only."""
from pathlib import Path
from collections import Counter
import csv,io,html,json
from prepare import DEFAULT_OUT,read,write,sha,event

def text_file(path,text):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if path.read_text(encoding='utf-8-sig')!=text:raise RuntimeError('Refuse differing artifact '+str(path))
 else:path.write_text(text,encoding='utf-8')

def csv_file(path,rows):
 if not rows:raise ValueError('No rows')
 buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(rows[0]),lineterminator='\n')
 writer.writeheader()
 for r in rows:
  writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
 text_file(path,buf.getvalue())

def main(out=DEFAULT_OUT):
 out=Path(out);conds=read(out/'conditions.json');rows=conds['rows'];regions=read(out/'regions.json')['rows']
 parents=read(out/'parent_inventory.json')['rows']
 qq={r['condition_id']:r for r in read(out/'quality/condition_summary.json')['rows']}
 regmap={r['region_id']:r for r in regions}
 tables=[]
 for r in rows:
  qr=qq[r['condition_id']]['counts']
  base={k:r[k] for k in ('condition_id','uid','split','region_type','region_id','region_status','tier','requested_tiers','N','K','face_fraction','region_area_fraction','parent_area_fraction','C_area','R_area','M_area','source_face_ids','free_source_ids','npz','sha256','parent_sha256','source_partition_bitwise')}
  for part in ('CC','CF','FF'):
   base[part+'_confirmed_pairs']=qr[part]['confirmed_illegal_pair_count']
   base[part+'_unknown_pairs']=qr[part]['unknown_pair_count']
  base['review_kind']='AI_PROXY';base['R_definition']=regmap[r['region_id']]['reason']
  tables.append(base)
 csv_file(out/'tables/conditions.csv',tables)
 rtable=[]
 for r in regions:
  rtable.append(dict(uid=r['uid'],split=r['split'],region_type=r['region_type'],region_id=r['region_id'],
   status=r['status'],K_R=len(r['source_face_ids']),source_face_ids=r['source_face_ids'],reason=r['reason'],
   accepted_unique_conditions=sum(c['region_id']==r['region_id'] for c in rows),
   quality_at_R=r.get('quality_at_R',{}),review_kind='AI_PROXY'))
 csv_file(out/'tables/regions.csv',rtable)
 byparent=[]
 for p in parents:
  cc=[c for c in rows if c['uid']==p['uid']]
  q=read(out/'quality/parents'/f"{p['uid']}.json")
  # Parent totals inferred from any condition partition, without counting repeated pair occurrences.
  if cc:
   qr=qq[cc[0]['condition_id']]['counts']
   defects=sum(x['confirmed_illegal_pair_count'] for x in qr.values());unknown=sum(x['unknown_pair_count'] for x in qr.values())
  else:
   from quality import partition_condition
   qr=partition_condition(p['uid'],[0],out/'quality')['partitions']
   defects=sum(x['confirmed_illegal_pair_count'] for x in qr.values());unknown=sum(x['unknown_pair_count'] for x in qr.values())
  byparent.append(dict(uid=p['uid'],split=p['split'],N=p['N'],accepted_conditions=len(cc),
    accepted_seat_conditions=sum(c['region_type']=='seat' for c in cc),
    accepted_back_conditions=sum(c['region_type']=='back' for c in cc),
    parent_confirmed_pairs=defects,parent_unknown_pairs=unknown,
    included_in_new_pool=bool(cc),parent_npz=p['npz'],sha256=p['sha256']))
 csv_file(out/'tables/parents.csv',byparent)
 alias=[]
 for r in regions:
  for e in r.get('extent_preview',[]):
   for tier in e['requested_tiers']:
    alias.append(dict(uid=r['uid'],split=r['split'],region_type=r['region_type'],requested_tier=tier,
      actual_unique_tier=e.get('tier','NA'),K=e['K'],region_area_fraction=e.get('region_area_fraction',e.get('area_fraction')),
      note='Full dev extent previews are not extra exported DEV conditions'))
 csv_file(out/'tables/extent_aliases.csv',alias)
 train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
 accepted=[r for r in regions if r['status'].startswith('ACCEPTED')]
 tr=[r for r in accepted if r['split']=='train']
 summary=dict(train_parents=len({r['uid'] for r in train}),train_regions=len(tr),train_conditions=len(train),
   train_seat_regions=sum(r['region_type']=='seat' for r in tr),train_back_regions=sum(r['region_type']=='back' for r in tr),
   training_conditions_by_part=dict(Counter(r['region_type'] for r in train)),
   unique_training_conditions_by_tier=dict(Counter(r['tier'] for r in train)),
   deduplicated_training_tier_aliases=3*len(tr)-len(train),
   dev_conditions=len(dev),dev_parents=len({r['uid'] for r in dev}),old_regression_conditions=8,old_regression_requests=16,
   future_total_output_requests=3*(len(dev)*2+16),future_new_generation_if_old16_reusable=3*(len(dev)*2+16)-16,
   train_K_min=min(r['K'] for r in train),train_K_max=max(r['K'] for r in train),
   train_N_min=min(r['N'] for r in train),train_N_max=max(r['N'] for r in train),
   train_face_fraction_range=[min(r['face_fraction'] for r in train),max(r['face_fraction'] for r in train)],
   train_whole_area_fraction_range=[min(r['parent_area_fraction'] for r in train),max(r['parent_area_fraction'] for r in train)],
   dev_requested_tiers=dict(Counter(regmap[r['region_id']]['dev_requested_tier'] for r in dev)),
   rejected_regions=[dict(uid=r['uid'],split=r['split'],region_type=r['region_type'],status=r['status'],reason=r['reason']) for r in regions if not r['status'].startswith('ACCEPTED')],
   excluded_training_parents=[r['uid'] for r in byparent if r['split']=='train' and not r['included_in_new_pool']],
   source_CY_arrays=len(rows),source_CY_npz_bytes=sum(Path(r['npz']).stat().st_size for r in rows),
   training_updates=0,model_calls=0,OT_calls=0,new_Gaussian_draws=0,generated_full_meshes=0,
   route='R2 Native/T1+role+Geo, FM, full-course HYBRID=.25; new C pool replaces the old C20/C40 construction',
   semantic_status='AI_PROXY; data ready for review, not human certified or training executed')
 write(out/'data_summary.json',summary)
 esc=html.escape
 blocks=[]
 for r in tables:
  rel=r['condition_id']+'.png'
  meta=f"{r['split']} | {r['region_type']} | {'/'.join(r['requested_tiers'])} | K/N={r['K']}/{r['N']} | C/R={r['region_area_fraction']:.1%} | C/M area={r['parent_area_fraction']:.1%}"
  quality=' / '.join(f"{p} {r[p+'_confirmed_pairs']} + {r[p+'_unknown_pairs']}?" for p in ('CC','CF','FF'))
  ids=json.dumps(r['source_face_ids'])
  blocks.append(f'<article data-split="{r["split"]}" data-part="{r["region_type"]}" data-tier="{r["tier"]}" data-uid="{r["uid"]}"><h2>{esc(r["condition_id"])}</h2><p>{esc(meta)}<br>来源几何对数（确认 + 未决?）：{esc(quality)}<br>{esc(r["R_definition"])}</p><a href="conditions/{rel}"><img loading="lazy" src="conditions/{rel}" alt="{esc(r["condition_id"])}"></a><details><summary>原面ID及数据</summary><p>C IDs: {esc(ids)}</p><a href="../conditions/{r["condition_id"]}.npz">C / Y NPZ</a></details></article>')
 nas=''.join(f'<tr><td>{esc(r["uid"])}</td><td>{r["split"]} / {r["region_type"]}</td><td>{esc(r["status"])}</td><td>{esc(r["reason"])}</td><td><a href="proposals/{r["uid"]}__candidates.png">原模型</a></td></tr>' for r in summary['rejected_regions'])
 page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>R2 条件数据审阅</title>
<style>body{font:15px system-ui,sans-serif;margin:24px auto;max-width:1500px;padding:0 20px;color:#192b35;background:#f6f8fa}header,article{background:white;padding:18px 24px;border-radius:10px;margin:16px 0}h1{font-size:28px}h2{font-size:16px;overflow-wrap:anywhere}p{line-height:1.65}img{width:100%;height:auto}nav{position:sticky;top:0;background:#e5eff4;padding:12px;z-index:2}select,input{margin:0 12px 0 4px;padding:6px}table{border-collapse:collapse;width:100%;background:white;font-size:13px}td,th{border:1px solid #ddd;padding:9px;overflow-wrap:anywhere}details{padding:8px;background:#f5f7f9}[hidden]{display:none!important}a{color:#005b88}</style>
<header><h1>R2 用户式条件数据审阅</h1><p>25个训练父 / 47个区域 / 122条去重条件；7条新DEV条件。所有图均为来源数据划分，不是模型生成结果。两行是同一对象的两个视角；每张图从左到右为 M｜R｜C｜Y｜C+Y。蓝色为输入C，橙色为来源补集Y。每对象各栏共享相机和尺度。</p><p>AI_PROXY视觉核对，不等于人工语义标注。完整（full）指声明的区域R；原网格缺陷未修复。数值分开列CC／CF／FF，“0”不认证几何完美；“?”是未决。后续训练配方保持R2。</p><a href="../DATA_PREPARATION_REPORT.md">报告</a> · <a href="../tables/conditions.csv">逐条件CSV</a> · <a href="../tables/regions.csv">区域与NA表</a> · <a href="../dataset_handoff.json">后续数据合同</a></header>
<nav>集合<select id="s"><option value="">全部</option><option>train</option><option>dev</option></select>区域<select id="p"><option value="">全部</option><option>seat</option><option>back</option></select>独立档位<select id="t"><option value="">全部</option><option>small</option><option>medium</option><option>full</option></select>父ID<input id="q" placeholder="如 12c2"><b id="n"></b></nav>'''
 page+=''.join(blocks)+f'<header><h2>未采用区域：完整记录</h2><table><tr><th>父ID</th><th>集合／区域</th><th>状态</th><th>原因</th><th>查看</th></tr>{nas}</table></header>'
 page+='''<script>const controls=["s","p","t","q"].map(x=>document.getElementById(x));function filter(){let[s,p,t,q]=controls.map(x=>x.value.toLowerCase());let n=0;for(let a of document.querySelectorAll("article")){a.hidden=!!((s&&a.dataset.split!==s)||(p&&a.dataset.part!==p)||(t&&a.dataset.tier!==t)||(q&&!a.dataset.uid.includes(q)));if(!a.hidden)n++;}document.getElementById("n").textContent=n+" 条";}controls.forEach(x=>x.addEventListener("input",filter));filter();</script></html>'''
 text_file(out/'gallery/all_exported.html',page)
 event(out,'TABLES_AND_GALLERY_INDEX_WRITTEN',conditions=len(rows),csv_tables=4,model_calls=0)
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
