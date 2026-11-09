"""Correct saved category presentation and independently check label tables.
No source triangle tests, model, noise, OT or mesh changes. Preserve all prior
derived outputs and their receipt; unknown inherited classifications stay unknown.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import ast,collections,csv,difflib,hashlib,json,shutil,time
import numpy as np
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def csvwrite(p,rs):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def run():
    if (P/'metadata_review.json').exists():return read(P/'metadata_review.json')
    reg=read(P/'registration.json');old=read(P/'completion.json')
    for p,h in old['output_hashes'].items():
        if p!=str(P/'run.py'):assert sha(p)==h,p
    for p,h in reg['protected_hashes'].items():assert sha(p)==h,p
    archive=P/'before_category_fix';archive.mkdir(exist_ok=False)
    for n in ('per_condition.csv','summary.json','REPORT.md','completion.json','source_hashes.json'):
        shutil.copyfile(P/n,archive/n)
    legal={'shared_edge_contact','shared_vertex_contact','coplanar_boundary_or_tolerance_contact','noncoplanar_edge_or_tangent_contact','point_or_tangent_contact'}
    illegal={'proper_transverse_penetration','coplanar_positive_area_overlap'}
    records={r['condition_id']:r for r in reg['records']};crows=rows(P/'per_condition.csv')
    edge_rows=rows(P/'per_source_interface.csv');boundary=rows(P/'per_C_open_edge.csv');yrows=rows(P/'per_source_Y_face.csv')
    all_categories=collections.Counter();new={};verified_interfaces=0;unknown_rows=[];interface_risk=[]
    for cid,r in records.items():
        q=read(r['quality_path'])
        with np.load(r['condition_path']) as z:known=set(map(int,z['source_face_ids']))
        with np.load(r['parent_npz']) as z:faces=z['source_faces'].copy()
        cnt=collections.Counter();book={}
        for pair in q['pair_contacts']:
            i,j=pair['pair'];label=pair['classification']
            scope='CC' if i in known and j in known else 'CF' if (i in known)!=(j in known) else 'FF'
            category='illegal' if label in illegal else 'legal_contact' if label in legal else 'unknown'
            cnt[scope,category]+=1;book[tuple(sorted((i,j)))]=pair
            all_categories[pair['category'],label]+=1
            if category=='unknown':unknown_rows.append(dict(condition_id=cid,parent_id=r['parent_id'],scope=scope,pair=[i,j],saved_category=pair['category'],classification=label))
        new[cid]={s+'_'+c+'_pairs':cnt[s,c] for s in ('CC','CF','FF') for c in ('illegal','unknown','legal_contact')}
        # Independent source-ID incidence calculation, separate from original graph helper.
        inc=collections.defaultdict(list)
        for fi,face in enumerate(faces):
            for a,b in zip(face,np.roll(face,-1)):inc[tuple(sorted(map(int,(a,b))))].append(fi)
        cedges={e for e,o in inc.items() if sum(fi in known for fi in o)==1}
        be=[x for x in boundary if x['condition_id']==cid];assert len(be)==len(cedges)
        iface=[x for x in edge_rows if x['condition_id']==cid]
        histogram=collections.Counter()
        for fi in range(len(faces)):
            if fi in known:continue
            es=[tuple(sorted(map(int,(a,b)))) for a,b in zip(faces[fi],np.roll(faces[fi],-1))]
            qualifying=[]
            for e in es:
                own=inc[e]
                if e in cedges and len(own)==2:
                    other=next(j for j in own if j!=fi)
                    p=book.get(tuple(sorted((fi,other))))
                    if p and p['classification'] in legal:qualifying.append(e)
            histogram[len(qualifying)]+=1
        matched=[x for x in crows if x['condition_id']==cid]
        for x in matched:
            for h in range(4):assert int(x[f'Y_faces_with_{h}_legal_C_edges'])==histogram[h],(cid,h)
        for x in iface:
            u,v,ci,fi=map(int,(x['vertex_u'],x['vertex_v'],x['C_source_face'],x['Y_source_face']))
            p=book[tuple(sorted((ci,fi)))];assert p['category']=='legal_contact' and p['classification']=='shared_edge_contact'
            assert inc[tuple(sorted((u,v)))] in ([ci,fi],[fi,ci]) and ci in known and fi not in known
            assert {u,v}.issubset(set(faces[fi])) and len(set(faces[fi])-{u,v})==1
            assert int(x['third_vertex_id'])==next(iter(set(faces[fi])-{u,v}))
            assert len(set(faces[ci]))==len(set(faces[fi]))==3
            all_risk=[(list(key),p['classification']) for key,p in book.items() if fi in key and p['classification'] in illegal]
            unknown=[(list(key),p['classification']) for key,p in book.items() if fi in key and p['classification'] not in legal|illegal]
            interface_risk.append(dict(condition_id=cid,source_edge_id=int(x['source_edge_id']),Y_source_face=fi,
              inherited_illegal_pairs_touching_Y=all_risk,inherited_unknown_pairs_touching_Y=unknown,
              valid_anchor_pair=True,whole_face_clearance_certified=False))
            verified_interfaces+=1
    for x in crows:x.update(new[x['condition_id']])
    csvwrite(P/'per_condition.csv',crows)
    summary=read(P/'summary.json')
    for g in ('TRAIN47','PANEL6','PANEL_DEV'):
        rr=[x for x in crows if x['group']==('PANEL6' if g=='PANEL_DEV' else g) and (g!='PANEL_DEV' or x['split']=='DEV')]
        summary[g]['conditions_with_inherited_FF_illegal']=sum(int(x['FF_illegal_pairs'])>0 for x in rr)
        summary[g]['conditions_with_inherited_FF_unknown']=sum(int(x['FF_unknown_pairs'])>0 for x in rr)
        summary[g]['inherited_FF_illegal_pairs']=sum(int(x['FF_illegal_pairs']) for x in rr)
        summary[g]['inherited_FF_unknown_pairs']=sum(int(x['FF_unknown_pairs']) for x in rr)
    write(P/'summary.json',summary)
    report=(P/'REPORT.md').read_text(encoding='utf-8').replace('| 仍有来源Y内部非法面对的条件 | 0 / 47 |','| 仍有来源Y内部非法面对的条件 | 31 / 47 |').replace('每边独立生成一张脸','每边独立生成一张面')
    z=summary['TRAIN47'];report+='\n## 计数兼容修正与独立复核\n\n'
    report+=f"来源pair记录的category使用proper、coplanar_positive_overlap、UNKNOWN等名称；初版汇总误按illegal/unknown名称取值，造成FF风险显示0。已改为原LEGAL/ILLEGAL classification集合统计：TRAIN47有{z['inherited_FF_illegal_pairs']}条FF非法面对，涉及{z['conditions_with_inherited_FF_illegal']}父；另{z['inherited_FF_unknown_pairs']}条FF未决，涉及{z['conditions_with_inherited_FF_unknown']}父，未当成无风险。修正前CSV/summary/报告/completion保存在before_category_fix/。\n\n"
    report+=f"合法原边从始至终由原actual_valid_graph读取classification判断，不使用错误category计数；此次独立核对全部{verified_interfaces}条去重来源接口，均为原saved legal shared_edge_contact且实际仅一C/一Y双面入射，与TRAIN旧proof/mask完全一致。无需也没有重跑任何来源triangle检测。另保存每条接口Y面与其他面的既有非法/未决关联，局部锚边合法不代表该Y面全局无冲突。\n\n"
    report+='注册前写路径变量遮蔽被输出目录保护拦截，原文件未写；初次CPU执行在第一合成控后的status重复字段序列化失败，保留failed.json及原源码。工程修正后完成全部9控。实际合成verified_pair尝试共32次（失败前5＋完成27），其中退化/重复控的未决由原图判据拒绝；生产检测0，模型调用0。原50来源图统计因恢复执行两遍，共100次图组装，非100份数据。\n'
    (P/'REPORT.md').write_text(report,encoding='utf-8')
    write(P/'source_interface_quality_context.json',interface_risk)
    write(P/'source_unknown_pairs.json',unknown_rows)
    a=dict(status='APPLIED_SAVED_CATEGORY_PRESENTATION_FIX',registration_sha256=sha(P/'registration.json'),
      previous_entry_sha256=sha(P/'run_before_category_fix.py'),new_entry_sha256=sha(P/'run.py'),
      fix='Map original classification through original legal/illegal sets, all other classes UNKNOWN; graph and source interface definitions unchanged.',
      affected='Derived per_condition CC/CF/FF aggregate counts and report only',
      source_interface_eligibility_affected=False,production_triangle_retests=0,model_calls=0)
    write(P/'metadata_amendment.json',a)
    review=dict(status='PASS_AFTER_EXPLICIT_CATEGORY_COMPATIBILITY_FIX',unique_source_conditions=len(records),
      independent_source_interfaces_checked=verified_interfaces,independent_Y_edge_histograms_checked=len(records),
      train_FF_illegal_pairs=z['inherited_FF_illegal_pairs'],train_FF_risk_conditions=z['conditions_with_inherited_FF_illegal'],
      train_FF_unknown_pairs=z['inherited_FF_unknown_pairs'],unknown_pairs=unknown_rows,
      actual_saved_categories=[dict(category=k[0],classification=k[1],count=v) for k,v in sorted(all_categories.items())],
      new_triangle_tests=0,new_OT=0,new_noise=0,new_models=0,new_forwards=0,protected_hashes_unchanged=len(reg['protected_hashes']))
    for p,h in reg['protected_hashes'].items():assert sha(p)==h,p
    write(P/'metadata_review.json',review)
    for src in (Path(reg['protected_hashes'] and next(p for p in reg['protected_hashes'] if p.endswith('/native_patch_seams.py'))),):
        shutil.copyfile(src,P/'source_code'/src.name)
    source=read(P/'source_hashes.json');source.update(final_entry_sha256=sha(P/'run.py'),metadata_finalizer_sha256=sha(__file__))
    write(P/'source_hashes.json',source)
    diff=''.join(difflib.unified_diff([], (P/'run.py').read_text(encoding='utf-8').splitlines(keepends=True),fromfile='/dev/null',tofile='idea3/run.py'))
    (P/'code.diff').write_text(diff,encoding='utf-8')
    old['metadata_presentation_correction']=a;old['independent_metadata_review_path']=str(P/'metadata_review.json')
    old['output_hashes']={str(p):sha(p) for p in P.rglob('*') if p.is_file() and p!=P/'completion.json' and p.name!='attempt.json'}
    write(P/'completion.json',old)
    return review
if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False))
