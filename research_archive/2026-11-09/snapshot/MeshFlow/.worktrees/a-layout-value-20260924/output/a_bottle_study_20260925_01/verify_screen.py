from pathlib import Path
import argparse, collections, hashlib, json
import numpy as np
from experiments.a_radial.io_utils import array_hash, file_hash
from experiments.a_radial.exporter import RadialSupport
from experiments.a_radial.mesh_ops import ray_hits

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):
    with p.open('x',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def order(uids,salt): return sorted(uids,key=lambda u:(hashlib.sha256((salt+u).encode()).hexdigest(),u))
def run(screen, out):
    screen=Path(screen).resolve(); out=Path(out).resolve()
    complete=read(screen/'COMPLETE.json'); p=read(screen/'SCREEN_PROTOCOL.json'); s=read(screen/'SELECTION.json')
    rows=[json.loads(x) for x in (screen/'SCREEN_ROWS.jsonl').read_text().splitlines()]
    assert len(rows)==225 and len({r['uid'] for r in rows})==225
    assert [r['uid'] for r in rows]==p['processing_uids']==order(p['train_uids'],'a-condition16-parent-v1:')
    assert s['screened_count']==complete['screened_count']==225
    assert file_hash(screen/'SCREEN_PROTOCOL.json')==s['protocol_sha256']==complete['screen_protocol_sha256']
    assert file_hash(screen/'SELECTION.json')==complete['selection_sha256']
    assert file_hash(screen/'SCREEN_ROWS.jsonl')==s['screen_rows_sha256']
    assert file_hash(screen/'FILE_MANIFEST.json')==complete['file_manifest_sha256']
    for name,digest in read(screen/'FILE_MANIFEST.json')['files'].items(): assert file_hash(screen/name)==digest, name
    for name,digest in p['sources_sha256'].items(): assert file_hash(name)==digest,name
    assert file_hash(p['source_archive'])==p['source_archive_sha256']
    counts=dict(collections.Counter(r['status'] for r in rows)); assert counts==s['counts']==complete['counts']
    eligible=[r['uid'] for r in rows if r['status']=='eligible']; assert eligible==s['eligible_uids_hash_order']
    assert not set(p['old_development_uids'])&set(p['train_uids'])
    errors=sum(r['status']=='error' for r in rows)
    selected=eligible[:16] if len(eligible)>=16 and not errors else []
    assert selected==s['selected16'] and sorted(selected)==s['lex_sorted_uids']
    expected_status='SCREEN_RUNTIME_ERROR' if errors else ('SELECTED16' if selected else 'INSUFFICIENT_ELIGIBLE')
    assert s['status']==complete['status']==expected_status
    support=RadialSupport(); raw_field_checks=[]
    with np.load(screen/'TEMPLATE.npz',allow_pickle=False) as z:
        base=np.array(z['base_u']); faces=np.array(z['faces'])
    for uid in selected:
        b=s['data'][uid]
        assert file_hash(b['path'])==b['sha256'] and file_hash(b['source_path'])==b['source_sha256']
        with np.load(b['path'],allow_pickle=False) as z: d={k:np.array(z[k]) for k in z.files}
        assert d['logr'].shape==(2050,) and d['logr'].dtype==np.float32
        assert np.array_equal(d['base_u'],base) and d['base_u'].dtype==np.float64
        assert np.array_equal(d['faces'],faces) and d['faces'].shape==(512,3) and d['faces'].dtype==np.int32
        rv,rf=support.mesh(d['logr'])
        assert np.array_equal(rv,d['ref_vertices']) and np.array_equal(rf,d['ref_faces'])
        assert array_hash(d['logr'])==b['mainfield_hash'] and array_hash(rv,rf)==b['reference_hash']
        with np.load(b['source_path'],allow_pickle=False) as z: v=np.asarray(z['vertices'],np.float64); f=np.asarray(z['faces'],np.int64)
        c=(v.min(0)+v.max(0))/2; diag=np.linalg.norm(np.ptp(v,axis=0))
        hits,r=ray_hits((v-c)/diag,f,support.u)
        assert np.all(hits==1)
        y=np.log(r/np.sqrt(np.mean(r*r))).astype(np.float32)
        assert np.array_equal(y,d['logr']),uid
        raw_field_checks.append({'uid':uid,'raw_to_logr_bitwise':'PASS','source_faces':len(f)})
    du=order(selected,'a-condition16-donor-v1:')
    expected_donors={u:du[(i+1)%len(du)] for i,u in enumerate(du)} if du else {}
    assert s['donor_mapping']==expected_donors
    result={'status':'PASS','screen_result':s['status'],'screened':225,'eligible':len(eligible),'selected':len(selected),
      'runtime_errors':errors,'counts':counts,'selected_raw_to_field_checks':raw_field_checks,
      'eligible_source_faces_above512':sum(r['status']=='eligible' and r['faces']>512 for r in rows),
      'selected_source_faces_above512':sum(r['source_faces']>512 for r in raw_field_checks),
      'no_model_optimizer_or_MC_used_in_verifier':True,
      'script_sha256':file_hash(__file__),
      'inputs_sha256':{str(screen/n):file_hash(screen/n) for n in ('SCREEN_PROTOCOL.json','SCREEN_ROWS.jsonl','SELECTION.json','COMPLETE.json','FILE_MANIFEST.json')}}
    write(out/'SCREEN_VERIFICATION.json',result)
    names={'eligible':'合格独立输入','topology_reject':'拓扑拒绝','ray_reject':'径向单交拒绝','exact_reject':'精确相交/退化拒绝','source_face_count_reject':'原面数范围拒绝','exact_duplicate':'精确重复输入','old_mainfield_alias':'与旧开发输入精确重复','error':'运行异常'}
    lines=['# 瓶类225对象几何筛查\n',f"完整筛查225个原train父对象，合格{len(eligible)}个，选定{len(selected)}个；运行异常{errors}个。状态：{s['status']}。\n",
      '| 首个终止状态 | 对象数 |\n|---|---:|\n']
    lines += [f'| {names.get(k,k)} | {v} |\n' for k,v in counts.items()]
    lines += [f"\n几何适用率：{len(eligible)}/225 = {len(eligible)/225*100:.2f}%。这个分母与随后若开展的16对象质量交付率不同。\n",
      '输入保留原20<F<800；最终模板512面。没有要求源面数<=512，没有简化、补洞或改变原表面。只按源几何资格和预先固定精确重复规则筛选，未运行模型或优化效果筛查。\n',
      '全部225行、源SHA、各阶段检查及拒绝原因见SCREEN_ROWS.jsonl；选中对象由固定UID哈希顺序决定。若数量不足16或存在未解决运行错误，不进入正式144任务，也不放宽规则补足。\n',
      '独立回读验证了全部源绑定/输出哈希、225分母、固定选择与donor映射；若选出16，还从原始网格重新计算2050方向半径，逐位核对float32主场和参考曲面。\n',
      f"原筛选记录时间：{complete['elapsed_seconds']:.3f}秒。源网格→主场是离线数据准备，不能计作或冒称后续从原始资产开始的TTQ总时间。\n",
      '瓶类来自官方原train划分，但本轮不用它训练A；它是冻结3k的跨类别验证候选，不能称所有比较系统均未见的测试集。\n']
    with (out/'SCREEN_RESULT_zh.md').open('x',encoding='utf-8') as f: f.write('\n'.join(lines))
    print(json.dumps({k:result[k] for k in ('status','screen_result','screened','eligible','selected','runtime_errors')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--screen',required=True); p.add_argument('--out',required=True); a=p.parse_args(); run(a.screen,a.out)