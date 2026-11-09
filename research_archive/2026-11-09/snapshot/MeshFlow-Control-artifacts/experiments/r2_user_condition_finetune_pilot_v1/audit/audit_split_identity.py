"""Read-only split-identity supplement; no model, RNG, OT or GPU imports."""
from pathlib import Path
import hashlib,json,csv,time
ROOT=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1')
SOURCE=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1')
OUT=ROOT/'audit'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
start=time.perf_counter()
paths=dict(active=SOURCE/'active_conditions.json',inventory=SOURCE/'parent_inventory.json',old_regression=SOURCE/'old_regression_selection.json',historical_train_manifest=Path('/mnt/e/MeshFlow-Control-artifacts/data/train_manifest.json'),original_splits=Path('/mnt/e/MeshFlow/native_t1/runs/control_mvp48_v1/data/splits.json'),paired_plan=ROOT/'data/paired_plan.json')
before={str(p):digest(p) for p in paths.values()}
a,i,o,m,s,p=[load(paths[k]) for k in ('active','inventory','old_regression','historical_train_manifest','original_splits','paired_plan')]
sets=dict(current_train25={r['uid'] for r in a['rows'] if r['split']=='train'},new_DEV6={r['uid'] for r in a['rows'] if r['split']=='dev'},old_DEV8={r['uid'] for r in o['rows']},historical_R2_train26={r['uid'] for r in i['rows'] if r['split']=='train'},original_train_manifest32=set(m['splits']['train']),manifest_parent_rows32={r['uid'] for r in m['parents']},control_mvp48_train32=set(s['train']),control_mvp48_validation8=set(s['validation']))
expected=dict(current_train25=25,new_DEV6=6,old_DEV8=8,historical_R2_train26=26,original_train_manifest32=32,manifest_parent_rows32=32,control_mvp48_train32=32,control_mvp48_validation8=8)
assert {k:len(v) for k,v in sets.items()}==expected
assert sets['original_train_manifest32']==sets['manifest_parent_rows32']==sets['control_mvp48_train32']
assert sets['old_DEV8']==sets['control_mvp48_validation8']
assert sets['current_train25'] < sets['historical_R2_train26'] < sets['original_train_manifest32']
assert sets['new_DEV6'] < sets['old_DEV8']
assert {r['uid'] for r in p['records']['R2_CONT_OLD']}=={r['uid'] for r in p['records']['R2_CONT_USER']}==sets['current_train25']
intersections={}
group_intersections={}
for train in ('current_train25','historical_R2_train26','original_train_manifest32'):
 for dev in ('new_DEV6','old_DEV8'):
  key=train+'__INTERSECT__'+dev
  intersections[key]=sorted(sets[train]&sets[dev])
  group_intersections[key]=sorted({s['group_ids'][u] for u in sets[train]}&{s['group_ids'][u] for u in sets[dev]})
  assert not intersections[key] and not group_intersections[key]
intersections['new_DEV6__INTERSECT__old_DEV8']=sorted(sets['new_DEV6']&sets['old_DEV8'])
assert all(s['official_splits'][u]=='train' for u in sets['old_DEV8'])
after={str(q):digest(q) for q in paths.values()};assert before==after
result=dict(schema='r2_user_condition_finetune_split_identity_audit_v1',status='PASS',source_hashes=before,source_before_after_equal=True,counts=expected,uid_sets={k:sorted(v) for k,v in sets.items()},UID_intersections=intersections,registered_group_ID_intersections=group_intersections,excluded_original_train32_to_R2_26=sorted(sets['original_train_manifest32']-sets['historical_R2_train26']),excluded_R2_26_to_current25=sorted(sets['historical_R2_train26']-sets['current_train25']),excluded_old_DEV8_to_new_DEV6=sorted(sets['old_DEV8']-sets['new_DEV6']),DEV_official_split={u:s['official_splits'][u] for u in sorted(sets['old_DEV8'])},interpretation='25 current training parents and historical26/32 conditional-training parent sets are disjoint from both DEV sets at exact UID and the pre-existing group-ID level. The new6 DEV is a subset of old8, not six additional parents. 32 refers only to original training parents, excluding the separate8 validation parents. All8 DEV are official train split, repeatedly observed development evidence; this does not certify unseen pretraining, independent testing or absence of near-duplicate families beyond the registered groups.',new_model_calls=0,new_OT_calls=0,new_random_arrays=0,wall_seconds=time.perf_counter()-start)
save(OUT/'split_identity_audit.json',result)
with (OUT/'split_parent_membership.csv').open('w',encoding='utf-8-sig',newline='') as f:
 fields=['uid','registered_group_id','official_split','original_train32','historical_R2_train26','current_train25','old_DEV8','new_DEV6'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for u in sorted(sets['original_train_manifest32']|sets['old_DEV8']):
  w.writerow(dict(uid=u,registered_group_id=s['group_ids'][u],official_split=s['official_splits'][u],original_train32=int(u in sets['original_train_manifest32']),historical_R2_train26=int(u in sets['historical_R2_train26']),current_train25=int(u in sets['current_train25']),old_DEV8=int(u in sets['old_DEV8']),new_DEV6=int(u in sets['new_DEV6'])))
main=load(OUT/'data_pairing_audit.json');main['split_identity_audit']=result;main['split_identity_audit_path']=str(OUT/'split_identity_audit.json');main['split_identity_audit_sha256']=digest(OUT/'split_identity_audit.json');main['paired_K_comparison'].setdefault('equal',0)
save(OUT/'data_pairing_audit.json',main)
text='''\n实际ID交集另经原清单核验：`train_manifest.json`的32个parent行与`splits.train`、原`control_mvp48_v1/data/splits.json`的32个train完全相同；旧8 DEV恰好是原split中另列的8个validation，因此32不是train+DEV的总数。历史R2的26父是这32训练父的子集，当前25父再排除400a。当前25、历史26及原32三个训练集合，与新6/旧8两个DEV集合的全部6种UID交集及已有group-ID交集均为0。新6 DEV全部属于旧8 DEV，只排除了341b与c1b，因此两个评估面板在父对象上有6父重叠，不能当作14个独立父。旧8 DEV全部标记为官方train split；这里只证明登记条件训练划分隔离，不宣称官方预训练未见、独立测试或未登记近重复不存在。完整UID及排除集见 `split_identity_audit.json` 和 `split_parent_membership.csv`。\n\n8,000对对应槽位的K全部不同：USER有7,971对K更小、29对K更大；两臂共同排列种子下，实际完整面排列数组仅36对相同。固定seed规则与输入身份配对不意味着形状不同的free噪声、OT映射或排列数组逐字节相同。\n'''
with (OUT/'DATA_PAIRING_AUDIT.md').open('a',encoding='utf-8') as f:f.write(text)
print(json.dumps({k:result[k] for k in ['status','counts','UID_intersections','registered_group_ID_intersections','excluded_R2_26_to_current25','excluded_old_DEV8_to_new_DEV6']},indent=2))