"""Verify retained files against pre-cleanup metadata, archive identity and source counts."""
from pathlib import Path
import csv,json,hashlib,collections,os
ROOT=Path('E:/MeshFlow').resolve();OUT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

rows=list(csv.DictReader((OUT/'file_inventory_before.csv').open(encoding='utf8')))
removed=json.loads((OUT/'cache_deletion_candidates.json').read_text())
deleted={r['path'] for r in removed};changed=[];missing=[];retained=0
for row in rows:
 p=Path(row['path'])
 if row['path'] in deleted:
  if p.exists():raise RuntimeError('Cache still exists: '+str(p))
  continue
 if p.is_relative_to(OUT):continue
 if not os.path.lexists(p):missing.append(str(p));continue
 s=p.stat(follow_symlinks=False)
 if (s.st_size,s.st_mtime_ns)!=(int(row['bytes']),int(row['mtime_ns'])):changed.append(str(p))
 retained+=1
if changed or missing:raise RuntimeError(json.dumps(dict(changed=changed,missing=missing)))
chair=ROOT/'experiments/mf_h1_local0/assets/shapenet-03001627'
fs=sorted((chair/'objaverse_occ_v5_ids').glob('*.npz'))
assert len(fs)==3135 and len({p.stem for p in fs})==3135
# Existing exact-geometry count is historical, not claimed as a new geometry computation.
old=json.loads((ROOT/'native_t1/maintenance/temp_cleanup_20261002_160601/dataset_scan.json').read_text())
ready=json.loads((ROOT/'native_t1/runs/control_mvp48_v1/DATA_READY.json').read_text())
archive=ROOT/'native_t1/runs/chair_control_attribution_v1/CHAIR_CONTROL_ATTRIBUTION_REVIEW.zip'
expected=(archive.with_name(archive.name+'.sha256')).read_text().split()[0]
assert sha(archive)==expected
dataset_hash=hashlib.sha256()
for p in fs:
 dataset_hash.update(p.name.encode());dataset_hash.update(bytes.fromhex(sha(p)))
s=json.loads((OUT/'inventory_summary.json').read_text())
summary=dict(status='COMPLETE',deleted_files=len(deleted),deleted_bytes=sum(r['bytes'] for r in removed),remaining_before_inventory_files_unchanged_metadata=retained,unexpected_changed_files=changed,unexpected_missing_files=missing,
 current_chair_UIDs=len(fs),chair_mesh_bytes=sum(p.stat().st_size for p in fs),chair_with_splits_bytes=s['chair_source_bytes'],chair_content_aggregate_sha256=dataset_hash.hexdigest(),
 historical_exact_chair_geometries=old['datasets']['chair']['summary']['exact_unordered_triangle_geometry_count'],historical_exact_geometry_scope='2026-10-02 scan, not newly recomputed',
 prepared_parent_objects=ready['object_count'],prepared_parent_split=ready['splits'],original_chair_train_count=2822,original_chair_test_count=313,
 dataset_notes={k:dict(files=v['summary']['files'],unique_uids=v['summary']['unique_uids'],exact_geometries=v['summary']['exact_unordered_triangle_geometry_count'],scope='Historical scan; file sizes/mtimes verified unchanged in this cleanup') for k,v in old['datasets'].items()},
 project_logical_bytes_before=s['bytes'],checkpoint_pt_bytes=s['extensions']['.pt']['bytes'],git_metadata_bytes=next(r['bytes'] for r in s['top_directories'] if r['directory']=='.git'),
 original_attribution_zip_sha256=expected,original_attribution_zip_unchanged=True,protected_data_weights_raw_reports_deleted=0,
 exclusions=['Git internals and worktrees including their isolated virtual environments','all weights/checkpoints','raw and saved geometry/distance evidence, including files named cache','reports, ZIP deliverables and original datasets'],
 size_note='Logical file length sum; hard links may be counted twice and symlink/junction targets not followed. Deletion bytes are cache bytes, not net disk free change after audit files.')
write('cleanup_summary.json',summary)
report=f'''# 临时缓存清理与数据盘点

已删除主工作区13个可重建Python字节码文件，共145,763 bytes（约146 KB）。逐文件先验证绝对路径位于E:\\MeshFlow、普通文件类型、SHA256与盘点记录一致，再按LiteralPath删除；没有递归删除。空的__pycache__目录随后移除。

当前Chair源数据：{len(fs):,}个UID/NPZ，网格文件47.19 MB，加train/test划分文件47.45 MB；原始官方划分为训练2,822、测试313。2026-10-02既有几何去重盘点为3,125组，此轮没有重算几何去重。

已准备实验父对象48个：训练32、验证8、测试8，是上述Chair源数据的子集。此前的B宽度面板仅用了其中8父，不等于项目只有8个对象。

其他历史测试数据包括ss_overfit的128个文件（同一个cow几何的复制），以及历史worktree中的closed16及其512面派生版本；不能把这些文件数相加当作新增独立对象。shapenet-cls.tar.gz原归档25.48 MB继续保留，不从归档大小推断全部网格已下载。

项目盘点合计{s['bytes']/1e9:.2f} GB（逻辑文件长度，含Git/worktree，链接不展开）。其中.pt权重/检查点约{s['extensions']['.pt']['bytes']/1e9:.2f} GB，Git元数据约{summary['git_metadata_bytes']/1e9:.2f} GB。大体积源于这些保留资产。未把raw、coverage_vectors、pullback_cache、检查点或报告当临时文件删除。

保留文件逐项核对大小/修改时间：{retained:,}项一致，意外改动或缺失0项。刚交付的CHAIR_CONTROL_ATTRIBUTION_REVIEW.zip SHA256仍为`{expected}`。原始数据、权重、raw和报告删除数0。

清理清单见cache_deletion_candidates.json；实际删除回执见deletion_receipts.jsonl；统计见cleanup_summary.json与directory_sizes_depth*.csv。完整文件盘点CSV仅作为本轮核验临时输入，核验完成后移除；保留小型摘要与删除证据。没有训练、模型调用、Git reset/clean/commit/push。
'''
(OUT/'CLEANUP_REPORT.md').write_text(report,encoding='utf8')
# Remove only this audit's own bulky temporary inventory after successful verification.
tmp=(OUT/'file_inventory_before.csv').resolve()
assert tmp.is_relative_to(OUT.resolve()) and tmp.name=='file_inventory_before.csv'
tmp.unlink()
print(json.dumps(summary,ensure_ascii=False))
