from pathlib import Path
import shutil,difflib,ast
root=Path('/mnt/e/MeshFlow');out=root/'native_t1/runs/chair_bounded_attachment_v1';p=root/'experiments/mf_h1_local0/chair_bounded_attachment_v1/reporting.py'
e=out/'engineering/report_wording';e.mkdir(parents=True,exist_ok=False)
old=p.read_text();(e/'reporting.py').write_text(old)
for name in ('BOUNDED_ATTACHMENT_CORRECTION_REPORT.md','report_summary.json','DELIVERY_INDEX.md'):
 shutil.copy2(out/name,e/name)
new=old.replace('该下一步未执行。本固定任务到此停止；','本固定任务到此停止；').replace('每个 outputs/{request_id}/ 保存实际 raw.npz、mesh.ply、result.json；修改输出另含 intervention.json 和 geometry_arrays.npz。','已执行的 outputs/{request_id}/ 保存实际 raw.npz、mesh.ply、result.json；修改输出另含 intervention.json 和 geometry_arrays.npz。未执行 TRAJECTORY 仅保存明确的 NOT_RUN 状态 result.json。')
assert new!=old and new.count('该下一步未执行')==0
ast.parse(new);p.write_text(new)
(e/'code.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='reporting.before_wording.py',tofile='reporting.py')))
print('WORDING_ONLY_AST_PASS')