import pathlib,difflib,json,hashlib
P=pathlib.Path('/mnt/e/MeshFlow/experiments/r2_condition_width_response_v1')
dest=P/'engineering/report_clarification';dest.mkdir(parents=True,exist_ok=True)
p=P/'REPORT.md';before=p.read_text(encoding='utf-8')
old='I 沿用 ConflictCov_C 的内部采样覆盖定义；不是座面采用率。'
new='I 沿用 ConflictCov_C 定义：满足内部资格的冲突查询权重占完整 C 查询权重之比；不是座面采用率。\n\n每条原 C 的 2,048 个查询样本身份固定，变体只作相同 X 仿射，不重抽点；法线及边界距离按新 C 更新。原面积采样权重乘对应面的新/旧面积比后重权归一，补偿非均匀面积变化。I 的分母仍是完整注册 C 查询的权重总和，内部资格只决定哪些项进入分子，不对内部子集重新归一。原宽度查询与旧缓存逐位一致，变体查询具有独立 hash。'
assert before.count(old)==1
after=before.replace(old,new)
old2='required-interface 评分 S/E 为 NA；没有使用完整来源 Y 或唯一 GT 款式排名。'
new2='S 因没有指定 required-interface 而为 NA；E 因没有指定曝光要求而为 NA。没有使用完整来源 Y 或唯一 GT 款式排名。'
assert after.count(old2)==1
after=after.replace(old2,new2)
(dest/'REPORT.before.md').write_text(before,encoding='utf-8')
(dest/'code.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='REPORT.before.md',tofile='REPORT.md')),encoding='utf-8')
p.write_text(after,encoding='utf-8')
h=lambda x:hashlib.sha256(x).hexdigest()
audit=P/'audit/final_artifact_check.json';oldcheck=audit.read_bytes();(dest/'final_artifact_check.before.json').write_bytes(oldcheck)
check=json.loads(oldcheck);check['report_sha256']=h(p.read_bytes());check['report_clarifications']='S and E NA reasons separated; exact inherited I denominator and transformed query measure documented';audit.write_text(json.dumps(check,indent=2)+'\n')
receipt=dict(status='CORRECTED',before_sha256=h(before.encode()),after_sha256=h(p.read_bytes()),scientific_results_changed=False,model_calls=0,scoring_calls=0,issues=['Separate S required-interface from E exposure contract','Document affine fixed queries, area importance weights and full-query I denominator'],first_attempt='PowerShell pipeline encoded Chinese literals incorrectly; search assertion stopped before report writes; rerun from UTF8 file')
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
