"""Preserve report v1; clarify formula and the saved CPU timer scope only."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

out = Path(__file__).resolve().parents[2]
root = out.parents[2]
code = root/'experiments/mf_h1_local0/chair_control_usability_quick_v1/reporting.py'
folder = Path(__file__).resolve().parent
before = code.read_text(encoding='utf-8')
old1 = 'D 是 L0² 归一的新双向表面 MSE；接口是'
new1 = 'D=[mean d(Q_Y,F)²+mean d(Q_F,Y)²]/(2L0²)，两侧各2048个面积采样点到对方有限闭三角面的距离；Q_Y按条件固定，两个方向分别保存在逐输出表。接口是'
old2 = '逐输出保存的 80 份 CPU 主评分耗时合计'
new2 = '逐输出保存的 80 份 CPU 评分耗时合计'
old3 = '，独立敏感性/六对距离、读取缓存和重叠等待不应当加成单一端到端基准；图库阶段'
new3 = '（已含4项预登记加密复核，不含末尾向量存盘）；六对距离汇总、缓存读取和重叠等待另有开销，不能把这些计时相加当作端到端基准；图库阶段'
if old1 not in before:
    assert new1 in before and new3 in before
    print('Already applied; no mutation')
else:
    assert before.count(old1) == before.count(old2) == before.count(old3) == 1
    shutil.copyfile(code, folder/'reporting.before.py.txt')
    shutil.copyfile(out/'QUICK_CONTROL_USABILITY_REPORT.md',folder/'REPORT.before.md')
    after=before.replace(old1,new1).replace(old2,new2).replace(old3,new3)
    code.write_text(after,encoding='utf-8')
    patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
        fromfile='reporting.before.py',tofile='reporting.py'))
    (folder/'code.diff').write_text(patch,encoding='utf-8')
    (folder/'receipt.json').write_text(json.dumps(dict(status='APPLIED',model_calls=0,
        before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(after.encode()).hexdigest(),
        reason='Clarify explicit D formula and scoring_seconds including sensitivity computations; data unchanged'),indent=2)+'\n',encoding='utf-8')
    print('Applied report-only clarification; data unchanged')
