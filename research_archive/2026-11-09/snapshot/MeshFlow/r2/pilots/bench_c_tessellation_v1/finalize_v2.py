"""Versioned documentation correction; no scientific recomputation."""
import json
import zipfile
from pathlib import Path
from . import common as C
from .closeout import put


def main():
    out=C.OUT;old=C.read(out/'COMPLETE.json');reg=C.read(out/'registration.json')
    C.verify(reg['protected']);C.verify(reg['inputs'])
    changed=[rel for rel,h in old['artifact_hashes'].items() if C.sha(out/rel)!=h]
    assert changed==['RUNBOOK.md'],changed
    note='''# 完成态恢复说明更正（v2）

独立只读复核通过：16比较格、6缓存＋10新RAW＋1复现、550前向；12曲线/612逐步行；10完整评分及8额外COMMON_C CF扫描；每父两噪声均值、RAW阶段、资源并行记录和科学结论均一致。

唯一工程更正：RUNBOOK不再建议完成状态运行analyze curves。曲线计算完成后，metrics_v2元数据修正改变了analyze.py hash；旧curves函数没有完成态早退，重跑会先重写派生表/图再因旧不可变收据不一致而失败。此次只修改使用说明，没有实际触发该重跑，没有重写科学表/图/收据。

已完成状态仅用run status/prepare检查；不要将closeout当任意代码版本下的通用重建命令。实际看图记录均为AI_PROXY，非人类评阅。

原COMPLETE.json和首版ZIP原样保留，包含更正前RUNBOOK的历史hash。当前权威完成核验为COMPLETE_v2.json，当前交付包为R2_BENCH_C_TESSELLATION_REVIEW_V2.zip。所有科学结果完全相同。
'''
    put(out/'CLOSEOUT_CORRECTION.md',note.encode('utf-8'))
    src=Path(__file__);dst=out/'source_archive_closeout_v2/finalize_v2.py';put(dst,src.read_bytes())
    v2=dict(old)
    v2.update(version='closeout_v2_documentation_only',prior_complete_sha256=C.sha(out/'COMPLETE.json'),
        changed_scientific_artifacts=[],changed_documents=['RUNBOOK.md'],
        independent_review='retessellation_design_audit: scientific claims/counts/metrics/resources PASS; completed-state documentation correction only')
    v2['artifact_hashes']=dict(old['artifact_hashes'])
    for p in (out/'RUNBOOK.md',out/'CLOSEOUT_CORRECTION.md',dst):v2['artifact_hashes'][str(p.relative_to(out))]=C.sha(p)
    C.write(out/'COMPLETE_v2.json',v2)
    C.verify({str(out/p):h for p,h in v2['artifact_hashes'].items()})
    original=out/'R2_BENCH_C_TESSELLATION_REVIEW.zip'
    assert C.sha(original)==C.read(out/'archive_receipt.json')['sha256']
    target=out/'R2_BENCH_C_TESSELLATION_REVIEW_V2.zip'
    replacements={p.name:p for p in (out/'RUNBOOK.md',out/'CLOSEOUT_CORRECTION.md',out/'COMPLETE_v2.json')}
    replacements['source_archive_closeout_v2/finalize_v2.py']=dst
    with zipfile.ZipFile(original) as z:
        entries={n:z.read(n) for n in z.namelist()}
    entries.update({n:p.read_bytes() for n,p in replacements.items()})
    if target.exists():
        with zipfile.ZipFile(target) as z:
            assert set(z.namelist())==set(entries)
            assert all(z.read(n)==b for n,b in entries.items())
    else:
        with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for n,b in sorted(entries.items()):z.writestr(n,b)
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    receipt=dict(status='COMPLETE',path=str(target),sha256=C.sha(target),bytes=target.stat().st_size,
        entries=len(entries),scientific_recomputation=0,model_calls=0,weights_included=False,
        original_archive_retained=True,authoritative_complete='COMPLETE_v2.json')
    C.write(out/'archive_receipt_v2.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
