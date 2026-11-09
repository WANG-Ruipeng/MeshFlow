"""Make a <10 MB review subset of the already-sealed full review ZIP.

Only duplicated reductions, detailed geometry ledgers and four zoom pages are
omitted; never select figures by outcomes. Full original package is untouched.
"""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
source=ROOT/'R2_BENCH_HIGH_JOINT_COVERAGE_REVIEW.zip'
target=ROOT/'R2_BENCH_HIGH_JOINT_COVERAGE_REVIEW_SMALL.zip'
excluded={
 'results/analysis_v1/deletion_fate_per_slot.csv',
 'backend_geo_only/analysis/corner_displacement_ledger.csv',
 'backend_geo_only/analysis/slot_stage_ledger.csv',
 'data/schedule_WIDE_HIGH.json',
 'training_closeout/exposure_identity_checks.csv',
 'results/analysis_v1/full177_parent_macro.csv',
 'results/analysis_v1/paired177_parent_macro.csv',
}


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    assert not target.exists(), 'Refuse overwrite sealed package'
    manifest=[]; omitted=[]
    with zipfile.ZipFile(source) as zin:
        assert zin.testzip() is None
        names=zin.namelist()
        for name in names:
            if name.startswith('backend_geo_only/gallery/') and name.endswith('__detail.png'):
                excluded.add(name)
        assert excluded<=set(names)
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zout:
            for info in zin.infolist():
                data=zin.read(info.filename)
                rec=dict(path=info.filename,bytes=len(data),sha256=sha(data))
                if info.filename in excluded: omitted.append(rec); continue
                zout.writestr(info.filename,data);manifest.append(rec)
            code=Path(__file__).read_bytes()
            zout.writestr('make_small_review.py',code)
            note='''# 网页审阅小包（小于10 MB）

本包是已封存完整审阅ZIP的确定性子集，所有保留成员逐字节一致。
保留完整逐输出177字段CSV、主要逐父/配对表、逐输出面积与删除原因、真实五轴图审、
曝光/权重/学习曲线、数据身份、178个归档源码、四RAW后端对照、全部新增8请求三阶段24页图，
以及四份GEO_ONLY全流程原图。没有按好坏选图或改小图片。

仅省略训练流逐实例副本、详细逐角/逐槽几何账本、可由完整逐输出复算的两张大型宏表、
以及四张额外放大图。完整文件在同目录完整ZIP及本地原件中；报告中指向这些项目的链接
在小包内可能不可用，参见MINI_MEMBERS.json的omitted字段。
大权重、RAW/POST数组仍留本地，两个ZIP均不是独立训练发行包。
'''
            zout.writestr('MINI_README.md',note.encode('utf-8'))
            receipt=dict(full_package_sha256=sha(source.read_bytes()),included=manifest,omitted=omitted,
                         package_script_sha256=sha(code),selection='Fixed artifact-type omissions; no outcome filtering')
            zout.writestr('MINI_MEMBERS.json',json.dumps(receipt,ensure_ascii=False,indent=2).encode('utf-8'))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for item in manifest: assert sha(z.read(item['path']))==item['sha256']
    assert target.stat().st_size<10_000_000, target.stat().st_size
    result=dict(status='COMPLETE',path=str(target),bytes=target.stat().st_size,sha256=sha(target.read_bytes()),
        full_package=str(source),full_package_sha256=sha(source.read_bytes()),
        original_members_retained=len(manifest),original_members_omitted=len(omitted),
        model_calls=0,scoring_calls=0,image_edits=0,originals_modified=False,
        package_script_sha256=sha(Path(__file__).read_bytes()))
    dest=ROOT/'small_package_receipt.json';assert not dest.exists()
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
