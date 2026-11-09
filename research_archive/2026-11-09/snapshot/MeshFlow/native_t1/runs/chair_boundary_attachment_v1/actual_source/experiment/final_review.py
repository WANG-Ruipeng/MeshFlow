"""CPU-only final gallery coverage and training-diagnostic report addendum."""
import csv
import shutil
from .common import *


def run(out=OUT):
    out=Path(out)
    final=read_json(out/'final_completion.json')
    assert final['status']=='DONE'
    assert read_json(out/'independent_audit.json')['status']=='PASS'
    assert read_json(out/'protected_after_audit.json')['status']=='PASS'
    diagnostic=read_json(out/'review_training_diagnostics.json')
    assert diagnostic['status']=='PASS'
    gallery=read_json(out/'gallery/summary.json')
    assert gallery['status']=='COMPLETE'
    assert gallery['rendered_P1']==160 and gallery['rendered_P5']==40
    assert gallery['generated_images']==412
    expected={r['request_id'] for r in read_json(out/'requests.json') if r['panel'] in ('P1','P5')}
    records=gallery['records']
    assert len(records)==200 and {r['request_id'] for r in records}==expected
    for row in records:
        for field in ('geometry','overlay'):
            assert digest(resolve(row[field+'_image']))==row[field+'_sha256']
        assert row['fixed_views']==4 and row['wireframe'] and row['geometry_repairs']==0 and row['C_bitwise']
    assert len(list((out/'gallery/assets').glob('*.png')))==412
    archive=out/'engineering/final_review'
    archive.mkdir(parents=True,exist_ok=True)
    for name in ('CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_REPORT.md','final_completion.json','controller_status.json','DELIVERY_INDEX.md'):
        dst=archive/(name+'.before')
        if not dst.exists():shutil.copyfile(out/name,dst)
    report=out/'CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_REPORT.md'
    text=report.read_text(encoding='utf-8')
    text=text.replace('其余尚未完成。','其余未完成0。').replace('设备采样峰值MiB','训练设备占用峰值MiB')
    marker='\n## 完整图库与训练诊断复核\n'
    if marker in text:text=text.split(marker)[0]
    table=(out/'review_training_diagnostics.md').read_text(encoding='utf-8').split('\n\n')[1]
    table=table.replace('Arm','训练臂').replace('FM last100 mean','末100步FM均值').replace('weighted seam/FM median','加权接缝/FM比中位数').replace('final boundary norm','边界向量末步范数').replace('boundary grad median','向量梯度中位数').replace('boundary update median','向量更新中位数').replace('marked C-corner mean','标记known角比例均值').replace('None','NA')
    text+=marker+'\n逐请求与图片hash复核通过：全部 **160个P1输出 + 40个P5输出** 均有四视角/线框和三指标叠图，共 **412张PNG**（含12张GT参考）。同父统一画框，未裁掉异常面；这属于技术核验，不是真实用户研究。\n\n'
    text+=table+'\n\n四组各1000个更新与8000个样本的输入、OT结果和监督身份全部配对一致。B/D每步边界向量梯度与更新均非零，8000例均未出现近常数标记。C/D加权辅助loss与FM之比约1%，这是训练loss数值比，不能与校准时输出梯度比例目标25%混同；这些训练量本身不证明生成收益。\n\n'
    text+='详见[独立训练诊断](review_training_diagnostics.json)、[P1完整图库](gallery/index.html)、[P5多片图库](gallery/nested.html)。最终复核没有模型前向、训练更新或新生成。\n'
    sensitivity=list(csv.DictReader((out/'aggregation/sensitivity.csv').open(encoding='utf-8',newline='')))
    assert len(sensitivity)==12
    text+='\n密度敏感性的全部4个固定单元（首两父候选0，START/D）如下。差值只反映离散评分变化，未替换主结果；4个单元不能代表全部8父的数值误差。\n\n'
    text+='| 请求 | 指标 | 主密度 | 加密 | 加密减主密度 |\n|---|---|---:|---:|---:|\n'
    for row in sensitivity:
        text+='| '+row['request_id']+' | '+row['metric']+' | '+row['main']+' | '+row['dense']+' | '+row['dense_minus_main']+' |\n'
    text+="\n\u56db\u5355\u5143\u5355\u8f93\u51fa\u5bc6\u5ea6\u53d8\u5316\u6700\u5927\u503c\u5206\u522b\u4e3a\uff1aSeamCov 0.16343\u3001ConflictCov_C 0.17090\u3001ExposedRetain 1.03177 \u4e2a\u767e\u5206\u70b9\u300233ec\u2026\u5019\u90090\u7684D\u2212START\u5916\u9732\u53d8\u5316\u4ece\u4e3b\u5bc6\u5ea6 \u22120.20141 \u4e2a\u767e\u5206\u70b9\u7ffb\u4e3a\u52a0\u5bc6\u540e\u7684 [+0.84457,+0.86978] \u4e2a\u767e\u5206\u70b9\uff1b**\u8be5\u5c40\u90e8\u5916\u9732\u589e\u51cf\u65b9\u5411\u6807\u4e3a LIMITED_NUMERICAL_SUPPORT**\u3002\u4e24\u5bc6\u5ea6\u5747\u672a\u8d8a\u8fc7 \u22122 \u4e2a\u767e\u5206\u70b9\u5bb9\u8bb8\u7ebf\uff0c\u63a5\u7f1d\u5bf9\u6bd4\u4e5f\u672a\u7ffb\u8f6c\uff0c\u56e0\u6b64\u8fd9\u4e9b\u68c0\u67e5\u6ca1\u6709\u63a8\u7ffb\u672c\u8f6e NO_ATTACHMENT_SIGNAL\uff1b\u4e0d\u80fd\u4ee54\u4e2a\u5355\u5143\u58f0\u79f0\u5168\u90e88\u7236\u7684\u8bc4\u5206\u5bc6\u5ea6\u5747\u5df2\u9a8c\u8bc1\u3002Gamma\u70b9\u6570\u3001C\u6837\u672c\u91cf\u4e0e\u65b9\u5411\u540c\u65f6\u6539\u53d8\uff0c\u672a\u5355\u72ec\u5f52\u56e0\u5230\u67d0\u4e2a\u56e0\u7d20\u3002\u89c1[\u5bc6\u5ea6\u72ec\u7acb\u590d\u6838](review_density_interpretation.json)\u3002\n\nP5\u63a5\u7f1d\u6c47\u603b\u53ea\u8986\u76d6\u80fd\u591f\u5b8c\u6574\u8ba1\u7b97\u76841/2\u7236\uff1a\u53e6\u4e00\u4e2a\u7236\u7684C3\u771f\u5b9eGamma\u539f\u672c\u4e0d\u9002\u7528\uff0c\u4e0d\u80fd\u7f29\u5c0f\u5206\u6bcd\u540e\u5ba3\u79f0\u4e24\u7236\u5747\u6539\u5584\u3002\u53ef\u5b8c\u6574\u8ba1\u7b97\u7236\u7684C2/C3\u5e73\u5747SeamCov\u4e3aSTART 14.54%\u3001A 10.22%\u3001B 10.21%\u3001C 8.99%\u3001D 9.11%\uff1bConflict\u4e0eExposure\u4ecd\u5bf9\u4e24\u7236\u5168\u90e8\u8ba1\u7b97\u3002P4\u7559\u51fa\u4e0aC\u76f8\u5bf9A\u6709\u7ea60.40\u4e2a\u767e\u5206\u70b9\u7684\u5c40\u90e8\u63a5\u7f1d\u589e\u76ca\uff0cD\u5374\u4e0b\u964d\u7ea60.71\u4e2a\u767e\u5206\u70b9\uff1bP2\u8bad\u7ec3\u7236\u56db\u7ec4\u7ee7\u7eed\u8bad\u7ec3\u540e\u7684\u63a5\u7f1d\u5747\u503c\u5747\u4f4e\u4e8eSTART\u3002\u73b0\u6709\u7ed3\u679c\u4e0d\u652f\u6301\u8de8\u9762\u677f\u4e00\u81f4\u7684\u7ec4\u5408\u6536\u76ca\u3002\n"
    text+='\n'+(out/'review_conclusions.md').read_text(encoding='utf-8')+'\n'
    report.write_text(text,encoding='utf-8')
    receipt=dict(status='PASS',main_outputs=160,nested_outputs=40,png_files=412,
        all_200_request_ids_unique_and_present=True,all_400_output_image_hashes_match=True,
        training_diagnostic_sha256=digest(out/'review_training_diagnostics.json'),
        report_before_sha256=digest(archive/(report.name+'.before')),report_after_sha256=digest(report),
        source_sha256=digest(__file__),model_forward_calls=0,backwards=0,optimizer_updates=0,new_generation=0)
    atomic_json(out/'final_review.json',receipt)
    final.update(report_sha256=digest(report),final_review_sha256=digest(out/'final_review.json'),
        gallery_P1=160,gallery_P5=40,gallery_PNG=412)
    atomic_json(out/'final_completion.json',final)
    state=read_json(out/'controller_status.json');state['result']=final
    atomic_json(out/'controller_status.json',state)
    index=out/'DELIVERY_INDEX.md';doc=index.read_text(encoding='utf-8')
    note='\n- [完整图库与训练诊断最终复核](final_review.json)\n- [训练诊断逐项收据](review_training_diagnostics.json)\n'
    if '训练诊断最终复核' not in doc:index.write_text(doc+note,encoding='utf-8')
    print(receipt,flush=True)
    return receipt


if __name__=='__main__':run()
