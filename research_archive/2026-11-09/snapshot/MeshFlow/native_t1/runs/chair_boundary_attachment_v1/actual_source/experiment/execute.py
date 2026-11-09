"""Complete resumable entry, including independent and protected-source audits."""
import argparse
import re
import traceback
from .common import *


def finish(out=OUT):
    out=Path(out)
    from . import independent_audit,delivery_audit
    audit=independent_audit.run(out)
    if audit['status']!='PASS':raise RuntimeError('Independent final audit not PASS: '+str(out/'independent_audit.json'))
    protection=delivery_audit.run(out)
    if protection['status']!='PASS':raise RuntimeError('Final protected-source audit did not pass')
    report=out/'CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_REPORT.md'
    text=report.read_text(encoding='utf-8')
    text=re.sub(r'最终公共源码保护hash[^\n]*',
        '最终公共源码与旧资产保护核验已通过，实际源码、source_hashes.json 与 code.diff 均已交付；详见文末独立复核收据。',text)
    marker='\n## 独立复核与交付完整性\n'
    if marker in text:text=text.split(marker)[0]
    text+=marker+'\n独立核算 **PASS**，共 '+str(audit['total_checks'])+' 项检查；直接读取逐例表、训练日志与原调用账本，未调用主汇总函数。见 [独立核算收据](independent_audit.json)、[逐噪声配对差](paired_noise_differences.csv) 和 [独立父均](independent_parent_means.csv)。\n\n'
    text+='受保护的实际依赖、原输入和缓存共 '+str(protection['protected_files'])+' 个文件，前后哈希全部一致；已跟踪公共代码无差异，无自动提交、push或默认模型变更。见 [保护核验](protected_after_audit.json)。\n\n'
    text+='[实际实验与原评分源码](actual_source/)、[源文件哈希](source_hashes.json)、[新增代码完整diff](code.diff)、[400请求的raw完整性索引](raw_integrity_index.csv) 已保存。原评分修订、失败原件、diff和重复CPU评分记录保留于 engineering/，未重生成已有网格。\n\n'
    text+='完整恢复入口：python -B -m experiments.mf_h1_local0.chair_boundary_attachment_v1.execute。固定任务至此结束；未执行报告建议的下一步。\n'
    report.write_text(text,encoding='utf-8')
    for filename in ('RUN_AND_RESUME.md','DELIVERY_INDEX.md'):
        p=out/filename
        doc=p.read_text(encoding='utf-8').replace('chair_boundary_attachment_v1.run_all','chair_boundary_attachment_v1.execute')
        if filename=='DELIVERY_INDEX.md':
            section='\n最终独立交付：\n\n- [独立核算收据](independent_audit.json)\n- [逐噪声配对差](paired_noise_differences.csv)\n- [受保护源文件核验](protected_after_audit.json)\n- [实际源码](actual_source/)\n- [源文件hash](source_hashes.json)\n- [code.diff](code.diff)\n- [完整raw身份索引](raw_integrity_index.csv)\n'
            if '\n最终独立交付：' in doc:doc=doc.split('\n最终独立交付：')[0]
            doc+=section
        p.write_text(doc,encoding='utf-8')
    from .ledger import Budget
    account=Budget(out,'final_completion').snapshot()
    result=dict(status='DONE',classification=read_json(out/'aggregation/summary.json')['classification'],
        per_arm_optimizer_updates=account['per_arm_optimizer_attempts'],total_forward_calls=account['total_model_forward_attempts'],
        independent_checks=audit['total_checks'],protected_files=protection['protected_files'],
        report_sha256=digest(report),source_hashes_sha256=digest(out/'source_hashes.json'),
        stopped_after_fixed_task=True,additional_training_or_generation=0)
    atomic_json(out/'final_completion.json',result)
    atomic_json(out/'controller_status.json',dict(status='DONE',entrypoint=__package__+'.execute',result=result))
    return result


def run(out=OUT,through='all'):
    out=Path(out)
    sources={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),CODE/'delivery_audit.py')}
    freeze_json(out/'protocol/completion_entry_sources.json',sources)
    from . import run_all
    try:
        value=run_all.run(out,through=through)
        if through!='all':return value
        atomic_json(out/'controller_status.json',dict(status='FINAL_AUDIT',entrypoint=__package__+'.execute'))
        return finish(out)
    except BaseException as exc:
        detail=dict(status='BLOCKED',entrypoint=__package__+'.execute',through=through,
            error=repr(exc),traceback=traceback.format_exc(),final_audits_not_complete=True)
        atomic_json(out/'controller_status.json',detail)
        if through=='all':atomic_json(out/'final_completion.json',detail)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT))
    p.add_argument('--through',choices=['prepare','precheck','train','sample','score','all'],default='all')
    a=p.parse_args();print(run(a.out,a.through),flush=True)
