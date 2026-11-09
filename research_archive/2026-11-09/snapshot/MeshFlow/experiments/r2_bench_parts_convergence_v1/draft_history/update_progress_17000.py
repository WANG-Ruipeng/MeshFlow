"""Document an already completed stopping decision; no model/log access."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
decision_path=ROOT/'dispatch/decisions/step17000.json'
d=json.loads(decision_path.read_text(encoding='utf-8'))
assert d['current_global_step']==17000 and d['stage_updates']==5000
assert d['status']=='CONTINUE_BOTH' and d['next_common_target']==18000
draft=ROOT/'REPORT_DRAFT.md';receipt=ROOT/'REPORT_DRAFT_RECEIPT.json'
old=draft.read_text(encoding='utf-8')
history=ROOT/'draft_history/paired17000_completed_process_eval_pending'
history.mkdir(parents=True,exist_ok=False)
(history/'REPORT_DRAFT.md').write_text(old,encoding='utf-8')
(history/'REPORT_DRAFT_RECEIPT.json').write_text(receipt.read_text(encoding='utf-8'),encoding='utf-8')
start=old.index('本草稿更新到')
end=old.index('\n\n',start)
new=old[:start]+'''本草稿更新到两臂均已完成全局17000步（各分支新增5000步）、正在执行17000过程评测的时点。已完成的15000完整配对结果、全部实际图审及三次恢复开销保留。17000停止判定为CONTINUE_BOTH→18000：PLAIN达到有限平台，SEAM的ALL_free/bulk已满足平台规则，但seam_tip仍明显变化，因此两臂继续共同预算。此为监控快照，不是最终完成、数学收敛或模型晋级声明；没有读取活动日志推断实时步数。'''+old[end:]
new=new.replace('|两臂各自新增更新数与共同终点|已各完成3000步至全局15000，正在16000训练段；最终共同终点PENDING|',
    '|两臂各自新增更新数与共同终点|已各完成5000步至全局17000，17000过程评测中，判定继续18000；最终共同终点PENDING|')
new=new.replace('|分叉过程13000／17000／19000|同一步PLAIN与SEAM|每臂8，只有到达才执行|PENDING|',
    '|分叉过程13000／17000／19000|同一步PLAIN与SEAM|每臂8，只有到达才执行|13000已完成；17000评测中；19000未预填结果|')
anchor='## 5. 正式生成与124字段结果（PENDING）'
assert anchor in new
lines=['### 已有17000停止判定与最后区间（16000→17000）','',
 '仅摘录`dispatch/decisions/step17000.json`中的现有TRAIN固定探针数值。判定窗口为14000、15000、16000、17000。PLAIN为PLATEAU；SEAM唯一继续原因是`TRAIN_seam_tip_NOT_STABLE_OR_INCOMPLETE`。DEV没有用于停止或checkpoint选择。','',
 '|臂|层|16000 MSE|17000 MSE|相对变化|有效坐标分母|四点平台|','|---|---|---:|---:|---:|---:|---|']
for arm in ('PLAIN','SEAM'):
    for metric in ('ALL_free','bulk','seam_tip'):
        m=d['assessments'][arm]['metrics'][metric];last=m['intervals'][-1]
        b,a=last['before'],last['after']
        if a['count']==0:
            lines.append(f'|{arm}|{metric}|NA|NA|NA|0|无有效坐标，跳过|')
        else:
            lines.append(f"|{arm}|{metric}|{b['mse']:.6f}|{a['mse']:.6f}|{last['relative_change']*100:+.3f}%|{a['count']}|{'PASS' if m['plateau_pass'] else 'NOT_PASS'}|")
lines+=['','SEAM tip下降34.943%，说明该监督层仍在变化，因此不能因总体FM较平就宣布两臂共同达到平台；这也不是生成质量已经改善的结论。按冻结合同继续18000，不改数据、噪声、指标或预算上限。','']
new=new.replace(anchor,'\n'.join(lines)+'\n'+anchor)
draft.write_text(new,encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt.write_text(json.dumps(dict(schema='bench_parts_report_draft_update_v4',
    status='DRAFT_PAIRED17000_TRAIN_COMPLETE_PROCESS_EVAL_PENDING_CONTINUE18000',
    updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    prior_draft_sha256=sha(history/'REPORT_DRAFT.md'),current_draft_sha256=sha(draft),
    stopping_decision_sha256=sha(decision_path),final_paired_result='PENDING',model_promotion=False,
    model_calls=0,new_probes=0,rescoring=0,active_training_logs_read=0,training_code_changes=0),
    ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert '325个丢失已完成更新' in new and '### 5.1 固定15000步结果' in new
print('DRAFT_UPDATED_17000_OLD15000_RESULTS_AND_325_RECOVERY_UPDATES_PRESERVED')
