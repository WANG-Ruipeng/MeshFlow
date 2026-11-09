"""Create this incident's fixed recovery preparation from the audited predecessor."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=HERE.parent/'metadata_read_shutdown_20261019/prepare_recovery.py'
text=source.read_text(encoding='utf-8')
replacements={
    'second recovery':'third engineering recovery',
    'BOTH --apply and a matching explicit user approval':'BOTH --apply and a matching recorded user-authorized scope interpretation',
    "INCIDENT='metadata_read_shutdown_20261019'":"INCIDENT='worker_write_race_20261019'",
    "CHECKPOINT_SHA='c322eea82d0691a15a04c55e126f3e5820af7d0ad34893a62194cd69fedae377'":"CHECKPOINT_SHA='2c37e0cd7c0c7054f1bea215e9e6634e96be1f52838f8cf9ab11cad1f33a936c'",
    "STATE_SHA='3503cbc9207c77b5dce5379a33bb4cab8b4c1e286d6c2fcb55d4deb507eec660'":"STATE_SHA='87530c7cb64f257d52a300ed7dfbb09699e268a1995e57bdd41f449195fda5d7'",
    "PAYLOAD_AUDIT=ROOT/'recovery2_cpu_audit_20261019_1792345993997925922/receipt.json'":"PAYLOAD_AUDIT=AUDIT/'checkpoint_audit.json'",
    "('run_all','train','resume_train','sample','postprocess')":"('run_all','train','resume_train','worker_entry','sample','postprocess')",
    "(336,3398)":"(464,5003)",
    "for key in ('authorization_text','authorization_source'):":"for key in ('authorization_text','authorization_source','scope_interpretation'):",
    "'Missing explicit user evidence: '":"'Missing user instruction/scope evidence: '",
    "and old_status['status']=='TRAINING'":"and old_status['status']=='BLOCKED'",
    "changed={CODE/'run_all.py',CODE/'controller.py'}":"changed={CODE/'run_all.py'}",
    "wrapper=CODE/'resume_train.py'":"wrapper=CODE/'worker_entry.py'",
    "'Off-grid wrapper missing'":"'Resilient worker entry missing'",
    "('train','resume_train','sample','postprocess','evaluation','curves','convergence')":"('train','resume_train','worker_entry','sample','postprocess','evaluation','curves','convergence')",
    "attempt['start_step']==4000":"attempt['start_step']==8000",
    "SECOND_EXPLICIT_USER_AUTHORIZED_RECOVERY_FROM_COMMITTED5250":"THIRD_BOUNDED_ENGINEERING_RECOVERY_FROM_COMMITTED8750",
    "PREPARED_AFTER_EXPLICIT_SECOND_USER_AUTHORIZATION":"PREPARED_UNDER_USER_PRIORITY_TRAINING_COMPLETION_SCOPE",
    "'Explicit second user authorization record'":"'Recorded user instruction and bounded engineering interpretation'",
    "READY_FOR_SECOND_AUTHORIZED_RECOVERY":"READY_FOR_THIRD_BOUNDED_ENGINEERING_RECOVERY",
    "r2.pilots.bench_parts_convergence_v1.resume_train":"r2.pilots.bench_parts_convergence_v1.worker_entry",
    '.recovery2_partial':'.recovery3_partial',
}
for old,new in replacements.items():
    assert old in text,old
    text=text.replace(old,new)
# Exact numerical identities; no scientific update/end-point budget changes.
for old,new in [('5363','8887'),('5362','8886'),('5251','8751'),('5250','8750'),('896','1088'),('112','136'),('6000','10000')]:
    text=text.replace(old,new)
text=text.replace("'Missing CPU payload PASS'","'Missing fixed8750 CPU payload PASS'")
target=HERE/'prepare_recovery.py'
assert not target.exists()
target.write_text(text,encoding='utf-8')
print(target)
