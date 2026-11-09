"""Recover only completed-run final CPU audits and review; never rerun models."""
import traceback
from .common import *


def run(out=OUT):
    out=Path(out)
    expected=read_json(out/'protocol/completion_entry_sources.json')
    assert all(digest(resolve(k))==v for k,v in expected.items())
    for arm in ARMS:
        for phase in ('train','sample'):
            assert read_json(out/'workers'/f'{phase}_{arm}.json')['status']=='DONE'
    assert read_json(out/'aggregation/summary.json')['status']=='COMPLETE'
    assert read_json(out/'gallery/summary.json')['rendered_P1']==160
    from .execute import finish
    from .final_review import run as review
    append_event(out/'ledger/controller.jsonl',dict(event='CPU_FINAL_RECOVERY_BEGIN',model_calls=0))
    atomic_json(out/'controller_status.json',dict(status='FINAL_AUDIT',entrypoint=__package__+'.resume_final_audits'))
    try:
        result=finish(out)
        review(out)
        append_event(out/'ledger/controller.jsonl',dict(event='CPU_FINAL_RECOVERY_RETURN',status='DONE',model_calls=0))
        return read_json(out/'final_completion.json')
    except BaseException as exc:
        detail=dict(status='BLOCKED',entrypoint=__package__+'.resume_final_audits',error=repr(exc),traceback=traceback.format_exc())
        atomic_json(out/'controller_status.json',detail);atomic_json(out/'final_completion.json',detail)
        raise


if __name__=='__main__':print(run(),flush=True)
