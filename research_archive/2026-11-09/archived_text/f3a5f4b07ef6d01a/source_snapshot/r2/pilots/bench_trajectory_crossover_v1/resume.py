"""Completed task recovery entry: never dispatch GPU or silently repeat work."""
from .common import *

def main():
    if (OUT/'COMPLETE.json').exists():
        from .closeout import verify_complete
        verify_complete();return
    pending=[]
    for combo in ('CU','UC'):
        if not (OUT/'execution'/combo/'complete.json').exists():pending.append(combo+' suffix: inspect calls before any replay')
    for name in ('scores_complete.json','gallery/complete.json','analysis/complete.json'):
        if not (OUT/name).exists():pending.append(name)
    print(json.dumps(dict(status='PENDING_REVIEW' if pending else 'READY_FOR_CLOSEOUT',pending=pending,
        model_calls_started=0,next='Do not replay unfinished GPU. Completed outputs can be scored/rendered using their immutable receipts.'),ensure_ascii=False))

if __name__=='__main__':main()
