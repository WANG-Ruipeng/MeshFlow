"""One-time CPU-only completion-state correction before first outer dispatch."""
import difflib
import shutil
from .common import *


def run():
    target=CODE/'execute.py'
    folder=OUT/'engineering/completion_revision004'
    folder.mkdir(parents=True,exist_ok=True)
    original=folder/'execute.py.before'
    if original.exists():raise RuntimeError('Revision already applied; no replay')
    shutil.copyfile(target,original)
    old=target.read_text(encoding='utf-8')
    new=old.replace('import re\n','import re\nimport traceback\n')
    needle="    value=run_all.run(out,through=through)\n    return finish(out) if through=='all' else value\n"
    replacement="""    try:
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
"""
    if old.count(needle)!=1:raise RuntimeError('Unexpected source')
    new=new.replace(needle,replacement)
    compile(new,str(target),'exec')
    target.write_text(new,encoding='utf-8')
    (folder/'code.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='execute.py.before',tofile='execute.py')),encoding='utf-8')
    atomic_json(folder/'receipt.json',dict(status='APPLIED',before=digest(original),after=digest(target),
        reason='Final audit errors must persist BLOCKED and invalidate stale completion receipts.',
        model_forward=0,model_backward=0,optimizer_updates=0,new_generation=0,scientific_changes=False))
    print('Completion state revision applied before first outer dispatch',flush=True)


if __name__=='__main__':run()
