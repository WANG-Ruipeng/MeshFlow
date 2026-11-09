"""Record and fix WSL-vs-Windows EOL audit mismatch; no scientific reruns."""
import difflib
import shutil
import subprocess
from .common import *


def run():
    folder=OUT/'engineering/delivery_revision005'
    folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'revision_receipt.json'
    if receipt.exists():raise RuntimeError('Revision already applied')
    for p in (CODE/'delivery_audit.py',CODE/'final_review.py',OUT/'protocol/completion_entry_sources.json',
              OUT/'controller_status.json',OUT/'final_completion.json',OUT/'CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_REPORT.md'):
        shutil.copyfile(p,folder/(p.name+'.before'))
    git={}
    for name,args in [('WSL_config',['config','--get','core.autocrlf']),('WSL_default_diff',['diff','--stat']),
                      ('Windows_compatible_diff',['-c','core.autocrlf=true','diff','--stat']),
                      ('Windows_compatible_status',['-c','core.autocrlf=true','status','--short'])]:
        r=subprocess.run(['git',*args],cwd=ROOT,text=True,capture_output=True,timeout=30)
        git[name]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    atomic_json(folder/'git_comparison.json',git)
    assert git['Windows_compatible_diff']['returncode']==0 and not git['Windows_compatible_diff']['stdout'].strip()
    target=CODE/'delivery_audit.py';old=target.read_text(encoding='utf-8')
    needle="subprocess.run(['git',*args],cwd=ROOT,text=True,capture_output=True,timeout=20)"
    assert old.count(needle)==1
    new=old.replace(needle,"subprocess.run(['git','-c','core.autocrlf=true',*args],cwd=ROOT,text=True,capture_output=True,timeout=20)")
    new=new.replace("    assert git['tracked_diff']['returncode']==0", "    atomic_json(out/'delivery_git_audit.json',dict(policy='Match existing Windows core.autocrlf=true using command-only override; protected file bytes separately SHA256 verified',results=git))\n    assert git['tracked_diff']['returncode']==0")
    compile(new,str(target),'exec');target.write_text(new,encoding='utf-8')
    (folder/'code.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='delivery_audit.py.before',tofile='delivery_audit.py')),encoding='utf-8')
    frozen=OUT/'protocol/completion_entry_sources.json';values=read_json(frozen)
    key=str(target.relative_to(ROOT));assert key in values
    values[key]=digest(target);atomic_json(frozen,values)
    # Final editorial correction only: the table reads training device peaks,
    # and there are zero unfinished requests after the completed 400-row audit.
    final=CODE/'final_review.py';s=final.read_text(encoding='utf-8')
    needle="    marker='\\n## 完整图库与训练诊断复核\\n'"
    assert s.count(needle)==1
    s=s.replace(needle,"    text=text.replace('其余尚未完成。','其余未完成0。').replace('设备采样峰值MiB','训练设备占用峰值MiB')\n"+needle)
    compile(s,str(final),'exec');final.write_text(s,encoding='utf-8')
    atomic_json(receipt,dict(status='APPLIED',reason='WSL git default EOL conversion differs from the existing Windows checkout; no tracked bytes changed.',
        delivery_before=digest(folder/'delivery_audit.py.before'),delivery_after=digest(target),
        final_review_before=digest(folder/'final_review.py.before'),final_review_after=digest(final),
        protocol_and_scoring_definitions_unchanged=True,persistent_git_config_changed=False,
        model_calls=0,optimizer_updates=0,new_generation=0,new_geometry_scores=0))
    print('Delivery Git compatibility repair recorded; scientific stages unchanged',flush=True)


if __name__=='__main__':run()
