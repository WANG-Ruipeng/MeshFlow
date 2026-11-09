"""Pre-dispatch bookkeeping correction; no model calls."""
from pathlib import Path
p=Path(__file__).with_name('training.py')
s=p.read_text()
s=s.replace('total=fm+auxiliary','total=fm+auxiliary if arm in SEAM_ARMS else fm')
s=s.replace("getattr(model,'last_boundary_diagnostics',None)","getattr(model,'boundary_observations',None)")
s=s.replace("gate=float(float(s['t'])>=.5)\n        result", "gate=float(float(s['t'])>=.5)\n        result['valid_seam_samples']+=int(diag['counts']['eligible_interface_edges']>0)\n        result")
s=s.replace("'arm','attachment_step','sample_index','task_id','alpha','K','t'", "'arm','attachment_step','sample_index','uid','task_id','alpha','N','K','t'")
p.write_text(s)
print('Initial bookkeeping repairs complete; zero model calls')
