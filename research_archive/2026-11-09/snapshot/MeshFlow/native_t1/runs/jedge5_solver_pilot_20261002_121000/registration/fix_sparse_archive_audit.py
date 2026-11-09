from pathlib import Path
p=Path('experiments/mf_h1_local0/jedge5_solver_pilot/final_audit.py')
s=p.read_text(encoding='utf-8-sig')
old="z=arrays(pathof(row['path'],out));states=z['states'].reshape(nfe+1,112,9);check(bits(states[0],references[cid]['states'][0].reshape(112,9)) and bits(states[-1],z['raw'].reshape(112,9)) and np.isfinite(states).all(),'Reused baseline initial/final state')"
new="z=arrays(pathof(row['path'],out));states=z['states'].reshape(-1,112,9);indices=z['state_indices'];check(len(indices)==len(states) and int(indices[0])==0 and int(indices[-1])==nfe and np.all(np.diff(indices)>0),'Archived baseline saved-state indices; sparse archive allowed');check(bits(states[0],references[cid]['states'][0].reshape(112,9)) and bits(states[-1],z['raw'].reshape(112,9)) and np.isfinite(states).all(),'Reused baseline initial/final state')"
assert old in s;s=s.replace(old,new);p.write_text(s,encoding='utf-8')
