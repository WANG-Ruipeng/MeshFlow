"""Scientific line figure from saved triangles, highlighting identified source slots."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from case_trace import segments

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seam_s2000_section_audit_v1'

def main():
 protocol=json.loads((OUT/'protocol.json').read_text())
 req={q['request_id']:q for q in protocol['requests']}
 cases=[('3f741759_n2','side',{21:'#c62828'},(-1.4,1.4),(-.7,1.02)),
        ('608b46b0_n3','front',{77:'#c62828',113:'#e58b12'},(-1.35,-.6),(-.16,.075))]
 fig,axes=plt.subplots(2,4,figsize=(16,7),dpi=150)
 sources={}
 for row,(rid,view,highlight,xlim,ylim) in enumerate(cases):
  q=req[rid];axis=0 if view=='side' else 2;proj=[2,1] if view=='side' else [0,1]
  position=protocol['bounds'][q['condition_id']][view+'_positions'][1]
  for col,stage in enumerate(protocol['stages']):
   p=Path(q['paths'][stage]);digest=hashlib.sha256(p.read_bytes()).hexdigest()
   assert digest==protocol['source_hashes'][str(p)];sources[str(p)]=digest
   with np.load(p,allow_pickle=False) as z:
    tri=z['raw' if stage=='RAW' else 'official_unconstrained_proposal' if stage=='D0' else 'output']
    ids=z['source_face_ids'] if stage=='POST' else np.arange(len(tri))
    ax=axes[row,col]
    for slot,t in zip(ids,tri):
     slot=int(slot);color=highlight.get(slot,'#1686c0' if slot<q['K'] else '#b8b8b8')
     for a,b,_ in segments(t,axis,position,1e-11):
      pp=np.stack([a[proj],b[proj]])
      ax.plot(pp[:,0],pp[:,1],color=color,lw=2.7 if slot in highlight else 1.,
              zorder=4 if slot in highlight else 2)
    ax.set_xlim(xlim);ax.set_ylim(ylim);ax.set_aspect('equal',adjustable='box')
    ax.grid(alpha=.15);ax.set_title(stage+(' (C moved)' if stage=='D0' else ''),fontsize=11)
    ax.set_xlabel(('Z' if view=='side' else 'X')+' model units',fontsize=9)
    if col==0:ax.set_ylabel(rid+'\n'+view+' / Y model units',fontsize=9)
    ax.tick_params(labelsize=8)
 fig.suptitle('Same saved faces across four stages: isolated cut line is not a topology verdict',fontsize=14)
 fig.text(.5,.055,'Top: red slot21 has a shared-edge path to C in POST. Bottom: red slot77 has no shared-edge neighbour; orange slot113 is a larger edge component.',ha='center',fontsize=9)
 fig.text(.5,.025,'Blue C / gray other free. Bottom is a declared local zoom; no mesh editing, D0, GEO or model call. Connectivity checked on complete POST.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.08,1,.95))
 p=OUT/'tracked_sections.png';fig.savefig(p);plt.close(fig)
 (OUT/'tracked_sections.json').write_text(json.dumps({'cases':cases,'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
  'source_hashes':sources,'model_calls':0,'GEO_runs':0,'new_scoring':0},indent=2))
 print(json.dumps({'figure':str(p),'source_files':len(sources)}))

if __name__=='__main__':main()
