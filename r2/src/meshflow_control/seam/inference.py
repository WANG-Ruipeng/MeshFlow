"""C-only occupancy/layout entry. No training/source-Y data imports."""
import numpy as np
import torch
from .slot_data import edge_features, inference_layout

ALLOWED_KEYS=frozenset(('C_vertices','C_faces_original','C_faces_model','total_N'))

def predict_layout(head, C_bundle):
    if set(C_bundle)!=ALLOWED_KEYS:
        raise ValueError('Inference accepts exactly C vertices/faces and requested N; source labels/parents/Y are forbidden')
    _,features=edge_features(C_bundle['C_vertices'],C_bundle['C_faces_original'],int(C_bundle['total_N']))
    device=next(head.parameters()).device
    x=torch.from_numpy(features).to(device)
    mask=torch.ones(len(features),dtype=torch.bool,device=device)
    with torch.no_grad():
        logits=head(x,mask)
        occupied=head.occupied(logits,mask).cpu().numpy().astype(bool)
    layout=inference_layout(**C_bundle,occupied=occupied)
    return layout,logits.detach().float().cpu().numpy()
