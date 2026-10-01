"""Trusted source labels aligned to the unchanged Native training stream."""
from pathlib import Path
import hashlib
import json
import numpy as np
from ..data import ArchiveData, file_sha256
from ..artifacts import array_hash


def _seal(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class TrainingGeometryTargets:
    """Read source topology from an explicit prepared archive, without OT or RNG.

    Source coordinates/IDs supervise losses only. They are never appended to
    model_inputs. Target faces and corners remain in the stream's actual order.
    """
    def __init__(self, archive):
        if not isinstance(archive,ArchiveData):
            raise TypeError('An explicitly loaded native_t1.data.ArchiveData is required')
        self.archive=archive; self.parents={}
        if file_sha256(archive.path)!=archive.sha256:
            raise ValueError('Prepared archive changed before source-label loading')
        parents=sorted({p for p,K in archive.cases})
        with np.load(Path(archive.path),allow_pickle=False) as saved:
            for p in parents:
                names=[f'parent{p}_source_vertices',f'parent{p}_source_face_vertex_ids',f'parent{p}_target']
                if any(name not in saved for name in names):
                    raise ValueError('Missing trusted source vertices/face vertex IDs; no coordinate welding fallback')
                V,F,Y=(saved[name].copy() for name in names)
                Y=Y.reshape(112,3,3)
                if V.dtype!=np.float32 or V.ndim!=2 or V.shape[1]!=3 or not np.isfinite(V).all():
                    raise ValueError('Source vertices must be finite FP32[V,3]')
                if F.dtype!=np.int64 or F.shape!=(112,3) or np.any(F<0) or np.any(F>=len(V)):
                    raise ValueError('Source face IDs must be valid int64[112,3]')
                if Y.dtype!=np.float32 or V[F].tobytes()!=Y.tobytes():
                    raise ValueError('Trusted raw-indexed archive does not reconstruct actual target')
                for (parent,K),case in archive.cases.items():
                    if parent==p and case.full_target.tobytes()!=Y.tobytes():
                        raise ValueError('Loaded cases differ from trusted indexed geometry')
                for value in (V,F,Y):value.setflags(write=False)
                self.parents[p]=dict(V=V,F=F,Y=Y)

    def sample(self,sample):
        parent=int(sample['parent_index']);K=int(sample['K']);entry=self.parents[parent]
        if int(sample['donor_index'])!=parent:
            raise ValueError('Interface supervision requires correctly matched context')
        ids=np.asarray(sample['source_face_ids']);corners=np.asarray(sample['corner_permutations'])
        known=sample['known_mask'];valid=sample['valid_mask']
        if sorted(ids.tolist())!=list(range(112)) or not valid.all() or int(known.sum())!=K:
            raise ValueError('N/mask/source map changed')
        if not np.array_equal(np.sort(corners,axis=1),np.broadcast_to(np.arange(3),(112,3))):
            raise ValueError('Invalid corner map')
        vids=np.take_along_axis(entry['F'][ids],corners,axis=1)
        coordinates=entry['V'][vids]
        Y=sample['x1'].reshape(112,3,3);C=sample['context'].reshape(112,3,3)
        if coordinates[~known].tobytes()!=Y[~known].tobytes() or coordinates[known].tobytes()!=C[known].tobytes():
            raise ValueError('Source labels not aligned with actual target slots (OT must reorder Z only)')
        L=self.archive.cases[parent,K].bbox_diagonal
        return dict(free_gt=Y[~known].copy(),known=C[known].copy(),free_ids=vids[~known].copy(),known_ids=vids[known].copy(),
            bbox_L=L,parent=parent,K=K,mapping_sha256=_seal(dict(ids=array_hash(vids),Y=array_hash(Y[~known]),C=array_hash(C[known]),mask=array_hash(known))))
