from pathlib import Path
import hashlib
import numpy as np
root = Path('/mnt/e/MeshFlow/downloaded_data/ss_overfit')
for split in ('train', 'test'):
    items = np.load(root / 'split' / (split + '.npz'), allow_pickle=True)['npz_list'].tolist()
    missing = []
    lengths = []
    hashes = set()
    for item in items:
        path = root / 'objaverse_occ_v5_ids' / (item['uid'] + '.npz')
        if not path.exists():
            missing.append(item['uid'])
            continue
        with np.load(path, allow_pickle=True) as mesh:
            lengths.append(int(mesh['faces_num']))
            hashes.add(hashlib.sha256(mesh['vertices'].tobytes() + mesh['faces'].tobytes()).hexdigest())
    print(split, 'samples=', len(items), 'missing=', len(missing), 'faces=', sorted(set(lengths)), 'unique_geometry=', len(hashes))