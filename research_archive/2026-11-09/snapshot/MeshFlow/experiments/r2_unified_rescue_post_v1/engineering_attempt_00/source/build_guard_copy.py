"""One-time source derivation, kept for review; never edits original GEO_C."""
from pathlib import Path
def build():
    root=Path(__file__).resolve().parents[2]
    source=root/'src/meshflow_control/geo_c.py'
    text=source.read_text(encoding='utf-8')
    text=text.replace('def process(raw, K, radius=RADIUS):','def process(raw, K, radius=RADIUS, protected_faces=None):')
    text=text.replace('    radius = float(radius)\n','''    radius = float(radius)
    protected_faces = np.zeros(len(raw), bool) if protected_faces is None else np.asarray(protected_faces, bool)
    if protected_faces.shape != (len(raw),) or protected_faces[:K].any():
        raise ValueError('Only valid free faces may be protected')
    cluster_faces = {}
    guard_records = []
    def conflict(i, j):
        return bool(protected_faces[i//3] and i//3 in cluster_faces.get(_bits(points[j]), set()))
    def claim(i, j):
        if protected_faces[i//3]:
            cluster_faces.setdefault(_bits(points[j]), set()).add(i//3)
''')
    text=text.replace('''            corrected[i] = points[j]
            target[i] = j
            kind[i] = "unique_C_anchor"''','''            if conflict(i, j):
                guard_records.append(dict(corner=i, target=j, kind="C_anchor_same_face_rejected"))
                kind[i] = "guard_C_unchanged"
                claim(i, i)
            else:
                corrected[i] = points[j]
                target[i] = j
                kind[i] = "unique_C_anchor"
                claim(i, j)''')
    text=text.replace('''            kind[i] = "ambiguous_C_unchanged"
        else:''','''            kind[i] = "ambiguous_C_unchanged"
            claim(i, i)
        else:''')
    text=text.replace('''            position = min(hits, key=lambda h: (float(distances[h]), int(h)))
            j = representatives[int(position)]
            corrected[i] = points[j]
            target[i] = j
            kind[i] = "free_to_representative"
        else:
            representatives.append(i)
            kind[i] = "free_representative"''','''            ordered = sorted(hits, key=lambda h: (float(distances[h]), int(h)))
            valid = []
            for h in ordered:
                j = representatives[int(h)]
                if conflict(i, j):
                    guard_records.append(dict(corner=i, target=j, kind="free_cluster_same_face_rejected"))
                else:
                    valid.append(h)
            if valid:
                j = representatives[int(valid[0])]
                corrected[i] = points[j]
                target[i] = j
                kind[i] = "free_to_representative"
                claim(i, j)
            else:
                representatives.append(i)
                kind[i] = "guard_free_representative"
                claim(i, i)
        else:
            representatives.append(i)
            kind[i] = "free_representative"
            claim(i, i)''')
    text=text.replace('''    return dict(output=output, vertices=vertices, faces=faces,''','''    stats.update(anti_collapse_protected_faces=int(protected_faces.sum()),
        anti_collapse_rejections=len(guard_records),
        protected_deleted_faces=int((protected_faces & ~keep).sum()),
        guard_version="raw_extent_protected_deterministic_geo_v1")
    return dict(guard_records=guard_records, output=output, vertices=vertices, faces=faces,''')
    text='''# Derived from original GEO_C; only same-protected-face cluster guard added.
# Original representative ordering, radius, cleanup and source maps retained.
'''+text
    target=Path(__file__).with_name('guard_geo.py')
    if target.exists():raise FileExistsError(target)
    target.write_text(text,encoding='utf-8')
if __name__=='__main__':build()
