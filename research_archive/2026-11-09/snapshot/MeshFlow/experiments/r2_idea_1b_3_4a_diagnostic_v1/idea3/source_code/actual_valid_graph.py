def actual_valid_graph(vertices, faces, K, pairs):
    triangles = vertices[faces].astype(np.float64)
    areas = .5 * np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    duplicate_groups = collections.defaultdict(list)
    for i, face in enumerate(faces):
        duplicate_groups[tuple(sorted(map(int, face)))].append(i)
    duplicate_faces = {i for group in duplicate_groups.values() if len(group) > 1 for i in group}
    edges = collections.defaultdict(list)
    for i, face in enumerate(faces):
        for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges[tuple(sorted((int(u), int(v))))].append((i, int(u), int(v)))
    cf = {tuple(map(int, p['pair'])): p['classification'] for p in pairs['CF']['contacts']}
    ff = {tuple(sorted(map(int, p['pair']))): p['classification'] for p in pairs['FF']['contacts']}
    adjacent = collections.defaultdict(set)
    rows = []
    for (u, v), incidents in sorted(edges.items()):
        if not any(i >= K for i, _, _ in incidents):
            continue
        has_C = any(i < K for i, _, _ in incidents)
        is_cf = has_C and any(i >= K for i, _, _ in incidents)
        reasons, classification, opposed = [], 'NA_NONMANIFOLD_OR_BOUNDARY', None
        length = float(np.linalg.norm(vertices[u].astype(float)-vertices[v]))
        if len(incidents) != 2:
            reasons.append('NOT_TWO_INCIDENT_FACES')
        else:
            (i, a, b), (j, c, d) = incidents
            opposed = a == d and b == c
            if i == j or u == v or length <= 0 or areas[i] <= 0 or areas[j] <= 0:
                reasons.append('DEGENERATE')
            if i in duplicate_faces or j in duplicate_faces:
                reasons.append('DUPLICATE_FACE')
            if is_cf:
                known, free = (i, j) if i < K else (j, i)
                classification = cf.get((known, free-K), 'UNKNOWN_MISSING_SHARED_EDGE_PAIR')
            else:
                classification = ff.get(tuple(sorted((i-K, j-K))), 'UNKNOWN_MISSING_SHARED_EDGE_PAIR')
            if classification not in LEGAL:
                reasons.append('ILLEGAL' if classification in ILLEGAL else 'UNKNOWN_OR_NOT_CONTACT')
            if not reasons:
                adjacent[i].add(j)
                adjacent[j].add(i)
        rows.append(dict(vertex_u=u, vertex_v=v, length_model=length, incident_face_ids=[i for i, _, _ in incidents],
            incidence=len(incidents), is_C_F=is_cf, valid_edge=not reasons, winding_opposite=opposed,
            pair_classification=classification, rejected_reasons=reasons))
    seen = set(range(K))
    todo = list(range(K))
    while todo:
        face = todo.pop()
        for other in adjacent[face] - seen:
            seen.add(other)
            todo.append(other)
    reached = sorted(i for i in seen if i >= K)
    cf_rows = [r for r in rows if r['is_C_F']]
    valid_cf = [r for r in cf_rows if r['valid_edge']]
    summary = dict(C_F_shared_edge_count=len(cf_rows), C_F_shared_edge_total_length=sum(r['length_model'] for r in cf_rows),
        C_F_valid_edge_count=len(valid_cf), C_F_has_valid_edge=int(bool(valid_cf)),
        C_F_valid_edge_total_length=sum(r['length_model'] for r in valid_cf),
        C_F_valid_opposite_winding_edge_count=sum(bool(r['winding_opposite']) for r in valid_cf),
        C_F_nonmanifold_edge_count=sum(r['incidence'] > 2 for r in cf_rows),
        C_F_rejected_duplicate_edge_count=sum('DUPLICATE_FACE' in r['rejected_reasons'] for r in cf_rows),
        C_F_rejected_illegal_edge_count=sum('ILLEGAL' in r['rejected_reasons'] for r in cf_rows),
        C_F_rejected_unknown_edge_count=sum('UNKNOWN_OR_NOT_CONTACT' in r['rejected_reasons'] for r in cf_rows),
        graph_valid_edges=sum(r['valid_edge'] for r in rows), reached_free_faces=len(reached),
        reached_free_area=float(areas[reached].sum()), free_area=float(areas[K:].sum()),
        reached_free_area_fraction=float(areas[reached].sum()/areas[K:].sum()) if areas[K:].sum() > 0 else None,
        exact_duplicate_faces=len(duplicate_faces), nonmanifold_edges=sum(len(v) > 2 for v in edges.values()),
        boundary_edges=sum(len(v) == 1 for v in edges.values()),
        status='PASS' if not any('UNKNOWN_OR_NOT_CONTACT' in r['rejected_reasons'] for r in rows if r['incidence'] == 2) else 'PARTIAL_UNKNOWN_EDGE_REJECTED')
    return summary, rows, reached

ILLEGAL = frozenset(('proper_transverse_penetration', 'coplanar_positive_area_overlap'))

LEGAL = frozenset(('shared_edge_contact', 'shared_vertex_contact', 'coplanar_boundary_or_tolerance_contact',
                   'noncoplanar_edge_or_tangent_contact', 'point_or_tangent_contact'))
