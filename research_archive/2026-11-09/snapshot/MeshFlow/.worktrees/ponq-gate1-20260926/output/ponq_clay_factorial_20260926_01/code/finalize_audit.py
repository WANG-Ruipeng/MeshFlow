"""Audit and summarize one frozen run; never repair or re-extract geometry."""
from common import *
import argparse
import csv
import networkx as nx


def protocol_slots(protocol):
    """Global states with optional frozen per-object state overrides."""
    objects = protocol['objects']
    states = protocol['config']['run_states']
    if not isinstance(objects, dict):
        objects = {name: {} for name in objects}
    slots = []
    for name, spec in objects.items():
        local = spec.get('run_states', spec.get('states', states)) if isinstance(spec, dict) else states
        slots.extend((name, state) for state in local)
    if len(slots) != len(set(slots)):
        raise ValueError('Frozen protocol contains duplicate object/state slots')
    return set(slots)


def state_groups(protocol):
    states = {state for _, state in protocol_slots(protocol)}
    explicit = protocol['config'].get('state_groups', {})
    groups = {
        'clean': set(explicit.get('clean', [s for s in states if s.startswith('clean')])),
        'mild': set(explicit.get('mild', [s for s in states if s.endswith('_light')])),
    }
    if any(not group <= states for group in groups.values()):
        raise ValueError('State groups contain states absent from the frozen protocol')
    return groups


def validate_slots(rows, protocol):
    expected = protocol_slots(protocol)
    actual = [(r['object'], r['state']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(dict(reason='ledger_protocol_slot_mismatch',
                              missing=sorted(expected-set(actual)),
                              unexpected=sorted(set(actual)-expected),
                              duplicate_count=len(actual)-len(set(actual))))
    return expected


def vertex_link_witnesses(v, f):
    """Find every nonmanifold link independently with NetworkX.

    Retain edge multiplicity. A connected cycle or boundary path is admissible.
    Unused vertices have no surface link and are not examined here.
    """
    incident = defaultdict(list)
    for face in f:
        for index in set(map(int, face)):
            incident[index].append(face)
    witnesses = []
    for index, faces in sorted(incident.items()):
        graph = nx.MultiGraph()
        for face in faces:
            others = [int(a) for a in face if a != index]
            if len(others) == 2:
                graph.add_edge(*others)
        components = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: c[0])
        degrees = dict(graph.degree())
        ds = list(degrees.values())
        cycle = bool(ds) and all(d == 2 for d in ds)
        path = ds.count(1) == 2 and all(d in (1, 2) for d in ds)
        if len(components) != 1 or not (cycle or path):
            witnesses.append(dict(vertex_id=index, vertex=v[index].tolist(),
                                  link_components=components, link_component_count=len(components),
                                  link_degrees=degrees,
                                  two_disjoint_cycles=len(components) == 2 and cycle))
    return witnesses


def historical_thin_u_comparison(rows, witnesses_by_run):
    """An observed historical failure is a comparison, never an acceptance rule."""
    row = next((r for r in rows if (r['object'], r['state']) == ('thin_u', 'qem_strong')), None)
    if row is None:
        return dict(applicable=False, reason='historical_object_state_not_in_protocol')
    observed = witnesses_by_run.get('thin_u__qem_strong')
    if observed is None:
        return dict(applicable=True, measured=False, blocking=False, status=row['status'], matches_original_signature=None)
    betti = [row.get('beta'+str(i)) for i in range(3)]
    two_cycles = sum(w['two_disjoint_cycles'] for w in observed)
    return dict(applicable=True, measured=True, blocking=False,
                original_signature=dict(betti=[2, 0, 3], nonmanifold_vertices=1, two_cycle_vertices=1),
                observed_signature=dict(betti=betti, nonmanifold_vertices=len(observed), two_cycle_vertices=two_cycles),
                matches_original_signature=betti == [2, 0, 3] and len(observed) == 1 and two_cycles == 1,
                note='Topology/link signature only; numbering and byte identity are separate comparisons.')


def summarize_rows(rows, protocol, fit=()):
    expected = validate_slots(rows, protocol)
    groups = state_groups(protocol)
    active = [r for r in rows if r['status'] == 'EXTRACTED']
    unrun = [r for r in rows if r['status'].startswith('NOT_RUN')]
    extracted = [r for r in rows if r['status'].startswith('EXTRACTED') or r.get('mesh_present') or 'extraction_seconds' in r]
    failed = [r for r in rows if not r['status'].startswith(('EXTRACTED', 'NOT_RUN'))]
    invalid = [r for r in rows if r['status'].startswith('EXTRACTED') and r['status'] != 'EXTRACTED']
    audited = [r for r in rows if r.get('geometry_audited')]
    mild = [r for r in active if r['state'] in groups['mild']]
    def values(records, field):
        return [float(r[field]) for r in records if r.get(field) is not None and np.isfinite(r[field])]
    def total(records, field):
        return sum(values(records, field))
    def positive(field):
        return sum(r.get(field, 0) > 0 for r in audited)
    durations = values(extracted, 'extraction_seconds')
    clean = sum(r['state'] in groups['clean'] for r in active)
    return dict(
        requested_slots=len(expected), ledger_slots=len(rows), status_counts=dict(Counter(r['status'] for r in rows)),
        actual_extractions=len(extracted), successful_extractions=len(active),
        executed_slots=len(rows)-len(unrun), unrun_slots=len(unrun),
        unrun_missing_assets=sum(r['status'] == 'NOT_RUN_MISSING_RAW_ASSET' for r in rows),
        execution_failures=len(failed), invalid_extractions=len(invalid), audited_outputs=len(audited),
        network_training_updates=protocol['config'].get('network_training_updates', 0),
        clean_requested_slots=sum(s in groups['clean'] for _, s in expected),
        clean_extractions=clean, perturbed_extractions=len(active)-clean,
        mild_requested_slots=sum(s in groups['mild'] for _, s in expected), mild_output_count=len(mild),
        exact_intersection_pairs=int(total(audited, 'exact_intersection_pairs')),
        exact_degenerate_faces=int(total(audited, 'exact_degenerate_faces')),
        outputs_with_boundary_edges=positive('boundary_edges'),
        outputs_with_nonmanifold_edges=positive('nonmanifold_edges'),
        outputs_with_nonmanifold_vertices=positive('nonmanifold_vertices'),
        closed_oriented_2manifold_outputs=sum(bool(r.get('closed_oriented_2manifold')) for r in audited),
        topology_matching_outputs=sum(bool(r.get('betti_match')) for r in audited),
        mild_topology_matching_outputs=sum(bool(r.get('betti_match')) for r in mild),
        extraction_seconds_min=min(durations) if durations else None,
        extraction_seconds_median=float(np.median(durations)) if durations else None,
        extraction_seconds_max=max(durations) if durations else None,
        extraction_seconds_total=sum(durations), fitting_seconds_total=total(fit, 'seconds'), fit_count=len(fit),
        exact_audit_seconds_total=total(audited, 'exact_audit_seconds'), metric_seconds_total=total(rows, 'metric_seconds'),
        decision='NO_AUTOMATIC_TRAINING_PROMOTION',
        scope=dict(protocol_version=protocol['config'].get('version'), objects=sorted({o for o, _ in expected}),
                   states=sorted({s for _, s in expected})))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['object', 'state', 'status']
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def perturbation_expectations(protocol, state):
    config = protocol['config']
    explicit = config.get('perturbation_expectations')
    if explicit is not None:
        return explicit.get(state, {})
    # Legacy names keep checks only when their defining doses are frozen.
    expectations = {}
    if state in ('position_light', 'position_strong') and 'position_local_spacing_fraction' in config:
        a = config['position_local_spacing_fraction'][state.endswith('strong')]
        expectations.update(position_shift_over_h_mean=a, position_shift_over_h_max=a)
    if state in ('normal_light', 'normal_strong', 'joint_light') and 'normal_rotation_degrees' in config:
        expectations.update(normal_rotation_deg_mean=config['normal_rotation_degrees'][state.endswith('strong')],
                            normal_magnitude_change_max=0.)
    if state in ('qem_light', 'qem_strong', 'joint_light') and 'quadric_factor_relative_frobenius' in config:
        expectations['factor_relative_frobenius_mean'] = config['quadric_factor_relative_frobenius'][state.endswith('strong')]
    return expectations


def finalize(root):
    root = Path(root)
    (root/'audit').mkdir(exist_ok=True)
    (root/'reports').mkdir(exist_ok=True)
    rows = json.loads((root/'LEDGER_RAW.json').read_text(encoding='utf-8'))
    protocol = json.loads((root/'FROZEN_PROTOCOL.json').read_text(encoding='utf-8'))
    checks, witnesses_by_run, unavailable_checks = [], {}, []

    def check(name, passed, **extra):
        checks.append(dict(name=name, passed=bool(passed), **extra))
        if not passed:
            dump(root/'audit/POSTRUN_CONSISTENCY_CHECKS.json', dict(passed=False, count=len(checks), checks=checks))
            raise AssertionError((name, extra))

    provenance = json.loads((root/'SOURCE_PROVENANCE.json').read_text(encoding='utf-8'))
    src = (root/'vendor/PoNQ_to_mesh.py').read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(src)).encode()+b'\0'+src).hexdigest()
    check('official_extractor_blob_unchanged', blob == provenance['official_extractor_blob'], blob=blob)
    ph = json.loads((root/'FROZEN_PROTOCOL_SHA256.json').read_text(encoding='utf-8'))['sha256']
    check('frozen_protocol_unchanged', sha(root/'FROZEN_PROTOCOL.json') == ph, sha256=ph)
    expected = validate_slots(rows, protocol)
    check('ledger_matches_frozen_slots', True, requested_slots=len(expected))

    for row in rows:
        rd = root/'runs'/f"{row['object']}__{row['state']}"
        row['mesh_present'] = (rd/'mesh.obj').is_file()
        row['geometry_audited'] = False
        if not row['mesh_present']:
            check(rd.name+'_required_output', not row['status'].startswith('EXTRACTED'))
            continue
        ex = json.loads((rd/'exact_audit.json').read_text(encoding='utf-8'))
        v, f = load_mesh(rd/'mesh.obj')
        with np.load(rd/'mesh.npz') as z:
            check(rd.name+'_obj_roundtrip', np.array_equal(v, z['vertices']) and np.array_equal(f, z['faces']))
        mesh_hash = sha(rd/'mesh.obj')
        check(rd.name+'_hash_matches', ex['source_sha256'] == mesh_hash and row.get('mesh_sha256', mesh_hash) == mesh_hash)
        row.update(ex)
        row['mesh_sha256'] = mesh_hash
        # Preserve late evaluation failures while auditing any successfully saved mesh.
        for key, value in mesh_audit(v, f).items():
            row.setdefault(key, value)
        m = trimesh.Trimesh(v, f, process=False)
        check(rd.name+'_euler_independent', m.euler_number == row['euler'])
        check(rd.name+'_edge_watertight_independent', m.is_watertight == (row['boundary_edges'] == 0 and row['nonmanifold_edges'] == 0))
        witnesses = vertex_link_witnesses(v, f)
        witnesses_by_run[rd.name] = witnesses
        check(rd.name+'_vertex_link_count_independent', len(witnesses) == row['nonmanifold_vertices'])
        dump(rd/'vertex_link_witnesses.json', dict(nonmanifold_vertex_count=len(witnesses), witnesses=witnesses))
        row['geometry_audited'] = True
        spec = protocol['objects'].get(row['object'], {}) if isinstance(protocol['objects'], dict) else {}
        betti = row.get('reference_betti', spec.get('expected_betti'))
        if betti is not None:
            row['reference_betti'] = betti
            row['betti_match'] = [row['beta'+str(i)] for i in range(3)] == betti
        for field, target in perturbation_expectations(protocol, row['state']).items():
            if field not in row:
                unavailable_checks.append(dict(run=rd.name, field=field, reason='measurement_not_recorded', status=row['status']))
            else:
                check(rd.name+'_'+field+'_dose', np.isfinite(row[field]) and abs(row[field]-target) < 1e-10,
                      expected=target, measured=row[field])
        if row.get('factor_relative_frobenius_mean', 0) > 0 and 'Q_min_eigenvalue' in row:
            check(rd.name+'_legal_factor_psd', row['Q_min_eigenvalue'] >= -1e-12)

    # Independent brute-force distance oracle over every available reference.
    rng = np.random.default_rng(5619)
    references = sorted((root/'inputs').glob('*/reference.obj'))
    for reference in references:
        obj = reference.parent.name
        v, f = load_mesh(reference)
        tri = v[f]
        q = rng.uniform(v.min(0), v.max(0), (30, 3))
        d, _, _ = SurfaceDistance(v, f).query(q)
        brute = []
        for p in q:
            near = trimesh.triangles.closest_point(tri, np.repeat(p[None, :], len(tri), axis=0))
            brute.append(np.linalg.norm(near-p, axis=1).min())
        err = float(np.max(np.abs(d-brute)))
        check(obj+'_nearest_triangle_vs_bruteforce', err < 1e-12, max_abs_error=err)
        q, _, _ = sample(v, f, 100, 888)
        err = float(SurfaceDistance(v, f).query(q)[0].max())
        check(obj+'_surface_self_distance', err < 1e-12, max_distance=err)

    history = historical_thin_u_comparison(rows, witnesses_by_run)
    dump(root/'audit/HISTORICAL_THIN_U_COMPARISON.json', history)
    dump(root/'audit/NONMANIFOLD_VERTEX_WITNESSES.json', witnesses_by_run)
    dump(root/'audit/POSTRUN_CONSISTENCY_CHECKS.json', dict(
        passed=all(c['passed'] for c in checks), count=len(checks), checks=checks,
        unavailable_checks=unavailable_checks, audited_reference_count=len(references),
        note='PASS means measurements are internally consistent; it is not a geometry-quality pass.'))
    dump(root/'LEDGER.json', rows)
    write_csv(root/'LEDGER.csv', rows)
    groups = state_groups(protocol)
    active = [r for r in rows if r['status'] == 'EXTRACTED']
    write_csv(root/'reports/CLEAN_TABLE.csv', [r for r in active if r['state'] in groups['clean']])
    write_csv(root/'reports/PERTURBATION_TABLE.csv', [r for r in active if r['state'] not in groups['clean']])
    fit = []
    for path in sorted(root.glob('inputs/*/fit_*_trace.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        trace = data.get('trace', [])
        fit.append(dict(object=path.parent.name, budget=int(path.stem.split('_')[1]),
                        seconds=data['fitting_seconds'], initial_cd=trace[0]['cd'] if trace else None,
                        last_recorded_cd=trace[-1]['cd'] if trace else None))
    write_csv(root/'reports/FITTING_TIMES.csv', fit)
    stats = summarize_rows(rows, protocol, fit)
    stats['historical_thin_u_comparison'] = history
    dump(root/'SUMMARY.json', stats)
    print(json.dumps(stats, indent=2))
    print('CONSISTENCY_CHECKS', len(checks))
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    finalize(parser.parse_args().root)
