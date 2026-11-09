"""Read-only observer insertion into the frozen layout optimizer.

Only one keyword-only argument and one callback statement are injected. The
original clock, steps, guard, losses, accepted candidates, and best-feasible
selection remain unchanged. Observer time is therefore part of later elapsed
values and the original wall-time safety cap. Callback returns are ignored;
callback exceptions propagate as runtime failures, never replacement outputs.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path

SOURCE_PATH = Path(__file__).resolve().parents[1] / 'layout_value' / 'optimizer.py'
EXPECTED_SOURCE_SHA256 = '5a21f093ba7a1e35906127e4832cdbbd213da05c39fe4b1b8879e196754a8729'
_COPY_GLOBAL = '__time_quality16_observer_deepcopy'
CALLBACK_FIELDS = ('iteration', 'current_u', 'delivered_u', 'faces', 'row',
                   'best_feasible_iteration')
INSERTION_SOURCE = '''if on_record is not None:
    on_record(
        iteration=iteration,
        current_u=u_np.copy(),
        delivered_u=(u_np if best_u is None else best_u).copy(),
        faces=f_np.copy(),
        row=__time_quality16_observer_deepcopy(row),
        best_feasible_iteration=best_iteration,
    )
'''


def _dump(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _function(tree):
    found = [x for x in tree.body if isinstance(x, ast.FunctionDef)
             and x.name == 'optimize_layout']
    if len(found) != 1:
        raise ValueError('Expected exactly one top-level optimize_layout')
    return found[0]


def _record(function):
    found = [x for x in function.body if isinstance(x, ast.FunctionDef)
             and x.name == 'record']
    if len(found) != 1:
        raise ValueError('Expected exactly one nested record function')
    record = found[0]
    if not isinstance(record.body[-1], ast.Return) or not isinstance(record.body[-1].value, ast.Name) \
            or record.body[-1].value.id != 'row':
        raise ValueError('Original record must end with return row')
    return record


def validate_transform(original, observed):
    """Undo exactly the permitted two edits; every other AST node must match."""
    restored = copy.deepcopy(observed)
    source = _function(original)
    target = _function(restored)
    args = target.args
    if len(args.kwonlyargs) != len(source.args.kwonlyargs) + 1 \
            or args.kwonlyargs[-1].arg != 'on_record' \
            or args.kwonlyargs[-1].annotation is not None \
            or not isinstance(args.kw_defaults[-1], ast.Constant) \
            or args.kw_defaults[-1].value is not None:
        raise ValueError('Only keyword-only on_record=None may be appended')
    args.kwonlyargs.pop()
    args.kw_defaults.pop()
    record = _record(target)
    expected = ast.parse(INSERTION_SOURCE).body[0]
    if len(record.body) < 2 or _dump(record.body[-2]) != _dump(expected):
        raise ValueError('Callback must be the exact insertion immediately before return row')
    record.body.pop(-2)
    if _dump(restored) != _dump(original):
        raise ValueError('Observer transformation changed other optimizer AST statements')
    return True


def instrument_tree(original):
    """Pure AST operation; no execution, imports, geometry, or clock changes."""
    function = _function(original)
    if any(isinstance(x, ast.Name) and x.id in ('on_record', _COPY_GLOBAL)
           for x in ast.walk(function)) \
            or any(x.arg == 'on_record' for x in ast.walk(function) if isinstance(x, ast.arg)):
        raise ValueError('Reserved observer identifier already present')
    record = _record(function)
    # The preceding snapshot branch must remain before the callback.
    if not isinstance(record.body[-2], ast.If) \
            or 'snapshots.append' not in ast.unparse(record.body[-2]):
        raise ValueError('Expected original snapshot materialization immediately before return')
    observed = copy.deepcopy(original)
    function = _function(observed)
    function.args.kwonlyargs.append(ast.arg(arg='on_record', annotation=None))
    function.args.kw_defaults.append(ast.Constant(value=None))
    _record(function).body.insert(-1, ast.parse(INSERTION_SOURCE).body[0])
    ast.fix_missing_locations(observed)
    validate_transform(original, observed)
    return observed


def _source_tree():
    raw = SOURCE_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        raise ValueError('Frozen optimizer source SHA256 changed')
    parsed = ast.parse(raw.decode('utf-8'), filename=str(SOURCE_PATH))
    function = _function(parsed)
    return raw, ast.Module(body=[copy.deepcopy(function)], type_ignores=[])


def source_identity():
    raw, original = _source_tree()
    observed = instrument_tree(original)
    return {
        'original_source_path': str(SOURCE_PATH.resolve()),
        'original_source_sha256': hashlib.sha256(raw).hexdigest(),
        'original_function_AST_sha256': hashlib.sha256(_dump(original).encode()).hexdigest(),
        'observed_function_AST_sha256': hashlib.sha256(_dump(observed).encode()).hexdigest(),
        'adapter_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'only_allowed_changes': ['append keyword-only on_record=None',
                                 'insert callback after snapshot and before record return'],
        'callback_keyword_fields': list(CALLBACK_FIELDS),
        'insertion_source': INSERTION_SOURCE,
        'array_arguments_are_independent_copies': True,
        'row_is_deep_copy': True,
        'return_value_used_for_control': False,
        'clock_or_optimizer_selection_changed': False,
        'observer_cost_in_later_elapsed_and_original_safety_cap': True,
        'callback_exceptions_propagate': True,
    }


def build_observed_optimizer():
    """Compile only the transformed function in the original helper namespace."""
    _, original = _source_tree()
    observed = instrument_tree(original)
    from ..layout_value import optimizer as original_module
    if Path(original_module.__file__).resolve() != SOURCE_PATH.resolve():
        raise ValueError('Imported optimizer module comes from a different source path')
    namespace = dict(vars(original_module))
    if _COPY_GLOBAL in namespace:
        raise ValueError('Reserved observer helper already present in optimizer namespace')
    namespace[_COPY_GLOBAL] = copy.deepcopy
    exec(compile(observed, str(SOURCE_PATH) + ' [observer-only AST]', 'exec'), namespace)
    function = namespace['optimize_layout']
    function.__observer_identity__ = source_identity()
    return function


def optimize_layout_observed(y, initial_u, faces, quality_floor, device='cuda',
                             max_steps=500, max_seconds=300, seed=71310,
                             snapshot_steps=(0, 25, 100, 250, 500), *, on_record=None):
    """Original API plus a synchronous, non-controlling keyword-only observer.

    on_record(**payload) is called at every completed record, including iteration
    zero, after any requested snapshot has been copied. delivered_u is the output
    the original best-feasible rule would return at this point, or current_u if
    no feasible state has yet been found. No fallback or audit result is read.
    """
    if on_record is not None and not callable(on_record):
        raise TypeError('on_record must be callable or None')
    function = build_observed_optimizer()
    return function(y, initial_u, faces, quality_floor, device=device,
                    max_steps=max_steps, max_seconds=max_seconds, seed=seed,
                    snapshot_steps=snapshot_steps, on_record=on_record)