"""Focused positive ENTRY commit controls; synthetic delegates are no physical pass.

Root executes these under the genuine immutable runtime. The author does not
import this file or execute any test. Every delegate below is an explicit
synthetic stand-in: no Native certificate, BedCommit physics or fit is claimed.
The genuine entry-vs-Dense Native oracle is separate Root qualification Source.
"""
import ast
from dataclasses import replace
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maple_syrup import sediment_bed
from maple_syrup import windowed_shared_bed as shared
from maple_syrup.sediment_bed import BedCommit, BedIntegrationError
from maple_syrup.sediment_event import SedimentEventControl
from maple_syrup.storm import StormControl

_MODULE = Path(shared.__file__)
_BASE = '07cac08'
_SEAMS = ('_positive_bootstrap', '_positive_gate')
_NEW_NAMES = ('_POSITIVE_ENTRY_SCOPE', '_POSITIVE_ENTRY_SOURCE_BYTES',
              '_POSITIVE_ENTRY_SOURCE_NODES', '_POSITIVE_ENTRY_FUNCTIONS',
              '_POSITIVE_ENTRY_SLOTS', '_POSITIVE_ENTRY_PROC_SITES',
              '_PositiveEntryScope', '_entry_bank', '_entry_extra_host',
              '_entry_frontier', 'commit_positive_windowed_entry')
_MAPPED, _HOST, _WORK = 1000, 1024, 7


def _limits(root):
    return shared.PositiveWindowedWaterBridgeLimits(
        max_host_bytes=4 << 30, max_mapped_bytes=128 << 20,
        max_logical_read_bytes=(1 << 63) - 1, max_work=(1 << 63) - 1,
        max_store_bytes=64 << 20, max_run_bytes=128 << 20,
        max_new_versions=1000000, max_entries=4096,
        max_external_backing_bytes=64 << 20, max_event_steps=10000,
        max_retries=10, max_transport_substeps=64, max_report_rows=4096,
        reserve_host_bytes=64 << 20, run_dir=root)


def _context(root):
    return shared.WindowedAvalancheContext(
        geometry=None, grain_classes=None, mass_resolution_kg=1e-12,
        topographic_commit_spec=None, avalanching_spec=None,
        initial_available_fraction=None, window_max_journal_records=320,
        native_iteration_limits=None, native_evidence_limits=None,
        native_solve_limits=None, native_validation_limits=None,
        native_publication_limits=None, native_qualification_limits=None,
        source_identity='0' * 64, bridge_limits=_limits(root))


def _control():
    return SedimentEventControl(storm=StormControl(max_steps=10000),
                                transport_scheme='upwind')


def _bed():
    # Exact holder only; deliberately no Native payload or physical authority.
    column = SimpleNamespace(pending_deep=None)
    return shared.WindowedSharedBedState(column, None, None, None, None, None, None)


def _owner(context, control, thread=None):
    import threading
    return shared._PositiveOwner(context, control, (123, 456),
                                 threading.get_ident() if thread is None else thread,
                                 shared._positive_control_projection(context, control))


class _Lane:
    """Explicit synthetic delegates recording the observed call order."""

    def __init__(self, monkeypatch, state, mapped=_MAPPED, fail=None, result=None):
        self.calls, self.owners, self.fail, self.result = [], [], fail, result
        self.error = RuntimeError('unchanged original delegate failure')
        self.state, self.metadata, self.admission = state, object(), object()
        monkeypatch.setattr(shared, '_POSITIVE_OWNER', None)
        monkeypatch.setattr(shared, '_POSITIVE_ENTRY_SCOPE', None)
        monkeypatch.setattr(shared, '_positive_current_mapped',
                            lambda *args: ((123, 456), mapped, 4 << 30))
        monkeypatch.setattr(shared, '_entry_bank', lambda context: (_HOST, _WORK))
        monkeypatch.setattr(shared, '_external_arrays', self.arrays)
        monkeypatch.setattr(shared, '_metadata', self.metadata_call)
        monkeypatch.setattr(shared, '_positive_gate', self.gate)
        monkeypatch.setattr(shared, '_value_validate', self.validate)
        monkeypatch.setattr(sediment_bed, 'commit_bed', self.commit)

    def _record(self, name):
        self.calls.append(name)
        owner = shared._POSITIVE_OWNER
        if owner is not None and not any(held is owner for held in self.owners):
            self.owners.append(owner)
        if self.fail == name:
            raise self.error

    def arrays(self, **kwargs):
        self._record('arrays')
        assert kwargs['max_rows'] == 4096
        return ()

    def metadata_call(self, state, context):
        self._record('metadata')
        return self.metadata

    def gate(self, state, context, metadata, arrays, role, *, control):
        owner = shared._positive_owner(context, control)
        assert metadata is self.metadata and role == 'commit_trigger'
        # The ENTRY-ONLY seam binds only the installed owner/context/thread.
        assert shared._entry_extra_host(owner, context) == _HOST
        owner.ordinal += 1  # Synthetic stand-in for the genuine gate increment.
        self._record('gate')
        return self.admission

    def validate(self, state, context, current_time_s, *, published):
        assert published is True
        self._record('validate')

    def commit(self, state, context, graph, reference, current_time_s, *, force,
               routing_source_m):
        assert force is True and state is self.state
        self._record('commit')
        if self.result is not None:
            return self.result
        return BedCommit(state=replace(state), graph=graph, did_commit=True, trigger=None,
                         rerouted=False, routing_source_m=routing_source_m)


def _enter(state, context, control, graph=None, source=None):
    graph = SimpleNamespace() if graph is None else graph
    source = np.zeros(4) if source is None else source
    return shared.commit_positive_windowed_entry(state, context, graph, None, 1.0, source,
                                                 control=control)


def test_observed_order_returns_exact_commit_and_actual_counters(tmp_path, monkeypatch):
    context, control, state = _context(tmp_path), _control(), _bed()
    lane = _Lane(monkeypatch, state)
    graph, source = SimpleNamespace(), np.zeros(4)
    result, summary = _enter(state, context, control, graph, source)
    assert lane.calls == ['arrays', 'metadata', 'gate', 'validate', 'commit']
    assert type(result) is BedCommit and result.state is not state
    assert type(result.state) is shared.WindowedSharedBedState
    assert result.state.voxel_column is state.voxel_column
    assert result.graph is graph and result.routing_source_m is source
    owner, = lane.owners
    total = _MAPPED + _HOST + context.bridge_limits.reserve_host_bytes
    assert summary == shared.PositiveWaterAdmissionSummary(1, 2 * _WORK, 0, 0, total)
    assert owner.closed and not owner.poisoned
    assert shared._POSITIVE_OWNER is None and shared._POSITIVE_ENTRY_SCOPE is None


@pytest.mark.parametrize('failing', ['arrays', 'metadata', 'gate', 'validate', 'commit'])
def test_every_delegate_error_keeps_identity_poisons_closes_clears(tmp_path, monkeypatch,
                                                                   failing):
    context, control, state = _context(tmp_path), _control(), _bed()
    lane = _Lane(monkeypatch, state, fail=failing)
    order = ['arrays', 'metadata', 'gate', 'validate', 'commit']
    with pytest.raises(RuntimeError) as caught:
        _enter(state, context, control)
    assert caught.value is lane.error
    assert lane.calls == order[:order.index(failing) + 1]
    owner, = lane.owners
    assert owner.poisoned and owner.closed
    assert shared._POSITIVE_OWNER is None and shared._POSITIVE_ENTRY_SCOPE is None


def test_insufficient_positive_bank_refuses_before_any_native_action(tmp_path, monkeypatch):
    context, control, state = _context(tmp_path), _control(), _bed()
    lane = _Lane(monkeypatch, state, mapped=4 << 30)
    with pytest.raises(BedIntegrationError, match='positive_entry_outer_bank_AS'):
        _enter(state, context, control)
    assert lane.calls == []
    assert shared._POSITIVE_OWNER is None and shared._POSITIVE_ENTRY_SCOPE is None


def test_unrerouted_identity_mismatch_refuses_after_one_commit(tmp_path, monkeypatch):
    context, control, state = _context(tmp_path), _control(), _bed()
    other = BedCommit(state=replace(state), graph=SimpleNamespace(), did_commit=True, trigger=None,
                      rerouted=False, routing_source_m=np.zeros(4))
    lane = _Lane(monkeypatch, state, result=other)
    with pytest.raises(BedIntegrationError, match='unrerouted_original_routing_identity'):
        _enter(state, context, control)
    assert lane.calls.count('commit') == 1
    assert lane.owners[0].poisoned and lane.owners[0].closed


@pytest.mark.parametrize('field', ['owner', 'context', 'thread'])
def test_seam_refuses_wrong_owner_context_or_thread(tmp_path, monkeypatch, field):
    context, control = _context(tmp_path), _control()
    owner = _owner(context, control)
    values = {'owner': owner, 'context': context, 'thread': owner.thread}
    values[field] = {'owner': _owner(context, control), 'context': _context(tmp_path),
                     'thread': owner.thread + 1}[field]
    monkeypatch.setattr(shared, '_entry_bank', lambda context: (_HOST, _WORK))
    monkeypatch.setattr(shared, '_POSITIVE_OWNER', owner)
    monkeypatch.setattr(shared, '_POSITIVE_ENTRY_SCOPE', shared._PositiveEntryScope(**values))
    with pytest.raises(BedIntegrationError, match='positive_entry_scope_binding'):
        shared._entry_extra_host(owner, context)


@pytest.mark.parametrize('existing', ['owner', 'scope'])
def test_nested_existing_owner_or_scope_is_never_replaced_or_cleared(tmp_path, monkeypatch,
                                                                     existing):
    context, control, state = _context(tmp_path), _control(), _bed()
    lane = _Lane(monkeypatch, state)
    held = _owner(context, control)
    if existing == 'owner':
        monkeypatch.setattr(shared, '_POSITIVE_OWNER', held)
    else:
        held = shared._PositiveEntryScope(held, context, held.thread)
        monkeypatch.setattr(shared, '_POSITIVE_ENTRY_SCOPE', held)
    with pytest.raises(BedIntegrationError, match='positive_nonreentrant_exact_lane'):
        _enter(state, context, control)
    assert lane.calls == [] and lane.owners == []
    if existing == 'owner':
        assert shared._POSITIVE_OWNER is held and not held.closed
    else:
        assert shared._POSITIVE_ENTRY_SCOPE is held and shared._POSITIVE_OWNER is None


# --- Source law: new entry AST/text/function/slot census and exact inverses. ---

def _tree(text):
    return ast.parse(text)


def _functions(tree):
    return {node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def _seam(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == '_POSITIVE_ENTRY_SCOPE')


def _new_nodes(text):
    tree, nodes = _tree(text), []
    for node in tree.body:
        targets = (node.target,) if isinstance(node, ast.AnnAssign) else getattr(node, 'targets', ())
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        if names & set(_NEW_NAMES) or getattr(node, 'name', None) in _NEW_NAMES:
            nodes.append(node)
    for name in _SEAMS:
        nodes += [n for n in ast.walk(_functions(tree)[name]) if _seam(n)]
    return nodes


def _old_text():
    root = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=_MODULE.parent,
                               check=True, capture_output=True, text=True).stdout.strip())
    path = _MODULE.relative_to(root).as_posix()
    return subprocess.run(['git', 'show', f'{_BASE}:{path}'], cwd=root, check=True,
                          capture_output=True, text=True).stdout


def _strip_seams(text):
    lines = text.splitlines(keepends=True)
    kept, skip = [], 0
    for line in lines:
        if skip:
            skip -= 1
            continue
        if line.strip().startswith('if _POSITIVE_ENTRY_SCOPE is not None:'):
            skip = 1  # The seam is exactly one guarded statement line.
            continue
        kept.append(line)
    return ''.join(kept)


def test_new_entry_source_stays_within_declared_ceilings():
    text = _MODULE.read_text()
    nodes = _new_nodes(text)
    names = {getattr(n, 'name', None) for n in nodes}
    assert names >= {'_PositiveEntryScope', '_entry_bank', '_entry_extra_host',
                     '_entry_frontier', 'commit_positive_windowed_entry'}
    assert sum(isinstance(n, ast.If) for n in nodes) == 2
    size = sum(len(ast.get_source_segment(text, n).encode()) for n in nodes)
    count = sum(1 for n in nodes for _ in ast.walk(n))
    defs = [d for n in nodes for d in ast.walk(n) if isinstance(d, ast.FunctionDef)]
    assert size <= shared._POSITIVE_ENTRY_SOURCE_BYTES
    assert count <= shared._POSITIVE_ENTRY_SOURCE_NODES
    assert len(defs) <= shared._POSITIVE_ENTRY_FUNCTIONS
    loops = (ast.For, ast.While, ast.AsyncFor, ast.comprehension, ast.Lambda)
    assert not any(isinstance(d, loops) for n in nodes for d in ast.walk(n))
    # Helper-bank census scope stays exact: no new `_positive_`-prefixed def.
    assert not any(d.name.startswith('_positive_') for d in defs)
    codes = [shared._PositiveEntryScope.__init__.__code__] + [
        getattr(shared, name).__code__ for name in
        ('_entry_bank', '_entry_extra_host', '_entry_frontier', 'commit_positive_windowed_entry')]
    slots = sum(c.co_nlocals + c.co_stacksize + len(c.co_cellvars) + len(c.co_freevars)
                for c in codes)
    assert slots <= shared._POSITIVE_ENTRY_SLOTS
    assert (shared._POSITIVE_HELPER_HOST, shared._POSITIVE_HELPER_WORK) == (
        284938012, 1851912394512)


def test_entry_proc_sites_imports_and_single_forced_commit_are_exact():
    tree = _tree(_MODULE.read_text())
    entry = _functions(tree)['commit_positive_windowed_entry']
    calls = [n.func for n in ast.walk(entry) if isinstance(n, ast.Call)]
    called = [f.id for f in calls if isinstance(f, ast.Name)]
    assert called.count('_positive_current_mapped') + called.count('_entry_frontier') == (
        shared._POSITIVE_ENTRY_PROC_SITES)
    assert called.count('commit_bed') == 1 and called.count('_positive_gate') == 1
    commit = next(n for n in ast.walk(entry) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == 'commit_bed')
    force = next(k for k in commit.keywords if k.arg == 'force')
    assert isinstance(force.value, ast.Constant) and force.value.value is True
    imports = [(getattr(n, 'module', None), tuple(a.name for a in n.names))
               for n in ast.walk(entry) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == [(None, ('threading',)),
                       ('maple_syrup.sediment_bed', ('BedCommit', 'commit_bed'))]


def test_two_seams_and_old_functions_are_exact_inverses_of_base():
    old, text = _old_text(), _MODULE.read_text()
    old_defs, new_defs = _functions(_tree(old)), _functions(_tree(text))
    for name in _SEAMS:
        segment = ast.get_source_segment(text, new_defs[name])
        assert _strip_seams(segment) == ast.get_source_segment(old, old_defs[name])
    for name in set(old_defs) - set(_SEAMS):
        assert (ast.get_source_segment(text, new_defs[name])
                == ast.get_source_segment(old, old_defs[name]))
    assert set(new_defs) - set(old_defs) == {
        '_PositiveEntryScope', '_entry_bank', '_entry_extra_host', '_entry_frontier',
        'commit_positive_windowed_entry'}
    # Whole-file inverse: drop the appended entry block and both seam lines.
    head = text[:text.index('\n\n\n# Append-only positive ENTRY commit.') + 1]
    assert _strip_seams(head) == old


def test_complete_positive_event_is_byte_exact_and_seams_default_absent(monkeypatch):
    old, text = _old_text(), _MODULE.read_text()
    name = 'complete_positive_windowed_event'
    assert (ast.get_source_segment(text, _functions(_tree(text))[name])
            == ast.get_source_segment(old, _functions(_tree(old))[name]))
    assert shared._POSITIVE_ENTRY_SCOPE is None
