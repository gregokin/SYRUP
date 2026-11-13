"""CC1 guards: the historical validators/loaders are AST-identical to the accepted baseline, and the continued validator copies differ from them in EXACTLY the declared statements."""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_continued_checkpoint as cc
from maple_syrup import pond_checkpoint as pc

# AST-dump SHA-256 prefixes of the accepted main-161 definitions (the baseline of this task): these definitions must remain bit-identical in behaviour-bearing source
BASELINE_FC = {"_validate_floating_state": "226a70251bb2c58a01af", "_validate_scalars": "2e192375ef113667fdba", "_check_held": "e081fdcf9c9173bfba0e", "_floating_arrays": "1688705a587102284df6",
               "_read_manifest": "7d88c1cd9b2fe1286529", "_check_canonical_depth": "5ed9443a6a3b89b890c1", "_floating_specs": "2f34f7f963768161eec9", "_dense_native_dims_check": "cd956b65760b1cc04d2d"}
BASELINE_PC = {"_validate_state": "ac0f5734c965449067f8", "_validate_scalars": "01f4910627bbf42f0e3f", "_rebuild_topology": "1dda78f1177d5872828f", "_build_state": "150ef827dcad08ef47eb", "_binding": "f7b5bf4d593680a16735",
               "_extract": "d0d38c48b15667b137b7", "_native_checks": "fc95099e4f268ab190b2"}


def _digest(fn):
    return hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]).encode()).hexdigest()[:20]


def _stmts(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    body = tree.body[1:] if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) else tree.body
    return ast.dump(tree.args), [ast.dump(n) for n in body]


def test_historical_definitions_are_unchanged_from_the_accepted_baseline():
    for name, want in BASELINE_FC.items():
        assert _digest(getattr(fc, name)) == want, name
    for name, want in BASELINE_PC.items():
        assert _digest(getattr(pc, name)) == want, name


def test_continued_validator_differs_from_the_historical_one_in_exactly_the_two_declared_statements():
    old_args, old = _stmts(fc._validate_floating_state)
    new_args, new = _stmts(cc.validate_continued_state)
    assert len(old) == len(new) > 20
    assert "arg='origin'" in new_args and "arg='origin'" not in old_args and "defaults=[Constant(value='dense')]" in old_args and "defaults=[]" in new_args  # one extra required argument, no default
    diff = [i for i, (a, b) in enumerate(zip(old, new, strict=True)) if a != b]
    assert len(diff) == 2
    assert "datum_full" in old[diff[0]] and "_check_continued_origin" in new[diff[0]]
    assert "_validate_scalars" in old[diff[1]] and "_validate_continued_scalars" in new[diff[1]]


def test_continued_scalar_rule_differs_from_the_historical_one_in_exactly_the_refresh_statement():
    old_args, old = _stmts(fc._validate_scalars)
    new_args, new = _stmts(cc._validate_continued_scalars)
    assert old_args == new_args and len(old) == len(new) > 10
    diff = [i for i, (a, b) in enumerate(zip(old, new, strict=True)) if a != b]
    assert len(diff) == 1 and "n_refreshes" in old[diff[0]] and "n_refreshes" in new[diff[0]]


def test_historical_defaults_are_the_none_origin():
    for fn in (fc.save_floating_pond_checkpoint, fc.load_floating_pond_checkpoint):
        p = inspect.signature(fn).parameters["original_topology"]
        assert p.default is None and p.kind is inspect.Parameter.KEYWORD_ONLY
    assert fc.READ_SCHEMAS == (fc.SCHEMA, fc.WINDOWED_SCHEMA, fc.SCHEMA_V2, fc.WINDOWED_SCHEMA_V2, fc.SCHEMA_V3, fc.WINDOWED_SCHEMA_V3) and not set(fc.CONTINUED_SCHEMAS) & set(fc.READ_SCHEMAS)
