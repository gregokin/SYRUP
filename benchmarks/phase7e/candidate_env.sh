# Source AFTER selecting your Python/Numba/CuPy environment (e.g. agent_handoffs/tasks/phase6_complete_event/env.sh).
# Selects actual MAPLE from the verified isolated Phase 7e candidate (selective bed exchange); installs/edits nothing.
# Override the candidate root with MAPLE_SYRUP_PHASE7E_ROOT (default outputs/dependencies/maple_phase7e_reviewed).
_p7e_project=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
_p7e_dependency="${MAPLE_SYRUP_PHASE7E_ROOT:-$_p7e_project/outputs/dependencies/maple_phase7e_reviewed}"
python3 "$_p7e_project/benchmarks/phase7e/verify_candidate.py" "$_p7e_dependency" || return 1
export PYTHONDONTWRITEBYTECODE=1
export MAPLE_SYRUP_EXPECTED_MAPLE_ROOT="$_p7e_dependency/source"
export GIT_CEILING_DIRECTORIES="$_p7e_dependency"
export PYTHONPATH="$_p7e_project/src:$_p7e_dependency/editable:$_p7e_dependency/source/src${PYTHONPATH:+:$PYTHONPATH}"
unset _p7e_project _p7e_dependency
