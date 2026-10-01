# Source after selecting your Python/Numba/CuPy environment. This selects actual
# MAPLE from the verified isolated snapshot; it installs or edits nothing.
_syrup_phase7d_project=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
_syrup_phase7d_dependency="$_syrup_phase7d_project/outputs/dependencies/maple_72310c49"
python3 "$_syrup_phase7d_project/benchmarks/phase7d/verify_candidate.py" "$_syrup_phase7d_dependency" || return 1
export PYTHONDONTWRITEBYTECODE=1
export MAPLE_SYRUP_EXPECTED_MAPLE_ROOT="$_syrup_phase7d_dependency/source"
export GIT_CEILING_DIRECTORIES="$_syrup_phase7d_dependency"
export PYTHONPATH="$_syrup_phase7d_project/src:$_syrup_phase7d_dependency/editable:$_syrup_phase7d_dependency/source/src${PYTHONPATH:+:$PYTHONPATH}"
unset _syrup_phase7d_project _syrup_phase7d_dependency
