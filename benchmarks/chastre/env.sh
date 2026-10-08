# Source AFTER selecting your Python/Numba/CuPy environment: first the ORIGINAL gpu_env
# (agent_handoffs/tasks/phase7_matched_benchmark/gpu_env.sh), THEN this file. Do NOT use the old candidate_env.sh.
# Selects actual MAPLE from the verified isolated Chastre candidate (72310c49 + one-file compiled-case receipt patch); installs/edits
# nothing and does not replace the accepted source environment. Override the root with MAPLE_SYRUP_CHASTRE_ROOT
# (default outputs/dependencies/maple_chastre_receipt). verify_maple.py refuses until candidate_digest.txt holds the bound digest.
_chastre_project=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
_chastre_dependency="${MAPLE_SYRUP_CHASTRE_ROOT:-$_chastre_project/outputs/dependencies/maple_chastre_receipt}"
python3 "$_chastre_project/benchmarks/chastre/verify_maple.py" "$_chastre_dependency" || return 1
export PYTHONDONTWRITEBYTECODE=1
export MAPLE_SYRUP_EXPECTED_MAPLE_ROOT="$_chastre_dependency/source"
export GIT_CEILING_DIRECTORIES="$_chastre_dependency"
export PYTHONPATH="$_chastre_project/src:$_chastre_dependency/editable:$_chastre_dependency/source/src${PYTHONPATH:+:$PYTHONPATH}"
unset _chastre_project _chastre_dependency
