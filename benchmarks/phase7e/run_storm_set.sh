#!/usr/bin/env bash
# Sequential Stage 1 storm set (no other benchmark may run concurrently):
#   1. accepted MAPLE 72310c49 baseline, 32 bins
#   2. Phase 7e candidate (benchmarks/phase7e/candidate_digest.txt), default public path
#   3. Phase 7e candidate with runner-level touched-inventory / trusted-cache opt-in
set -euo pipefail
cd /home/okin/SYRUP
root="${1:-outputs/phase7e/stage1}"
mkdir -p "$root"
run() {  # name env-kind extra-args...
  local name="$1"; local kind="$2"; shift 2
  if [ -e "$root/$name" ]; then echo "skip existing $root/$name"; return; fi
  echo "=== $name start $(date -u +%FT%TZ)"
  ( source agent_handoffs/tasks/phase6_complete_event/env.sh
    if [ "$kind" = accepted ]; then source benchmarks/phase7d/candidate_env.sh
    else export PYTHONPATH=/tmp/syrup-numba; source benchmarks/phase7e/candidate_env.sh; fi
    "$SYRUP_PYTHON" benchmarks/phase7e/run_storm.py --bins 32 --output "$root/$name" --allow-maple-source-change "$@" ) \
    > "$root/$name.log" 2>&1 || echo "FAILED $name (see $root/$name.log)"
  echo "=== $name end $(date -u +%FT%TZ)"
}
run baseline_b32 accepted
run candidate_b32 candidate
run candidate_touched_b32 candidate --touched-inventory
echo "storm set complete $(date -u +%FT%TZ)"
