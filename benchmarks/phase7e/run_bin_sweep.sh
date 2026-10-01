#!/usr/bin/env bash
# Sequential characteristic-bin sweep on the ACCEPTED dependency (MAPLE 72310c49, SYRUP source as checked out).
# Usage: bash benchmarks/phase7e/run_bin_sweep.sh <output_root> [bins...]
set -euo pipefail
cd /home/okin/SYRUP
root="$1"; shift
bins=("$@")
if [ ${#bins[@]} -eq 0 ]; then bins=(1 2 4 8 16 32 64 128); fi
source agent_handoffs/tasks/phase6_complete_event/env.sh
source benchmarks/phase7d/candidate_env.sh
mkdir -p "$root"
for b in "${bins[@]}"; do
  out="$root/b$b"
  if [ -e "$out" ]; then echo "skip existing $out"; continue; fi
  echo "=== bins=$b start $(date -u +%FT%TZ)"
  "$SYRUP_PYTHON" benchmarks/phase7b/run_characteristic_benchmark.py --bins "$b" --output "$out" \
      --allow-maple-source-change > "$root/b$b.log" 2>&1 || echo "FAILED bins=$b (see $root/b$b.log)"
  echo "=== bins=$b end $(date -u +%FT%TZ)"
done
echo "sweep complete $(date -u +%FT%TZ)"
