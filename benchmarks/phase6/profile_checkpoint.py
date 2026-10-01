"""CPU bundle I/O costs, separate from event stepping and compilation."""
import argparse
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np

from maple_syrup.case_import import verify_plot1_case
from maple_syrup.checkpoint import load_checkpoint, save_checkpoint
from maple_syrup.sediment_experiment import prepare_verified_sediment_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    verified = verify_plot1_case(args.case_dir)
    setup = prepare_verified_sediment_case(verified)
    manifest = json.loads((args.checkpoint / 'checkpoint.json').read_text())
    context, column, sediment = (setup[k] for k in ('context', 'column', 'sediment'))
    identity = manifest['identity']
    state = load_checkpoint(args.checkpoint, context, column, sediment, identity)
    setup_s = time.perf_counter() - start
    loads, saves = [], []
    with tempfile.TemporaryDirectory(prefix='syrup-checkpoint-profile-') as tmp:
        for i in range(5):
            dest = Path(tmp) / str(i)
            start = time.perf_counter()
            save_checkpoint(dest, state, context, column, sediment, identity)
            saves.append(time.perf_counter() - start)
            start = time.perf_counter()
            restored = load_checkpoint(dest, context, column, sediment, identity)
            loads.append(time.perf_counter() - start)
            np.testing.assert_array_equal(state.result.state.bed.voxel_column.mass_kg,
                                          restored.result.state.bed.voxel_column.mass_kg)
            del restored
    report = {'checkpoint': str(args.checkpoint), 'source_sha256': identity['syrup_source_sha256'],
              'bundle_bytes': sum(p.stat().st_size for p in args.checkpoint.iterdir() if p.is_file()),
              'setup_and_first_load_wall_s': setup_s, 'save_wall_s': saves, 'load_wall_s': loads,
              'median_save_wall_s': float(np.median(saves)), 'median_load_wall_s': float(np.median(loads)),
              'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'scope': 'Plot1 CPU only; peak RSS includes imports, prepared case and simultaneous source/restored state; warm filesystem cache; no fsync/power-loss persistence claim',
              'gpu': 'not exercised'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
