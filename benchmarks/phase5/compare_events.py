"""Compare provenance-matched SYRUP timestep runs; not a MAHLERAN storm comparison."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def comparison_identity(summary):
    config = copy.deepcopy(summary['resolved_config'])
    del config['syrup']['storm_control']['max_dt_s']
    return {
        'config_except_max_dt': config,
        'mahleran_xml_sha256': summary['sources']['mahleran_xml']['sha256_before'],
        'maple_artifact_sha256': summary['case']['maple_artifact_sha256'],
        'maple_sha256': summary['provenance']['maple']['package_source_digest']['digest_sha256'],
        'syrup_sha256': summary['provenance']['maple_syrup']['package_source_digest']['digest_sha256'],
    }


def read_run(path):
    summary_path = path / 'sediment_summary.json'
    summary = json.loads(summary_path.read_text())
    for name, expected in summary['outputs'].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'output hash mismatch: {path / name}')
    if not summary['sediment']['closure']['closed']:
        raise ValueError(f'failed sediment closure: {path}')
    with np.load(path / 'hydrograph.npz') as data:
        hydro = {key: data[key].copy() for key in data.files}
    with np.load(path / 'final_state.npz') as data:
        dz = (data['committed_elevation_m'] - data['initial_committed_elevation_m']).copy()
    return path, summary, hydro, dz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.output_prefix.parent / (args.output_prefix.name + suffix) for suffix in ('.json', '.svg', '.png')]
    if any(path.exists() for path in outputs):
        parser.error('output files must be new')
    runs = sorted((read_run(path) for path in args.runs), key=lambda r: r[1]['time']['max_dt_s'], reverse=True)
    if len(runs) < 2 or len({r[1]['time']['max_dt_s'] for r in runs}) != len(runs):
        parser.error('provide at least two distinct timesteps')
    identity = comparison_identity(runs[0][1])
    if any(comparison_identity(run[1]) != identity for run in runs[1:]):
        raise ValueError('runs differ in source, case, or resolved settings beyond max_dt_s')
    reference = runs[-1][1]
    ref_export = np.asarray(reference['sediment']['closure']['export_actual_kg'])
    records = []
    for path, summary, hydro, dz in runs:
        sed = summary['sediment']
        exported = np.asarray(sed['closure']['export_actual_kg'])
        records.append({
            'path': str(path), 'summary_sha256': hashlib.sha256((path / 'sediment_summary.json').read_bytes()).hexdigest(),
            'dt_s': summary['time']['max_dt_s'], 'accepted_steps': summary['time']['n_accepted_steps'],
            'rejected_attempts': summary['time']['n_rejected_attempts'],
            'max_transport_substeps': summary['time']['max_transport_substeps_used'],
            'max_sediment_courant': sed['max_sediment_courant'],
            'runoff_m3': summary['budget']['export_m3'],
            'water_residual_m3': summary['budget']['water_residual_m3'],
            'export_kg': float(exported.sum()), 'export_by_class_kg': exported.tolist(),
            'relative_export_difference_vs_finest': float(exported.sum() / ref_export.sum() - 1.) if ref_export.sum() else None,
            'relative_class_export_difference_vs_finest': [float(a / b - 1.) if b else None for a, b in zip(exported, ref_export, strict=True)],
            'gross_pickup_kg': sed['totals']['actual_pickup'], 'gross_deposition_kg': sed['totals']['deposition_actual'],
            'final_mobile_kg': sed['final_mobile_kg'], 'peak_mobile_kg': sed['peak_mobile_kg'],
            'peak_sediment_export_kg_s': sed['peak_export_rate_kg_s'], 'sediment_peak_time_s': sed['time_of_peak_export_s'],
            'peak_runoff_m3_s': summary['final_state']['peak_outlet_discharge_m3_s'],
            'runoff_peak_time_s': summary['final_state']['time_of_peak_outlet_discharge_s'],
            'max_abs_class_residual_kg': float(np.max(np.abs(sed['closure']['residual_kg']))),
            'commits': summary['commits']['n_commits'], 'direction_changes': summary['commits']['rerouted_cells_total'],
            'morphological_erosion_kg': sed['net_erosion_kg'], 'morphological_deposition_kg': sed['net_deposition_kg'],
            'max_abs_elevation_difference_vs_finest_m': float(np.max(np.abs(dz - runs[-1][3]))),
            'step_loop_wall_s': summary['timings']['step_loop']['wall_s'],
            'max_decay_exponent': sed['max_decay_exponent_v_r_dt'],
            'output_sha256': summary['outputs'],
            'curve': {key: hydro[key].tolist() for key in
                      ('t_s', 'outlet_discharge_m3_s', 'sed_cumulative_export_kg', 'sed_export_rate_kg_s')},
        })
    report = {'scope': 'SYRUP timestep sensitivity on one fixed case; finest run is a numerical reference, not truth',
              'identity': identity, 'runs': records,
              'comparison_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'timing_note': 'consult acceptance record for concurrent-run status; refinement timings are not a controlled speed comparison'}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for i, ((_, summary, hydro, _), record) in enumerate(zip(runs, records, strict=True)):
        label = f"dt ≤ {record['dt_s']:g} s"
        axes[0, 0].plot(hydro['t_s'] / 60., hydro['outlet_discharge_m3_s'], label=label)
        axes[0, 1].plot(hydro['t_s'] / 60., hydro['sed_cumulative_export_kg'], label=label)
        x = np.arange(len(ref_export)) + (i - (len(runs) - 1) / 2) * .8 / len(runs)
        axes[1, 0].bar(x, record['export_by_class_kg'], width=.8 / len(runs), label=label)
    axes[0, 0].set(xlabel='Time (minutes)', ylabel='Outlet discharge (m³/s)')
    axes[0, 1].set(xlabel='Time (minutes)', ylabel='Cumulative sediment export (kg)')
    axes[1, 0].set(xlabel='Grain class (fine to coarse)', ylabel='Exported mass (kg)', yscale='log',
                   xticks=np.arange(len(ref_export)), xticklabels=reference['sediment']['class_ids'])
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    dz_mm = runs[-1][3] * 1000.
    limit = max(float(np.max(np.abs(dz_mm))), 1e-12)
    ny, nx = dz_mm.shape
    dx = reference['domain']['dx_m']
    mesh = axes[1, 1].imshow(dz_mm, origin='lower', extent=(0., nx * dx, 0., ny * dx),
                             cmap='RdBu_r', vmin=-limit, vmax=limit)
    axes[1, 1].set(xlabel='Distance east (m)', ylabel='Distance north (m)',
                   title=f"Final bed change, dt ≤ {records[-1]['dt_s']:g} s")
    fig.colorbar(mesh, ax=axes[1, 1], label='Elevation change (mm)')
    fig.suptitle('MAPLE-SYRUP: Plot 1 timestep comparison')
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    fig.savefig(outputs[1])
    fig.savefig(outputs[2], dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
