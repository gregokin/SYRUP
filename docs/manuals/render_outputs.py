"""Small post-processing examples for documented SYRUP NPZ outputs.

This does not run the model, reconstruct checkpoints, infer flux from mass,
or create temporal maps that were not saved. Requires NumPy and Matplotlib.
See SYRUP_User_Guide.docx. Run --help for commands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import zipfile

import numpy as np


def inventory(path):
    """Inspect NPY headers inside an NPZ without loading the large payloads."""
    rows = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(stream)
            rows.append({'key': name[:-4], 'shape': list(shape), 'dtype': str(dtype),
                         'fortran_order': fortran})
    return rows


def select_map(array, class_index=None, sum_classes=False):
    a = np.asarray(array)
    if a.ndim == 3:
        if class_index is not None:
            if not 0 <= class_index < a.shape[-1]:
                raise ValueError('class index is outside the last array dimension')
            a = a[..., class_index]
        elif sum_classes:
            a = a.sum(axis=-1)
        else:
            raise ValueError('3D class maps require --class-index (zero-based) or --sum-classes')
    elif class_index is not None or sum_classes:
        raise ValueError('class selection applies only to a 3D (row, column, class) map')
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError('expected a finite two-dimensional physical map')
    return a


def snapshots(path, field):
    """Yield actual saved maps in numerical-time order for two source formats."""
    with np.load(path, allow_pickle=False) as data:
        if 't_s' in data and field in data:
            times = np.asarray(data['t_s'])
            stack = data[field]
            if times.ndim != 1 or stack.shape[0] != times.size:
                raise ValueError('snapshot time/stack shapes disagree')
            order = np.argsort(times, kind='stable')
            for i in order:
                yield float(times[i]), stack[i]
        else:
            found = []
            pattern = re.compile(r'^t(.+)_' + re.escape(field) + '$')
            for key in data.files:
                match = pattern.fullmatch(key)
                if match:
                    found.append((float(match.group(1)), key))
            if not found:
                raise ValueError(f'no saved temporal maps for {field!r}; use inspect to list actual keys')
            for time, key in sorted(found):
                yield time, data[key]


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def new_file(path):
    p = Path(path)
    if p.exists():
        raise ValueError(f'refusing to overwrite {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def map_figure(a, title, label, dx, vmin, vmax, cmap='viridis'):
    plt = plotting()
    fig, ax = plt.subplots(figsize=(6.6, 5.2), layout='constrained')
    extent = None if dx is None else (0, a.shape[1] * dx, 0, a.shape[0] * dx)
    im = ax.imshow(a, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(xlabel='Column' if dx is None else 'East distance (m)',
           ylabel='Row (south to north)' if dx is None else 'North distance (m)', title=title)
    fig.colorbar(im, ax=ax, label=label)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='list actual NPZ keys/shapes without loading all payloads')
    inspect.add_argument('input', type=Path)
    for mode in ('map', 'frames'):
        s = sub.add_parser(mode)
        s.add_argument('input', type=Path)
        s.add_argument('--key' if mode == 'map' else '--field', default='depth_m')
        s.add_argument('--output' if mode == 'map' else '--frames-dir', type=Path, required=True)
        s.add_argument('--dx-m', type=float)
        s.add_argument('--label', default='Depth (m)')
        s.add_argument('--cmap', default='viridis', help='Matplotlib colormap; use RdBu_r for signed height change')
        s.add_argument('--vmin', type=float)
        s.add_argument('--vmax', type=float)
        group = s.add_mutually_exclusive_group()
        group.add_argument('--class-index', type=int)
        group.add_argument('--sum-classes', action='store_true')
    hydro = sub.add_parser('hydrograph')
    hydro.add_argument('input', type=Path)
    hydro.add_argument('--key', default='outlet_discharge_m3_s')
    hydro.add_argument('--output', type=Path, required=True)
    hydro.add_argument('--label', default='Outlet discharge (m³/s)')
    args = p.parse_args(argv)
    if args.command == 'inspect':
        print(json.dumps(inventory(args.input), indent=2))
        return
    if hasattr(args, 'dx_m') and args.dx_m is not None and (not np.isfinite(args.dx_m) or args.dx_m <= 0):
        raise ValueError('--dx-m must be finite and positive')
    if args.command in ('map', 'frames'):
        plotting()
        import matplotlib
        matplotlib.colormaps[args.cmap]  # validate before creating any output directory
    if args.command == 'hydrograph':
        output = new_file(args.output)
        with np.load(args.input, allow_pickle=False) as d:
            t, y = d['t_s'], d[args.key]
        if t.ndim != 1 or y.shape != t.shape or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
            raise ValueError('hydrograph requires equal finite one-dimensional time/value arrays')
        plt = plotting()
        fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
        ax.plot(t, y)
        ax.set(xlabel='Model time (s)', ylabel=args.label)
        fig.savefig(output, dpi=160)
        plt.close(fig)
    elif args.command == 'map':
        output = new_file(args.output)
        with np.load(args.input, allow_pickle=False) as d:
            a = select_map(d[args.key], args.class_index, args.sum_classes)
        fig = map_figure(a, args.key, args.label, args.dx_m, args.vmin, args.vmax, args.cmap)
        fig.savefig(output, dpi=160)
        plotting().close(fig)
    else:
        target = args.frames_dir
        if target.exists():
            raise ValueError(f'refusing to overwrite frames directory {target}')
        times = [t for t, _ in snapshots(args.input, args.field)]
        if not times or not all(np.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('snapshot times must be finite and strictly increasing')
        # One extra streaming pass establishes one fixed color range for the whole movie.
        lo, hi = np.inf, -np.inf
        for _, array in snapshots(args.input, args.field):
            a = select_map(array, args.class_index, args.sum_classes)
            lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
        lo = lo if args.vmin is None else args.vmin
        hi = hi if args.vmax is None else args.vmax
        if hi == lo:
            hi = lo + max(abs(lo) * 1e-6, 1e-12)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError('need finite --vmin < --vmax')
        target.mkdir(parents=True)
        for i, (t, array) in enumerate(snapshots(args.input, args.field)):
            a = select_map(array, args.class_index, args.sum_classes)
            fig = map_figure(a, f'{args.field}; t = {t:g} s', args.label, args.dx_m, lo, hi, args.cmap)
            fig.savefig(target / f'frame_{i:05d}.png', dpi=120)
            plotting().close(fig)
        (target / 'frame_times_s.json').write_text(json.dumps(times, indent=2) + '\n')
        print(f'{len(times)} saved times rendered; fixed color scale [{lo:g}, {hi:g}].')
        if len(times) > 2 and not np.allclose(np.diff(times), times[1] - times[0], rtol=1e-12, atol=1e-12):
            print('Unequal snapshot spacing: constant-fps video does not show constant physical time speed.')


if __name__ == '__main__':
    main()
