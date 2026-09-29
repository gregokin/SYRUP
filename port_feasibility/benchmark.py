"""Bounded plain-Python routing timing; no JIT or GPU performance claim."""
import importlib.util
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter
import numpy as np
from kernels import Network, water_step

rows=[]
for side in (16,32):
    n=side*side
    receiver=np.arange(n)+1
    receiver[np.arange(side-1,n,side)]+=side-1
    receiver[-1]=-1
    net=Network(receiver,np.arange(n),.5)
    slope=np.full(n,.01);rain=np.full(n,1e-5);h=np.full(n,.002)
    q=np.sqrt(8*9.81*slope)*h**1.5
    samples=[]
    for repetition in range(3):
        start=perf_counter()
        r=water_step(net,h,q,rain,slope,np.ones(n),.1)
        samples.append(perf_counter()-start)
        assert np.max(np.abs(r.balance_by_cell_m3)) < 1e-16
    rows.append(dict(nx=side,ny=side,cells=n,dt_s=.1,seconds_per_step=samples,
                     median_seconds_per_step=median(samples),
                     max_abs_cell_water_balance_m3=float(np.max(np.abs(r.balance_by_cell_m3)))))
report=dict(kind='plain Python CPU, isolated water kernel only',python=platform.python_version(),
            numpy=np.__version__,numba_available=importlib.util.find_spec('numba') is not None,
            rows=rows,limitations='No compiled comparison; no infiltration, sediment, moving terrain, MAPLE output, or full simulation timing. Bisection bracket/tolerance differ from legacy.')
path=Path(__file__).with_name('benchmark_results.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
