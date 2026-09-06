"""Servo vs elevation DBF for the production fan, 8 roof nodes, buildings on.

  python viz/dkr/dbf_elev.py
"""
import sys, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("ops", Path(__file__).resolve().parents[2] / "scripts" / "coverage_ops.py")
ops = importlib.util.module_from_spec(spec); spec.loader.exec_module(ops)
import numpy as np, math
depth = 400.0
X, Y = ops.zone_grid(depth, cell=10.0)
alts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
runs = ops.approaches(600, 5, 60, start_out=650.0)
def nodes_for(n, h, el, tilt, cone=True):
    out = []
    for x, y, nx, ny in ops.wall_points(n):
        tilts = ops.sector_tilts(int(tilt[7:]), h, depth, el) if isinstance(tilt, str) else [tilt]
        for t in tilts:
            out.append(ops.Node(x, y, h, nx, ny, t, 90.0, el, 595.0))
        if cone:
            out.append(ops.Node(x, y, h, nx, ny, 90.0, 120.0, 120.0, 131.0))
    return out
cfgs = [
    ("BOM as drawn: 15 deg fixed +10",                         15.0, 10.0),
    ("servo, 15 deg x 3 sectors (5 Hz each)",                  15.0, "sectors3"),
    ("servo, 15 deg x 4 sectors (4 Hz each)",                  15.0, "sectors4"),
    ("DBF 30 deg (32 ch), fixed, centred on the drone band",   30.0, None),
    ("DBF 45 deg (48 ch), fixed",                              45.0, None),
    ("DBF 60 deg (64 ch), fixed",                              60.0, None),
    ("DBF 30 deg + servo 2 positions",                         30.0, "sectors2"),
]
print(f"  {'config':<58}{'vol%':>6}{'street%':>8}{'CAUGHT':>8}{'warn m':>8}")
for label, el, tilt in cfgs:
    h = 45.0
    if tilt is None:   # centre the wide beam on what the zone needs: street at edge .. 60 m drone at 40 m
        lo = math.degrees(math.atan2(1.5 - h, depth)); hi = math.degrees(math.atan2(60 - h, 40.0))
        tilt = 0.5 * (lo + hi)
        if el < (hi - lo):   # can't cover both ends: bias toward the drone band
            tilt = math.degrees(math.atan2(5 - h, depth)) + el / 2 - 2
    nd = nodes_for(8, h, el, tilt)
    vol, _ = ops.volume_coverage(nd, X, Y, alts)
    st = ops.street_coverage(nd, X, Y)
    caught, warn, _ = ops.approach_stats(nd, runs)
    extra = f"  (tilt {tilt:+.0f} deg)" if not isinstance(tilt, str) else ""
    print(f"  {label:<58}{100*vol:>6.0f}{100*st:>8.0f}{100*caught:>7.0f}%{warn:>8.0f}{extra}")
