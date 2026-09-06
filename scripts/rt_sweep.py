#!/usr/bin/env python3
"""Run the PO model across elevations and check angular convergence.

  python scripts/rt_sweep.py

Elevation is the variable that decides whether the blade faces are seen near
specular (healthy PO regime) or at grazing (PO's weakest regime), so the model
is run at 0, 10, 30 and 60 deg. The convergence check re-traces one case at
double the angular resolution: if the energy above the tip Doppler moves, the
lookup is under-resolved.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RUN = os.path.join(HERE, "rt_drone.py")

# Sign convention (sim/rt/trace.py radar_position): el > 0 puts the radar ABOVE
# the rotor plane, el < 0 below. The analytic model's cos(el) null is symmetric;
# the PO model is not -- from below the body shadows the blades and the floor
# bounce is steeper -- so both signs are run.
cases = [
    (["--el", "0", "--yaw-sweep"], "level view -- blade faces at grazing, PO weakest"),
    (["--el", "10"], "radar 10 deg ABOVE the rotor plane"),
    (["--el", "-10"], "radar 10 deg BELOW -- body shadows the far blades"),
    (["--el", "30"], "radar 30 deg above"),
    (["--el", "60"], "radar 60 deg above -- toward the cos(el) null"),
    (["--el", "10", "--n-phi", "2880", "--no-figs", "--seconds", "2"],
     "CONVERGENCE: el 10 at 2x angular resolution"),
    (["--el", "0", "--no-floor", "--no-figs", "--seconds", "2"],
     "level view, no floor -- isolates the floor bounce"),
]
for args, why in cases:
    print("\n" + "#" * 78)
    print(f"# {why}")
    print(f"# rt_drone.py {' '.join(args)}")
    print("#" * 78, flush=True)
    r = subprocess.run([PY, RUN, *args], cwd=os.path.dirname(HERE))
    if r.returncode != 0:
        print(f"!! exited {r.returncode}")
