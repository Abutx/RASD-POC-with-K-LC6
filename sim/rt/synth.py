"""Lookup tables + RPM profile -> complex IF time series, in volts.

Reuses sim/drone.py for everything kinematic (per-rotor RPM wander, hover
jitter of the airframe) so the only thing that changes between the analytic
model and the PO model is the scattering. That is deliberate: it makes the two
directly comparable, and it means every non-obvious feature in the PO
spectrogram is a scattering effect, not a kinematic one.

Amplitude chain: the PO tables are complex RCS amplitudes at the reference
range, so |A|^2 is an effective RCS in m^2. That goes through the same
klc6.if_volts() link budget (RFbeam derating included) as the analytic model.
"""
from __future__ import annotations

import numpy as np

from .. import drone, klc6


def _interp_complex(phi_tab, tab, phi):
    """Periodic (period pi) linear interpolation of a complex lookup."""
    n = len(phi_tab)
    x = np.mod(phi, np.pi) / np.pi * n
    i0 = np.floor(x).astype(int) % n
    i1 = (i0 + 1) % n
    t = x - np.floor(x)
    return tab[i0] * (1 - t) + tab[i1] * t


def rotor_angles(t, rpm_mean, rpm_spread_frac=0.02, seed=0, **kw):
    """Four independent rotor angle tracks, radians, from sim.drone's RPM model."""
    rng = np.random.default_rng(seed)
    means = rpm_mean * (1.0 + rpm_spread_frac * np.array([0.5, -0.3, 0.2, -0.4]))
    dt = t[1] - t[0]
    out = np.zeros((4, len(t)))
    for i in range(4):
        rpm_i = drone.rpm_track(t, means[i], rng=rng, **kw)
        omega = 2 * np.pi * rpm_i / 60.0
        out[i] = np.cumsum(omega) * dt + rng.uniform(0, np.pi)
    return out


def signature(t, trace, range_m, rpm=drone.RPM_HOVER, seed=0, v_radial=None,
              include_static=True, **kw):
    """Complex IF voltage at the module pin for the traced pose.

    trace: dict from trace.DroneTracer.trace(). The static (body/arm/motor)
    return and each rotor's lookup are summed coherently, then the whole
    airframe's bulk motion (hover jitter or v_radial) is applied as a common
    phase, then the total is scaled by the link budget.
    """
    phi_tab = trace["phi"]
    angles = rotor_angles(t, rpm, seed=seed, **kw)
    A = np.zeros(len(t), dtype=complex)
    for i in range(4):
        A += _interp_complex(phi_tab, trace["rotor"][i], angles[i])
    if include_static:
        s = trace["static"]
        A += sum(complex(v[0], v[1]) for v in s.values())
    # bulk motion of the airframe: sim.drone.body_signal is unit-power phase
    A *= drone.body_signal(t, seed=seed, v_radial=v_radial)
    # |A|^2 is RCS in m^2 -> volts via the same budget the analytic model uses
    sigma = np.abs(A) ** 2 + 1e-30
    v = klc6.if_volts(sigma, range_m)
    return v * A / np.sqrt(sigma)


def effective_rcs(trace):
    """Time-averaged RCS split: static, rotors, and the coherent total."""
    s = sum(complex(v[0], v[1]) for v in trace["static"].values())
    rot = trace["rotor"]
    rot_pow = np.mean(np.abs(rot) ** 2, axis=1)         # per rotor, mean over angle
    return {
        "static_dbsm": 10 * np.log10(abs(s) ** 2 + 1e-30),
        "rotor_dbsm": (10 * np.log10(rot_pow + 1e-30)).tolist(),
        "rotors_total_dbsm": 10 * np.log10(rot_pow.sum() + 1e-30),
        "blade_below_static_db": 10 * np.log10(abs(s) ** 2 / max(rot_pow.sum(), 1e-30)),
    }
