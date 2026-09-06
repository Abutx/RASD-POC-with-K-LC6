"""Walking human micro-Doppler -- the confuser the drone has to be told apart from.

Same modelling contract as sim/drone.py: scatterers with real kinematics, phase
accumulated from radial motion, no signature painted in by hand. If the human
model is generous the classifier looks better than it is, so the parameters
here are chosen to make the human as drone-like as the physics allows: brisk
walk, arms swinging, and the fastest limb speeds a person actually reaches.

The separation this produces is not subtle, and that is the point. A person's
entire micro-Doppler lives below ~700 Hz because no part of a human body moves
faster than about 4 m/s. A Mini 3's blade tips run 48 m/s at hover. There is an
order of magnitude of empty spectrum between them, and that gap -- not the
cadence, not the RCS -- is the discriminant that does the real work.

Kinematics follow the usual Boulic-Thalmann style decomposition (torso plus
independently-phased limb segments) reduced to the radial component, which is
all a single-baseline Doppler radar can observe anyway.
"""
from __future__ import annotations

import numpy as np

from . import klc6

# RFbeam quote 1 m^2 for a moving person (datasheet p.6), which is the figure
# their own detection-range claim is built on, so we keep it and split it over
# the body the way the micro-Doppler literature does.
SIGMA_TOTAL_M2 = 1.0
GAIT_HZ = 1.8                      # step rate for a brisk walk, ~2 steps/s
WALK_MPS = 1.4

# fraction of total RCS, radial-velocity swing (m/s), phase (cycles)
# Feet get the biggest swing and the asymmetric stance/swing profile below;
# everything else is close enough to sinusoidal at this resolution.
SEGMENTS = [
    #  name      rcs_frac  v_swing  phase   profile
    ("torso",      0.50,     0.10,   0.00,  "sine2"),   # bob at 2x gait
    ("head",       0.08,     0.06,   0.00,  "sine2"),
    ("left_leg",   0.14,     1.30,   0.00,  "sine"),
    ("right_leg",  0.14,     1.30,   0.50,  "sine"),
    ("left_foot",  0.05,     1.70,   0.00,  "gait"),    # stance then swing
    ("right_foot", 0.05,     1.70,   0.50,  "gait"),
    ("left_arm",   0.02,     0.90,   0.50,  "sine"),    # counter-swings legs
    ("right_arm",  0.02,     0.90,   0.00,  "sine"),
]


def _profile(t, kind, f_gait, phase):
    """Radial-velocity shape for one body segment, unit amplitude."""
    ph = 2.0 * np.pi * (f_gait * t + phase)
    if kind == "sine":
        return np.sin(ph)
    if kind == "sine2":
        return np.sin(2.0 * ph)
    if kind == "gait":
        # A foot is STATIONARY during stance (~60% of the cycle) and then
        # swings through at up to twice the walking speed. That asymmetry is
        # what makes the classic human spectrogram flare, and a pure sinusoid
        # would not reproduce it.
        u = np.mod(f_gait * t + phase, 1.0)
        swing = u > 0.6
        s = np.zeros_like(t)
        s[swing] = np.sin(np.pi * (u[swing] - 0.6) / 0.4)
        return s
    raise ValueError(kind)


def micro_doppler(t, walk_mps=WALK_MPS, f_gait=GAIT_HZ, aspect_cos=1.0,
                  segments=None, seed=0):
    """Unit-power complex return from a walking person.

    `aspect_cos` projects the walk onto the line of sight: 1.0 is walking
    straight at the radar, 0.0 is walking across it (bulk Doppler zero, limbs
    still visible).
    """
    rng = np.random.default_rng(seed)
    segments = segments or SEGMENTS
    dt = t[1] - t[0]
    s = np.zeros(t.shape, dtype=complex)
    for name, frac, v_swing, phase, kind in segments:
        v = aspect_cos * (walk_mps + v_swing * _profile(t, kind, f_gait, phase))
        disp = np.cumsum(v) * dt
        amp = np.sqrt(frac)
        s += amp * np.exp(-1j * 4.0 * np.pi / klc6.LAMBDA * disp
                          + 1j * rng.uniform(0, 2 * np.pi))
    return s / np.sqrt(np.mean(np.abs(s) ** 2) + 1e-30)


def signature(t, range_m, sigma_m2=SIGMA_TOTAL_M2, **kw):
    """Complex IF signal in VOLTS at the module pin, for a person at range_m."""
    return klc6.if_volts(sigma_m2, range_m) * micro_doppler(t, **kw)


def max_limb_doppler_hz(walk_mps=WALK_MPS):
    """Fastest Doppler any part of a walking person produces.

    A swinging foot peaks near twice the walking speed. Even a sprinter's foot
    tops out around 8-10 m/s, i.e. 1.3-1.6 kHz -- still five times below a
    Mini 3's hover blade tips.
    """
    v_max = max(walk_mps + s[2] for s in SEGMENTS)
    return v_max * klc6.HZ_PER_MPS
