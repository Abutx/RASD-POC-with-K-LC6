"""DJI Mini 3 micro-Doppler signal model.

Not a spectrogram generator with plausible-looking smears painted on. Every
blade is a line of scatterers whose slant range to the radar is computed each
sample and turned into phase; the flashes, the HERM comb and the Doppler
pedestal all fall out of that sum rather than being drawn in. If the geometry
is wrong the signature disappears, which is the point -- FINDINGS trap 7 (a fan
pointed AT the radar is Doppler-invisible) has to reproduce here or the model
is not telling us anything we can trust.

Airframe reference: DJI Mini 3, 248 g, 251x362x72 mm unfolded, 6030F props
(6 in diameter, 3 in pitch, 2 blades), 1504C motors -- 6040 rpm hovering,
10700 rpm maximum. Sources in docs/DEMO_TWO_RADAR.md.

  hover   6040 rpm -> 100.7 rev/s -> tip 48.2 m/s -> 7.76 kHz Doppler
  max    10700 rpm -> 178.3 rev/s -> tip 85.4 m/s -> 13.75 kHz Doppler
  HERM line spacing = blades x rev/s = 201 Hz at hover
"""
from __future__ import annotations

import numpy as np

from . import klc6

# ---- DJI Mini 3 geometry ----
PROP_DIAMETER_M = 6 * 0.0254           # 6030F -> 6 inch
PROP_RADIUS_M = PROP_DIAMETER_M / 2    # 0.0762 m
N_BLADES = 2
N_ROTORS = 4
ROTOR_ARM_M = 0.247 / 2                # 247 mm motor-to-motor diagonal
MASS_KG = 0.248

RPM_HOVER = 6040.0
RPM_MAX = 10700.0

# Inner quarter of the blade is root and hub: slow, shadowed, and structurally
# not a good scatterer. Standard thin-blade rotor model integrates the outer
# fraction only.
BLADE_ROOT_FRAC = 0.25

# ---- RCS, and this is the weakest number in the whole analysis ----
# Nearest measured neighbours at our band (see docs/DEMO_TWO_RADAR.md refs):
#   DJI Phantom 4 Pro   -12.4 dBsm @ 25 GHz   (1375 g, 350 mm diagonal)
#   DJI Inspire 1 Pro   -11.1 dBsm @ 25 GHz
#   DJI Phantom 3       -13 to -14 dBsm @ 24 GHz
# The Mini 3 is 0.71x the diagonal and 0.18x the mass of a Phantom 4, so a
# projected-area scaling puts it 3-5 dB below: call it -17 dBsm median, and
# carry the aspect spread explicitly because it is large.
SIGMA_BODY_DBSM = -17.0                # ESTIMATE, extrapolated -- +/- 5 dB
SIGMA_BODY_SPREAD_DB = (-25.0, -12.0)  # aspect-dependent range

# Rahman & Robertson (K-band, IET RSN 2019) measure propeller returns
# 20-40 dB below the body, ~30 dB for a Phantom 3. Mini 3 props are small
# nylon/glass-fibre -- poor reflectors at lambda = 12.4 mm -- so 30 dB is the
# optimistic end of plausible for it, not the pessimistic one.
BLADE_BELOW_BODY_DB = 30.0             # ESTIMATE, literature -- 20 to 40 dB


def rpm_to_tip_mps(rpm, radius_m=PROP_RADIUS_M):
    return 2.0 * np.pi * radius_m * (np.asarray(rpm, float) / 60.0)


def rpm_to_tip_hz(rpm, radius_m=PROP_RADIUS_M):
    return rpm_to_tip_mps(rpm, radius_m) * klc6.HZ_PER_MPS


def herm_spacing_hz(rpm, n_blades=N_BLADES):
    """Blade-pass rate = HERM line spacing = the cadence peak we look for."""
    return n_blades * np.asarray(rpm, float) / 60.0


def sigma_body_m2(dbsm=SIGMA_BODY_DBSM):
    return 10 ** (dbsm / 10.0)


def sigma_blades_m2(dbsm=SIGMA_BODY_DBSM, below_db=BLADE_BELOW_BODY_DB):
    """Total micro-Doppler RCS of all four rotors combined."""
    return 10 ** ((dbsm - below_db) / 10.0)


# ------------------------------------------------------------------ signal
def rotor_positions(arm_m=ROTOR_ARM_M):
    """Four motors on an X frame, in the drone's body frame (z up)."""
    d = arm_m / np.sqrt(2.0)
    return np.array([[+d, +d, 0.0], [-d, +d, 0.0],
                     [-d, -d, 0.0], [+d, -d, 0.0]])


def rotor_spins():
    """Adjacent rotors counter-rotate: +1 CCW, -1 CW."""
    return np.array([+1.0, -1.0, +1.0, -1.0])


def los_unit(az_deg, el_deg):
    """Unit vector to the radar in the drone frame.

    el_deg is the elevation of the radar as seen from the drone: 0 means the
    radar is level with the rotor plane (full micro-Doppler), +/-90 means it
    is directly above/below (blade motion tangential, micro-Doppler nulled).
    Positive is above; this line-scatterer model is symmetric in the sign, so
    a ground radar under an overhead drone is the same case as el = +90.
    This one angle dominates everything -- see FINDINGS trap 7.
    """
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def rotor_signal(t, rpm, spin, phase0, u, n_seg=48, root_frac=BLADE_ROOT_FRAC,
                 radius_m=PROP_RADIUS_M, n_blades=N_BLADES):
    """Complex baseband return from one rotor, unit-normalised in power.

    Each blade is discretised into n_seg equal-length segments of equal
    reflectivity. The return is the coherent sum

        s(t) = SUM_k exp(-j*4*pi/lambda * (u . p_k(t)))

    over every segment of every blade, where p_k(t) is the segment's position
    in the rotor plane. The 4*pi (not 2*pi) is the two-way path.

    The specular flash comes out of this for free: when a blade is broadside to
    the line of sight every segment sits at the same range, the sum adds in
    phase, and the return spikes -- and at that instant the blade's radial
    velocity is at its maximum, which is why flashes land at the tip Doppler.
    """
    r = radius_m * np.linspace(root_frac, 1.0, n_seg)
    omega = spin * 2.0 * np.pi * np.asarray(rpm, float) / 60.0
    # Angle of blade 0 over time; remaining blades are evenly spaced.
    if np.ndim(omega) == 0:
        phi0 = omega * t + phase0
    else:
        phi0 = np.cumsum(omega) * (t[1] - t[0]) + phase0    # RPM varying in time

    s = np.zeros(t.shape, dtype=complex)
    ux, uy = u[0], u[1]                    # rotor plane is body x-y
    for b in range(n_blades):
        phi = phi0 + 2.0 * np.pi * b / n_blades
        # u . p for a segment at radius r and blade angle phi
        proj = np.outer(np.cos(phi), r) * ux + np.outer(np.sin(phi), r) * uy
        s += np.exp(-1j * 4.0 * np.pi / klc6.LAMBDA * proj).sum(axis=1)
    return s / (n_seg * n_blades)          # unit amplitude at perfect coherence


def rpm_track(t, rpm_mean, jitter_frac=0.012, wander_hz=3.0, rng=None):
    """Per-rotor RPM over time.

    A quad holds attitude by running its four rotors at slightly different and
    constantly-adjusted speeds. That is why real drone spectrograms have flash
    spacing that is not quite constant (Rahman & Robertson) and why the HERM
    comb smears if you integrate too long -- which is exactly what limits
    coherent integration time in the link budget.
    """
    rng = rng or np.random.default_rng()
    n = len(t)
    dt = t[1] - t[0]
    # Low-pass filtered white noise -> smooth wander at ~wander_hz
    w = rng.standard_normal(n)
    alpha = np.exp(-2.0 * np.pi * wander_hz * dt)
    for i in range(1, n):
        w[i] = alpha * w[i - 1] + (1 - alpha) * w[i]
    w /= (np.std(w) + 1e-12)
    return rpm_mean * (1.0 + jitter_frac * w)


def micro_doppler(t, rpm=RPM_HOVER, az_deg=0.0, el_deg=0.0, seed=0,
                  rpm_spread_frac=0.02, **kw):
    """Coherent sum over all four rotors -> unit-power complex micro-Doppler.

    Rotors are given slightly different mean RPM (rpm_spread_frac) plus
    independent wander, which is what produces the dense multi-comb structure
    seen on real quads rather than a single clean HERM ladder.
    """
    rng = np.random.default_rng(seed)
    u = los_unit(az_deg, el_deg)
    offsets = rotor_positions()
    spins = rotor_spins()
    means = rpm * (1.0 + rpm_spread_frac * np.array([0.5, -0.3, 0.2, -0.4]))

    s = np.zeros(t.shape, dtype=complex)
    for i in range(N_ROTORS):
        rpm_i = rpm_track(t, means[i], rng=rng, **kw)
        # Constant phase from the rotor's offset on the airframe: the four
        # rotors are at different ranges, so their returns add with fixed
        # relative phase rather than all in step.
        hub_phase = np.exp(-1j * 4.0 * np.pi / klc6.LAMBDA * float(u @ offsets[i]))
        s += hub_phase * rotor_signal(t, rpm_i, spins[i],
                                      rng.uniform(0, 2 * np.pi), u)
    return s / np.sqrt(np.mean(np.abs(s) ** 2) + 1e-30)


def body_signal(t, hover_jitter_m=0.05, wander_hz=0.8, seed=0, v_radial=None):
    """Bulk-body return: unit power, phase from the body's radial motion.

    Hovering is not stationary. The Mini 3 holds +/-0.1 m vertically and
    +/-0.3 m horizontally on vision positioning, and at lambda = 12.4 mm a
    5 cm wander is 8 whole cycles of phase -- so the body line is a narrow but
    real smear around zero Doppler, not a delta. That smear is what has to
    survive (or not) the MTI notch.
    """
    rng = np.random.default_rng(seed + 991)
    dt = t[1] - t[0]
    if v_radial is None:
        w = rng.standard_normal(len(t))
        alpha = np.exp(-2.0 * np.pi * wander_hz * dt)
        for i in range(1, len(t)):
            w[i] = alpha * w[i - 1] + (1 - alpha) * w[i]
        disp = hover_jitter_m * w / (np.std(w) + 1e-12)
    else:
        disp = np.cumsum(np.broadcast_to(np.asarray(v_radial, float), t.shape)) * dt
    return np.exp(-1j * 4.0 * np.pi / klc6.LAMBDA * disp)


def signature(t, range_m, rpm=RPM_HOVER, az_deg=0.0, el_deg=0.0,
              sigma_body_dbsm=SIGMA_BODY_DBSM,
              blade_below_db=BLADE_BELOW_BODY_DB,
              seed=0, v_radial=None, **kw):
    """Complete complex IF signal in VOLTS for a Mini 3 at `range_m`.

    This is the voltage AT THE MODULE PIN, before any external amplifier --
    sim.capture.render() owns the whole front end from here on, including gain.
    Do not apply IF gain here as well or it lands twice.

    Body and blades are scaled independently through the datasheet link budget,
    so their ratio at the IF is the physical one and the spectrogram shows the
    body line and the blade pedestal at their true relative levels.
    """
    v_body = klc6.if_volts(sigma_body_m2(sigma_body_dbsm), range_m)
    v_blade = klc6.if_volts(sigma_blades_m2(sigma_body_dbsm, blade_below_db),
                            range_m)
    md = micro_doppler(t, rpm=rpm, az_deg=az_deg, el_deg=el_deg, seed=seed, **kw)
    bd = body_signal(t, seed=seed, v_radial=v_radial)
    # if_volts returns rms; a unit-power complex signal already has rms 1.
    return v_body * bd + v_blade * md
