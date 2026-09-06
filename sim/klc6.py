"""K-LC6 link budget, noise floors and detection range.

Datasheet values are transcribed from RFbeam "K-LC6 RADAR TRANSCEIVER
Datasheet" rev 1.2 (02-Nov-2018), pages 2-3 and 6. Nothing in this module is
invented; the one derived constant (IMPL_MARGIN_DB) is reverse-engineered from
RFbeam's own two range claims and is documented where it is defined.

The central question this module answers: given a target RCS and a range, what
IF voltage appears at X1 pin 3, and is it above the floor? There are three
different floors and which one dominates is the whole story:

  1. module IF noise        45 nV/rtHz  -- the physics floor
  2. AD2 quantisation      613 nV/rtHz  -- 22.7 dB WORSE, and it dominates today
  3. indoor clutter + mains    measured -- dominates near 0 Doppler only

FINDINGS 1.1 measured floor 2; FINDINGS 3 measured floor 3. Floor 1 is only
reachable with an IF amplifier, which is why FINDINGS open item 3 calls it the
wall for everything.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- constants
C = 299_792_458.0
F_CARRIER = 24.125e9
LAMBDA = C / F_CARRIER                 # 0.0124266 m
HZ_PER_MPS = 2.0 / LAMBDA              # 160.946 Hz per m/s

# ---- datasheet table, page 2 (typ column unless noted) ----
EIRP_DBM = 18.0                        # +16 min / +18 typ / +20 max
G_ANT_DBI = 12.5                       # note 2: "theoretical value, given by design"
G_LNA_DB = 10.0
MIXER_LOSS_DB = 6.0
P_RX_SENS_DBM = -108.0                 # fIF=500Hz, B=1kHz, S/N=6dB
D_SYSTEM_DBC = -126.0                  # overall sensitivity, same conditions
IF_NOISE_V_RTHZ = 45e-9                # bare K-LC6, = -147 dBV/rtHz
IF_OFFSET_V = 0.2                      # +/- 0.2 V IF output offset
IF_BW_HZ = 50e6                        # bare module -3 dB bandwidth (0..50 MHz)

# K-LC6_V2 differs and the difference matters for drones -- see note in
# blade_band_is_clipped() below.
V2_IF_GAIN_DB = 20.0
V2_IF_NOISE_V_RTHZ = 450e-9
V2_IF_BW_HZ = (10.0, 15e3)             # 10 Hz .. 15 kHz  <-- clips blade Doppler
V2_IF_OFFSET_V = 2.5

# ---- VCO, datasheet page 2 + Fig. 3 ----
VCO_V_MIN, VCO_V_MAX = 1.0, 10.0       # usable input range; open-circuit = 5 V
VCO_TUNE_HZ = 250e6                    # total tuning range over 1..10 V
VCO_SENS_HZ_PER_V = 25e6               # datasheet "VCO sensitivity" 25 MHz/V
VCO_PULLUP_OHM = 10e3

# ---- antenna, datasheet page 3 ----
# The narrow 12 deg lobe lies along the module's LONG axis (8 patch elements).
# Which of azimuth/elevation that becomes is purely a mounting choice, and it
# is the single most consequential mechanical decision in the two-radar demo.
BEAM_NARROW_DEG = 12.0
BEAM_WIDE_DEG = 80.0
SIDELOBE_NARROW_DB = -20.0
SIDELOBE_WIDE_DB = -18.0

# ---- datasheet reference conditions for the sensitivity spec ----
REF_BW_HZ = 1000.0
REF_SNR_DB = 6.0

# Two-way link gain of the module itself, ideal radar equation:
#     Pr/Pt = Gt*Gr*lambda^2*sigma / ((4pi)^3 R^4)
G_ANT = 10 ** (G_ANT_DBI / 10.0)
K_LINK = G_ANT * G_ANT * LAMBDA ** 2 / (4 * np.pi) ** 3      # 2.4607e-5

# RFbeam's practical range formula (datasheet page 6) is
#     r = 0.0167 * 10^(-S/40) * sigma^(1/4),   S = -126 dBc
# which gives 23.6 m for a 1 m^2 person and 62.7 m for a 50 m^2 car -- both
# claims on that page. Running the ideal radar equation above with the same
# -126 dBc sensitivity instead gives 99.5 m for the person. The gap is exactly
# 25.0 dB of two-way power, and it is RFbeam's own real-world derating:
# implementation loss, RCS fluctuation, imperfect aspect, radome, processing.
# We keep it, because a range prediction that ignores it would be optimistic by
# a factor of 4.2 in range -- and this repo has already been burned once by
# trusting a datasheet number (the 300 MHz sweep, FINDINGS 5.3).
IMPL_MARGIN_DB = 25.0

# IF volts corresponding to a return at exactly the -126 dBc sensitivity:
# by definition that is 6 dB above the module's own noise in a 1 kHz bandwidth.
V_NOISE_REF = IF_NOISE_V_RTHZ * np.sqrt(REF_BW_HZ)           # 1.423 uV rms
V_SIG_REF = V_NOISE_REF * 10 ** (REF_SNR_DB / 20.0)          # 2.839 uV rms

# ---- Analog Discovery 2 front end (FINDINGS 1.1) ----
AD2_LSB_V = 336e-6                     # measured; 5 Vpp range / 14 bit
AD2_QUANT_RMS_V = AD2_LSB_V / np.sqrt(12.0)                  # 97.0 uV rms
AD2_FULLSCALE_V = 2.5                  # +/-2.5 V on the 5 Vpp range

# ---- MEASURED noise floor of the real chain ------------------------------
# Median Welch PSD (spur-immune) of out/baseline/20260829_061237_empty_baseline
# _60s.npz, I channel, 59.8 s at 50 kSa/s, 0.76 Hz bins. This is the number the
# whole drone budget hangs on, and it is measured on this bench, not derived.
#
#   quantisation alone predicts    613 nV/rtHz
#   measured at 7-14 kHz           ~845 nV/rtHz   (the extra 2.8 dB is the
#                                                  AD2's own analog front end)
#   module's own IF noise            45 nV/rtHz
#
# So the AD2 -- not the K-LC6 -- sets sensitivity, by 25 dB. Two consequences
# that shape the entire demo:
#   * the floor is FLAT above ~6 kHz. Fast blade tips land in clean spectrum.
#   * below 500 Hz it rises 4.5 dB into 1/f, mains skirts and clutter, which is
#     where the HERM comb (201 Hz at hover) unfortunately lives.
_FLOOR_HZ = np.array([0.0, 250.0, 1000.0, 3000.0, 5000.0, 7500.0,
                      11500.0, 17000.0, 25000.0])
_FLOOR_V_RTHZ = np.array([1494e-9, 1391e-9, 1015e-9, 928e-9, 883e-9, 845e-9,
                          834e-9, 836e-9, 844e-9])
MEASURED_FLOOR_REF = "out/baseline/20260829_061237_empty_baseline_60s.npz"


# ------------------------------------------------------------------ budget
def link_dbc(sigma_m2, range_m, impl_margin_db=IMPL_MARGIN_DB):
    """Two-way return relative to the transmitted carrier, in dBc.

    Includes RFbeam's 25 dB practical derating so that the result reproduces
    the detection ranges printed on datasheet page 6.
    """
    sigma_m2 = np.asarray(sigma_m2, dtype=float)
    range_m = np.asarray(range_m, dtype=float)
    ideal = 10 * np.log10(K_LINK * sigma_m2 / np.maximum(range_m, 1e-6) ** 4)
    return ideal - impl_margin_db


def if_volts(sigma_m2, range_m, if_gain_db=0.0, **kw):
    """RMS IF voltage at the module output (or after an external IF amp)."""
    return (V_SIG_REF * 10 ** ((link_dbc(sigma_m2, range_m, **kw) - D_SYSTEM_DBC) / 20.0)
            * 10 ** (if_gain_db / 20.0))


def measured_floor_v_rthz(freq_hz):
    """Input-referred noise density of the AD2 chain at a given IF frequency.

    Interpolated from the measured empty-room baseline. Use this rather than
    the quantisation formula: it is 2.8 dB higher and it is what the hardware
    actually does.
    """
    return np.interp(np.abs(np.asarray(freq_hz, dtype=float)),
                     _FLOOR_HZ, _FLOOR_V_RTHZ)


def noise_density_v_rthz(freq_hz, if_gain_db=0.0, module_noise=IF_NOISE_V_RTHZ,
                         module_noise_only=False):
    """Input-referred noise density of the whole chain, V/rtHz.

    Module noise sits in front of the amplifier and is amplified with the
    signal; everything the AD2 contributes -- quantisation plus its own analog
    front end -- sits after it and is not. So gain ahead of the ADC divides the
    AD2's contribution and leaves the module's alone. That asymmetry is the
    entire argument for the IF amp, and it is why 30 dB buys back 24 of the
    25 dB currently being lost.

    `module_noise_only` drops the AD2 entirely, which is the receiver RFbeam
    assumed when they printed their detection ranges. Used to check this model
    against the datasheet, not to predict our bench.
    """
    if module_noise_only:
        return np.full_like(np.asarray(freq_hz, dtype=float), module_noise)
    g = 10 ** (if_gain_db / 20.0)
    return np.sqrt(module_noise ** 2 + (measured_floor_v_rthz(freq_hz) / g) ** 2)


def sensitivity_penalty_db(freq_hz, if_gain_db=0.0, **kw):
    """dB of sensitivity the AD2 front end is costing, vs the module's own floor."""
    return 20 * np.log10(noise_density_v_rthz(freq_hz, if_gain_db, **kw)
                         / IF_NOISE_V_RTHZ)


def snr_db(sigma_m2, range_m, bin_hz, freq_hz, if_gain_db=0.0,
           n_spread_bins=1, n_incoherent=1, module_noise_only=False, **kw):
    """SNR of the detection statistic.

    Three detector shapes, selected by `n_spread_bins`:

    n_spread_bins == 1  -- COHERENT LINE. All the target power lands in one
        FFT bin (the body line, a HERM comb tooth). Noise in that bin falls
        with bin width, so narrow bins are pure gain. FINDINGS 1.1: everything
        ever detected on this bench got there on FFT processing gain alone.

    n_spread_bins > 1   -- BAND ENERGY. The power is smeared over N bins (the
        blade pedestal, which runs from 0 out to the tip Doppler). Summing the
        band recovers all the signal while the noise sum fluctuates by only
        1/sqrt(N), so the detector gains sqrt(N) over the per-bin SNR. This is
        the detector the fan study used to bound a blade line at 8.1 uV rms.

    `n_incoherent` frames averaged afterwards add a further 10*log10(sqrt(N)).
    """
    v_sig = if_volts(sigma_m2, range_m, if_gain_db=if_gain_db, **kw)
    nd = (noise_density_v_rthz(freq_hz, if_gain_db,
                               module_noise_only=module_noise_only)
          * 10 ** (if_gain_db / 20.0))
    n_spread = max(int(n_spread_bins), 1)
    # Noise in the whole band the signal occupies, then the sqrt(N) detector gain.
    v_noise = nd * np.sqrt(bin_hz * n_spread) / (n_spread ** 0.25)
    gain_db = 10 * np.log10(np.sqrt(max(n_incoherent, 1)))
    return 20 * np.log10(v_sig / v_noise) + gain_db


def detection_range_m(sigma_m2, bin_hz, freq_hz, snr_req_db=REF_SNR_DB, **kw):
    """Range at which `sigma_m2` reaches `snr_req_db`.

    SNR falls as R^-4, so solve in closed form off the 1 m value rather than
    searching.
    """
    snr_1m = snr_db(sigma_m2, 1.0, bin_hz, freq_hz, **kw)
    return 10 ** ((snr_1m - snr_req_db) / 40.0)


def datasheet_range_m(sigma_m2, sensitivity_dbc=D_SYSTEM_DBC):
    """RFbeam's own page-6 formula, for cross-checking: r = 0.0167*10^(-S/40)*sigma^0.25."""
    return 0.0167 * 10 ** (-sensitivity_dbc / 40.0) * np.asarray(sigma_m2, float) ** 0.25


# ------------------------------------------------------------------ FMCW
def sweep_bandwidth_hz(v_low, v_high, sens_hz_per_v=VCO_SENS_HZ_PER_V):
    """Sweep bandwidth from the VCO drive span actually applied.

    The AD2's W1 tops out around +/-5 V, so a single-ended 0.5..4.5 V drive
    reaches only ~4 V of the VCO's 1..10 V input range -- about 100 MHz of the
    available 250 MHz. Range resolution is c/2B, so this is a direct, and
    entirely recoverable, factor-of-2.5 penalty on positioning accuracy.
    """
    span = max(0.0, min(v_high, VCO_V_MAX) - max(v_low, VCO_V_MIN))
    return min(span * sens_hz_per_v, VCO_TUNE_HZ)


def range_resolution_m(bw_hz):
    return C / (2.0 * np.asarray(bw_hz, dtype=float))


def range_accuracy_m(bw_hz, snr_db_val):
    """1-sigma range accuracy of a peak centroid at a given SNR.

    sigma_R = dR / sqrt(2*SNR) -- the standard matched-filter bound. This is
    the number that feeds position accuracy, and it is much better than the
    range *resolution* whenever SNR is decent. It ignores VCO nonlinearity,
    which FINDINGS 7 lists as uncorrected and which currently adds a bias on
    top of this.
    """
    snr_lin = 10 ** (np.asarray(snr_db_val, dtype=float) / 10.0)
    return range_resolution_m(bw_hz) / np.sqrt(2.0 * snr_lin)


def blade_band_is_clipped(tip_doppler_hz, variant="K-LC6"):
    """True if the module variant's IF bandwidth cuts the blade line off.

    The bare K-LC6 runs to 50 MHz and clips nothing. The K-LC6_V2 -- the one
    with the 20 dB IF amp already built in, which is otherwise exactly what
    this project needs -- rolls off at 15 kHz. A DJI Mini 3 at full throttle
    puts blade tips at 13.8 kHz, inside that but on the shoulder; anything
    faster is attenuated. Worth knowing before ordering the V2 as the fix.
    """
    if variant.upper().endswith("V2"):
        return float(tip_doppler_hz) > V2_IF_BW_HZ[1]
    return float(tip_doppler_hz) > IF_BW_HZ


def beam_gain_db(off_axis_deg, beamwidth_deg):
    """Two-way gain relative to boresight, Gaussian main-lobe approximation.

    -6 dB of IF voltage at the -3 dB power points, per the datasheet's note on
    Fig. 2. Sidelobes are not modelled -- inside the main lobe this is good to
    about a dB, and outside it you should not be trying to measure anyway.
    """
    x = np.asarray(off_axis_deg, dtype=float) / (beamwidth_deg / 2.0)
    return -6.0 * x ** 2                                   # -6 dB two-way at the edge
