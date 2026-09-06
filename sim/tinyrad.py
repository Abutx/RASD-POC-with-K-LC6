"""EV-TINYRAD24G (Analog Devices) -- link budget, angle, waveform limits.

The second-phase front end from BOM_tinyrad_bench.md. Same modelling contract
as sim/klc6.py: chip datasheets first, this repo's measurements second,
estimates flagged. Where the TinyRad user guide (UG-1709) was not obtainable,
the value is marked ESTIMATE with the reasoning that produced it.

Sources, transcribed:
  UG-1709 rev 0 (2/2020)   -- EV-TINYRAD24G user guide: antenna gain 12.6 dBi,
                              76.5 x 17.6 deg single element, 75 x 15 deg
                              array, -20 dB sidelobes, antenna positions
                              (Table 2), 7-element lambda/2 virtual array with
                              one overlap, fs fixed at 1 MHz, N x 1 us + 20 us
                              < PWM period, 5 V / 780 mA, 8 dBm max RF,
                              100 m for RCS = 1 m^2, Windows 7+, Python via
                              usb.dll
  ADF5901 datasheet rev B  -- 24 GHz VCO + 2x PA, +8 dBm typ per channel
  ADF5904 datasheet rev A  -- 4-ch downconverter, 22 dB gain, 10 dB NF DSB
  ADAR7251 datasheet rev 0 -- 16-bit 4-ch sigma-delta, 0.3-1.8 MSPS, LNA+PGA
                              9..45 dB in 6 dB steps, noise table vs gain and
                              frequency (Table 1), 400 mW
  ADI EngineerZone 589531  -- chirp period >= N x 1 us + 22 us (UG says 20);
                              50 ms frame -> 20 Hz max update

The thing this module exists to make explicit: TinyRad trades the K-LC6's
free-running CW flexibility for a frame-based FMCW engine with a hard floor
on chirp period. That floor sits right on top of a DJI Mini 3's blade Doppler,
and no amount of processing moves it.
"""
from __future__ import annotations

import numpy as np

from . import klc6

# ---------------------------------------------------------------- constants
C = klc6.C
F_LO, F_HI = 24.0e9, 24.25e9
F_CENTER = 24.125e9
LAMBDA = C / F_CENTER                     # 0.012427 m, same band as K-LC6
HZ_PER_MPS = 2.0 / LAMBDA

# ---- ADF5901 transmitter (datasheet table, 3.3 V, 25 C) ----
TX_POWER_DBM = 8.0                        # per PA output, typ (2 min / 10 max)
N_TX = 2

# ---- ADF5904 receiver ----
RX_GAIN_DB = 22.0                         # voltage conversion gain, differential
RX_NF_DB = 10.0                           # DSB at 100 kHz IF
RX_NF_BLOCKED_DB = 15.0                   # with a -30 dBm interferer present
RX_IP1DB_DBM = -10.0
N_RX = 4

# ---- ADAR7251 ADC (datasheet rev 0, Table 1, fs = 1.2 MSPS) ----
ADC_BITS = 16
ADC_FS_HZ = 1.0e6                         # what the 1 us/sample chirp rule implies
ADC_FS_OPTIONS = (300e3, 450e3, 600e3, 900e3, 1.2e6, 1.8e6)
ADC_GAIN_RANGE_DB = 45.0                  # LNA + PGA, 9..45 dB in 6 dB steps
# Input-referred noise, nV/rtHz, indexed [gain_dB][measurement frequency].
# It is a 1/f profile: at 100 Hz even max gain sits at 10.8 nV/rtHz; by
# 100 kHz it is 2.4. FMCW beats land near 100 kHz and get the good number;
# a CW body line at 30-800 Hz lands in the knee. Either way it is 40 dB below
# what the AD2 did to the K-LC6.
ADC_NOISE_TABLE = {
    #  gain:  (100 Hz, 1 kHz, 100 kHz)
    9:  (44.7e-9, 16.0e-9, 9.7e-9),
    15: (23.6e-9, 8.7e-9, 5.2e-9),
    21: (15.0e-9, 5.4e-9, 3.3e-9),
    27: (12.0e-9, 4.3e-9, 2.67e-9),
    33: (11.3e-9, 4.0e-9, 2.5e-9),
    39: (10.9e-9, 3.86e-9, 2.44e-9),
    45: (10.8e-9, 3.83e-9, 2.4e-9),
}
ADC_NOISE_FREQS_HZ = (100.0, 1e3, 100e3)
ADC_NOISE_MAXGAIN_V_RTHZ = 2.4e-9         # headline figure, 100 kHz
ADC_FULLSCALE_VPP = {0: 5.6, 9: 1.987, 15: 0.995, 21: 0.498, 27: 0.249,
                     33: 0.124, 39: 0.062, 45: 0.031}
ADC_PASSBAND_HZ = 200e3                   # -0.1 dB at 0.166 fs, fs = 1.2 MSPS
ADC_HPF_CORNER_HZ = (0.729, 93.3)         # programmable, 8 steps
ADC_POWER_W = 0.400                       # 4 ch at 1.2 MSPS, internal DVDD


def adc_noise_v_rthz(pga_gain_db, freq_hz):
    """Input-referred ADAR7251 noise at a gain step and IF frequency (log interp)."""
    g = min(ADC_NOISE_TABLE, key=lambda k: abs(k - pga_gain_db))
    col = np.array(ADC_NOISE_TABLE[g])
    lf = np.log10(np.clip(freq_hz, ADC_NOISE_FREQS_HZ[0], ADC_NOISE_FREQS_HZ[-1]))
    return float(10 ** np.interp(lf, np.log10(ADC_NOISE_FREQS_HZ), np.log10(col)))

# ---- waveform engine limits ----
# UG-1709 p.13: "the sampling rate is fixed to 1 MHz ... The framework requires
# that N x tS + 20 us is smaller than the PWM period value." EngineerZone
# 589531 quotes 22 us. Use the UG's 20 and treat 22 as margin.
CHIRP_DEAD_TIME_S = 20e-6                 # per chirp, on top of N x 1 us
SAMPLE_PERIOD_S = 1e-6                    # fixed, not configurable
MAX_UPDATE_HZ_AT_50MS = 20.0
# UG-1709 p.13: N is independent of ramp duration. If N*tS > tRampUp the
# down-chirp is sampled too; if smaller, only the first part of the ramp is.

# ---- antenna (UG-1709 Table 1, Table 2, Figure 6) ----
ANT_GAIN_DBI = 12.6                       # realized gain, single serial-fed 6-patch
ANT_BEAM_H_DEG = 76.5                     # single element, 3 dB
ANT_BEAM_V_DEG = 17.6
ANT_SIDELOBE_DB = -20.0                   # E plane
BEAM_AZ_DEG = 75.0                        # array, as ADI states it
BEAM_EL_DEG = 15.0
ANGLE_RES_CLAIMED_DEG = 20.0
# Table 2, mm, Tx2 at the origin. Rx pitch is 6.2175 mm = lambda/2 at
# 24.125 GHz (6.2133 mm); Tx pitch 18.654 mm = 3 lambda/2. The 2x4 MIMO
# therefore forms a 7-element lambda/2 virtual array with ONE overlapping
# position (UG: "two elements overlap to allow the implementation of motion
# compensation algorithms"). Not 8. That explains the 20 deg claim landing
# between the 4- and 8-element predictions.
ANT_POS_MM = {"Tx1": -18.654, "Tx2": 0.000,
              "Rx1": 31.014, "Rx2": 37.231, "Rx3": 43.449, "Rx4": 49.666}
RX_SPACING_LAMBDA = 0.5
N_VIRTUAL = 7                             # distinct virtual positions
PCB_MM = (85.6, 54.0)

# ---- electrical (UG-1709 Table 3) ----
SUPPLY_V = 5.0
SUPPLY_A = 0.780                          # all four Rx enabled
POWER_W = SUPPLY_V * SUPPLY_A             # 3.9 W
TX_ONOFF_ISOLATION_DB = 30.0

# ---- ADI's own reference measurement (UG-1709 Table 4, Fig. 27) ----
# 78 mm corner cube (~1 m^2) at 5 m, fs 1 MHz, N 256, period 300 us, sweep
# 23.95-24.25 GHz (300 MHz -- 50 MHz outside the ISM band), ADC gain 21 dB.
# Together with "range = 100 m for RCS = 1 m^2" this anchors the link budget.
REF_RCS_M2 = 1.0
REF_RANGE_M = 5.0
REF_N = 256
REF_PERIOD_S = 300e-6
REF_ADC_GAIN_DB = 21.0
REF_SWEEP_HZ = 300e6
CLAIMED_RANGE_1M2 = 100.0

# ---- software (UG-1709 pp.1, 13) ----
GUI_OS = "Windows 7 or more recent"
PYTHON_USB_LAYER = "usb.dll"              # Windows DLL shipped in the Python tree
MATLAB_USB_LAYER = "USB mex driver (DemoRadUsb)"
USB_DEVICE_NAME = "BF707 Bulk Device"     # generic bulk endpoint -- libusb-able
ONBOARD_TEMP_SENSOR = "AD7415"            # +/-0.5 C, on the BOM -- use it

# ---- regulatory (47 CFR 15.245 / 15.249) ----
# 15.245 field-disturbance sensors: 24.075-24.175 GHz, 2500 mV/m @ 3 m
# 15.249 general:                   24.0-24.25 GHz,   250 mV/m @ 3 m
FCC_15245_BAND_HZ = (24.075e9, 24.175e9)
FCC_15245_FIELD_V_M = 2.5
FCC_15249_FIELD_V_M = 0.25
MEAS_DIST_M = 3.0

K_BOLTZ = 1.380649e-23
T0 = 290.0


# ------------------------------------------------------------------ helpers
def field_to_eirp_dbm(e_v_per_m, d_m=MEAS_DIST_M):
    """FCC field-strength limit -> EIRP. E = sqrt(30 P)/d  =>  P = (E d)^2/30."""
    return 10 * np.log10((e_v_per_m * d_m) ** 2 / 30.0 * 1e3)


def eirp_dbm(tx_power_dbm=TX_POWER_DBM, ant_gain_dbi=ANT_GAIN_DBI, n_tx_coherent=1):
    """EIRP per transmitter. TinyRad time-multiplexes its two Tx for MIMO,
    so only one radiates at a time -- n_tx_coherent stays 1 unless you
    deliberately beamform on transmit, which the stock firmware does not."""
    return tx_power_dbm + ant_gain_dbi + 10 * np.log10(n_tx_coherent)


# -------------------------------------------------------------- noise floor
def receiver_noise_dbm_per_hz(nf_db=RX_NF_DB):
    """kTB + NF, per Hz, referred to the antenna port."""
    return 10 * np.log10(K_BOLTZ * T0 * 1e3) + nf_db      # -174 + NF


def adc_penalty_db(pga_gain_db, if_freq_hz=100e3, rx_gain_db=RX_GAIN_DB,
                   nf_db=RX_NF_DB):
    """How much the ADC degrades the receiver's own noise floor.

    The ADF5904's output noise at the IF is kT*NF*gain. The ADAR7251's
    input-referred noise comes from its datasheet Table 1, which is a 1/f
    profile -- so the answer depends on where the signal sits in the IF.
    FMCW beats land near 100 kHz; a CW body line would sit at 30-800 Hz.
    This is the same calculation that indicted the AD2 in sim/klc6.py --
    there it came out at 25.5 dB.
    """
    n_dbm_hz = receiver_noise_dbm_per_hz(nf_db) + rx_gain_db
    v_rx = np.sqrt(10 ** (n_dbm_hz / 10.0) * 1e-3 * 50.0)
    v_adc = adc_noise_v_rthz(pga_gain_db, if_freq_hz)
    return 10 * np.log10(1 + (v_adc / v_rx) ** 2)


# ------------------------------------------------------------ link budget
def snr_db(sigma_m2, range_m, t_coherent_s, nf_db=RX_NF_DB,
           ant_gain_dbi=ANT_GAIN_DBI, tx_power_dbm=TX_POWER_DBM,
           n_rx_coherent=N_RX, n_tx_mimo=N_TX, adc_penalty=0.0,
           impl_loss_db=6.0, radome_loss_db=0.0):
    """Single-target SNR after coherent processing over t_coherent_s.

    Radar equation with thermal noise, not the RFbeam 'practical' formula --
    TinyRad publishes real NF and gain so we can do it properly. Then:
      * coherent integration gain = 1 / (noise bandwidth) = t_coherent_s
      * N_RX receivers summed coherently after DBF: +10log10(N_RX)
      * MIMO on transmit is time-multiplexed, so it buys aperture (angle),
        not SNR; each virtual channel sees one Tx at a time.
      * impl_loss_db covers windowing, straddle, CFAR, mismatch -- 6 dB is
        the textbook allowance and it is deliberately not RFbeam's 25 dB
        derating, because that number was reverse-engineered for THEIR
        module and does not transfer.
    """
    sigma_m2 = np.asarray(sigma_m2, dtype=float)
    range_m = np.asarray(range_m, dtype=float)
    pt = 10 ** ((tx_power_dbm - 30) / 10.0)
    g = 10 ** (ant_gain_dbi / 10.0)
    pr = pt * g * g * LAMBDA ** 2 * sigma_m2 / ((4 * np.pi) ** 3 * range_m ** 4)
    pr_dbm = 10 * np.log10(pr * 1e3) - 2 * radome_loss_db
    noise_dbm = receiver_noise_dbm_per_hz(nf_db) + 10 * np.log10(1.0 / t_coherent_s)
    return (pr_dbm - noise_dbm + 10 * np.log10(n_rx_coherent)
            - adc_penalty - impl_loss_db)


def detection_range_m(sigma_m2, t_coherent_s, snr_req_db=14.0, **kw):
    snr_1m = snr_db(sigma_m2, 1.0, t_coherent_s, **kw)
    return 10 ** ((snr_1m - snr_req_db) / 40.0)


# -------------------------------------------------------------- waveform
def min_chirp_period_s(n_samples):
    """EngineerZone 589531: period >= N x 1 us + 22 us."""
    return n_samples * SAMPLE_PERIOD_S + CHIRP_DEAD_TIME_S


def max_prf_hz(n_samples):
    return 1.0 / min_chirp_period_s(n_samples)


def max_unambiguous_doppler_hz(n_samples):
    return max_prf_hz(n_samples) / 2.0


def max_unambiguous_velocity_mps(n_samples):
    return max_unambiguous_doppler_hz(n_samples) / HZ_PER_MPS


def sweep_slope_hz_per_s(bw_hz, n_samples):
    """Slope when the ramp fills the sampled interval."""
    return bw_hz / (n_samples * SAMPLE_PERIOD_S)


def max_range_m(bw_hz, n_samples, fs=ADC_FS_HZ):
    """Beat frequency at Nyquist -> range."""
    return (fs / 2.0) * C / (2.0 * sweep_slope_hz_per_s(bw_hz, n_samples))


def range_resolution_m(bw_hz):
    return C / (2.0 * bw_hz)


def range_bins(bw_hz, n_samples):
    return max_range_m(bw_hz, n_samples) / range_resolution_m(bw_hz)


# --------------------------------------------------------------- angle
def array_beamwidth_deg(n_elements, spacing_lambda=RX_SPACING_LAMBDA):
    """3 dB beamwidth of a uniform linear array, broadside: ~0.886 lambda/(N d)."""
    return np.degrees(0.886 / (n_elements * spacing_lambda))


def angle_accuracy_deg(n_elements, snr_db_val, spacing_lambda=RX_SPACING_LAMBDA):
    """1-sigma bearing error at a given SNR -- the monopulse/CRB form.

    sigma_theta ~ beamwidth / (1.6 sqrt(2 SNR)). Like range accuracy, it is
    far better than resolution once SNR is decent, and it is the number that
    sets cross-range position error: sigma_x = R * sigma_theta.
    """
    bw = np.radians(array_beamwidth_deg(n_elements, spacing_lambda))
    snr = 10 ** (np.asarray(snr_db_val, dtype=float) / 10.0)
    return np.degrees(bw / (1.6 * np.sqrt(2.0 * snr)))


def cross_range_sigma_m(range_m, n_elements, snr_db_val, **kw):
    return np.asarray(range_m, float) * np.radians(
        angle_accuracy_deg(n_elements, snr_db_val, **kw))


def unambiguous_fov_deg(spacing_lambda=RX_SPACING_LAMBDA):
    """Grating-lobe-free field of view: +/- arcsin(lambda/(2d))."""
    arg = 1.0 / (2.0 * spacing_lambda)
    return 2.0 * np.degrees(np.arcsin(min(arg, 1.0)))


def virtual_array_mm():
    """Virtual element positions from UG-1709 Table 2: (Tx + Rx) for each pair.

    Returns the sorted distinct positions and the multiplicity of each. Should
    come out as 7 positions at ~6.22 mm pitch, one of them doubled.
    """
    tx = [ANT_POS_MM["Tx1"], ANT_POS_MM["Tx2"]]
    rx = [ANT_POS_MM[k] for k in ("Rx1", "Rx2", "Rx3", "Rx4")]
    pos = np.array([t + r for t in tx for r in rx])
    # The overlapping pair lands at 31.012 and 31.014 mm -- 2 um apart, which is
    # PCB tolerance, not a distinct element. Cluster to 0.1 mm.
    uniq, counts = np.unique(np.round(pos, 1), return_counts=True)
    return uniq, counts


def implied_snr_at_reference_db(snr_threshold_db=14.0):
    """What ADI's own two numbers imply about SNR of a 1 m^2 target at 5 m.

    If 1 m^2 is detectable to 100 m, and SNR falls as R^-4, then at 5 m the
    same target sits 40*log10(100/5) = 52 dB above threshold. This is a
    measured-by-the-manufacturer anchor for the model, independent of any
    assumption about implementation loss.
    """
    return snr_threshold_db + 40 * np.log10(CLAIMED_RANGE_1M2 / REF_RANGE_M)


# --------------------------------------------------------------- radome
def radome_loss_db(thickness_m, eps_r=2.75, tan_delta=0.008, f_hz=F_CENTER):
    """One-way loss of a flat dielectric sheet at normal incidence.

    Reflection from the slab (impedance mismatch, thickness-dependent) plus
    dielectric absorption. Polycarbonate at 24 GHz: eps_r ~2.7-2.8, tan-delta
    ~0.006-0.01. Half a wavelength IN THE MATERIAL is the reflectionless
    thickness; 3 mm is close to it, which is why the BOM's choice is a good
    one, but not exactly on it.
    """
    n = np.sqrt(eps_r)
    lam_m = C / f_hz
    k_d = 2 * np.pi * n * thickness_m / lam_m
    # Fabry-Perot slab, lossless part
    r = (1 - n) / (1 + n)
    denom = 1 - r ** 2 * np.exp(-2j * k_d)
    t = (1 - r ** 2) * np.exp(-1j * k_d) / denom
    refl_loss = -20 * np.log10(np.abs(t))
    # absorption: alpha = pi n tan_delta / lambda  (amplitude, per metre)
    alpha = np.pi * n * tan_delta / lam_m
    abs_loss = 20 * np.log10(np.e) * alpha * thickness_m
    return float(refl_loss + abs_loss)


def half_wave_thickness_mm(eps_r=2.75, f_hz=F_CENTER):
    return 1e3 * (C / f_hz) / (2 * np.sqrt(eps_r))


# -------------------------------------------------------------- weather
def rain_loss_db_per_km(rate_mm_hr, f_ghz=24.125):
    """ITU-R P.838 specific attenuation, rough coefficients at 24 GHz."""
    k, alpha = 0.124, 1.061           # horizontal pol, 24 GHz
    return k * rate_mm_hr ** alpha
