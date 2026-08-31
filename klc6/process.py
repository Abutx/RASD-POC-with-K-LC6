"""Processing for K-LC6 captures: preprocess, spectrogram in m/s, cadence.

Velocity is the native unit here, not frequency. Doppler at 24.125 GHz is
161.0 Hz per m/s, so every plot axis is converted once, at the boundary.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram

C = 299_792_458.0
F_CARRIER = 24.125e9
LAMBDA = C / F_CARRIER            # 0.012427 m
HZ_PER_MPS = 2.0 / LAMBDA         # 160.95


def to_complex(data, channel_names):
    """I+jQ when both are present, else the real channel.

    Complex input makes the spectrum one-sided in the correct direction, so
    approach and recede are distinguishable. Real-only gives a symmetric
    spectrum -- fine for blade detection, useless for direction.
    """
    names = [str(n) for n in channel_names]
    data = np.atleast_2d(data)
    if "I" in names and "Q" in names:
        return data[names.index("I")] + 1j * data[names.index("Q")]
    return data[0].astype(float)


def preprocess(x, fs, hp_hz=20.0, order=4):
    """Remove DC and high-pass, zero-phase.

    The K-LC6 IF sits on a DC pedestal from LO leakage; the high-pass also
    clears mains hum without touching Doppler content above 20 Hz (0.12 m/s).
    """
    x = x - np.mean(x)
    if hp_hz and hp_hz > 0:
        sos = butter(order, hp_hz, btype="highpass", fs=fs, output="sos")
        if np.iscomplexobj(x):
            x = sosfiltfilt(sos, x.real) + 1j * sosfiltfilt(sos, x.imag)
        else:
            x = sosfiltfilt(sos, x)
    return x


def spectrogram_mps(x, fs, nperseg=8192, overlap=0.75):
    """(t, v, S_db) with the frequency axis converted to radial velocity."""
    noverlap = int(nperseg * overlap)
    onesided = not np.iscomplexobj(x)
    f, t, S = spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                          window="hann", return_onesided=onesided,
                          scaling="spectrum", mode="magnitude", detrend=False)
    if not onesided:
        f = np.fft.fftshift(f)
        S = np.fft.fftshift(S, axes=0)
    return t, f / HZ_PER_MPS, 20 * np.log10(S + 1e-12)


def band_peak(x, fs, lo_hz=20.0, hi_hz=1000.0, mains_hz=60.0, mains_tol=2.0):
    """Dominant NON-MAINS frequency in a band, plus its dB over the band median.

    Mains harmonics must be excluded or the verdict is worthless: a static room
    measured +60.0 Hz at 26.6 dB, which converts to a tidy 0.373 m/s and reads
    as a walking person. 60 Hz multiples are notched out before the search --
    they are the supply, not a target.
    """
    n = len(x)
    win = np.hanning(n)
    if np.iscomplexobj(x):
        sp = np.abs(np.fft.fftshift(np.fft.fft(x * win)))
        fr = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    else:
        sp = np.abs(np.fft.rfft(x * win))
        fr = np.fft.rfftfreq(n, 1 / fs)
    band = (np.abs(fr) >= lo_hz) & (np.abs(fr) <= hi_hz)
    if mains_hz:
        harmonic = np.abs(np.abs(fr) - np.round(np.abs(fr) / mains_hz) * mains_hz)
        band &= (harmonic > mains_tol) | (np.abs(fr) < mains_hz - mains_tol)
    if not band.any():
        return 0.0, 0.0, 0.0
    idxs = np.flatnonzero(band)
    i = idxs[int(np.argmax(sp[idxs]))]
    snr = 20 * np.log10((sp[i] + 1e-30) / (np.median(sp[idxs]) + 1e-30))
    return float(fr[i]), float(fr[i] / HZ_PER_MPS), float(snr)


def cadence_velocity_diagram(S_db, t):
    """FFT along the TIME axis of the spectrogram, summed over velocity bins.

    Periodic blade flashes collapse into a sharp peak at the blade-pass rate.
    A drone shows a strong stable cadence line, a fan a different one, and an
    empty room none at all.
    """
    S = 10 ** (S_db / 20.0)
    S = S - S.mean(axis=1, keepdims=True)          # per-bin DC removal
    nt = S.shape[1]
    if nt < 4:
        return np.zeros(0), np.zeros(0)
    dt = float(np.mean(np.diff(t))) if nt > 1 else 1.0
    win = np.hanning(nt)
    Cf = np.abs(np.fft.rfft(S * win, axis=1))
    cad = np.fft.rfftfreq(nt, dt)
    return cad, Cf.sum(axis=0)


# =====================================================================
# FMCW (SPEC.md section 10)
# =====================================================================

SWEEP_BW_HZ = 300e6
RANGE_RES_M = C / (2 * SWEEP_BW_HZ)      # 0.4997 m


def beat_to_range(f_beat, S):
    """Beat frequency -> range in metres. S = B / T_chirp, Hz/s."""
    return np.asarray(f_beat) * C / (2.0 * S)


def range_profile(chirp, fs, S, nfft=None):
    """One chirp -> (range_m, magnitude). Hann window, FFT along fast time."""
    x = np.asarray(chirp, dtype=float)
    x = x - x.mean()
    n = len(x)
    nfft = int(nfft or n)
    sp = np.abs(np.fft.rfft(x * np.hanning(n), n=nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    return beat_to_range(f, S), sp


def range_doppler(chirps, fs, S, f_prf, range_nfft=None, dopp_nfft=None):
    """(n_chirps, samples_per_chirp) -> (range_m, velocity_mps, S_db).

    Range FFT along fast time (axis 1), Doppler FFT along slow time (axis 0).
    The slow-time axis is only meaningful if every chirp started at the same
    ramp phase -- see acquire.record_chirps, which triggers on the analog-out.
    """
    x = np.asarray(chirps, dtype=float)
    nch, nsamp = x.shape
    rn = int(range_nfft or nsamp)
    dn = int(dopp_nfft or nch)

    x = x - x.mean(axis=1, keepdims=True)          # per-chirp DC
    rng = np.fft.rfft(x * np.hanning(nsamp)[None, :], n=rn, axis=1)
    rd = np.fft.fftshift(np.fft.fft(rng * np.hanning(nch)[:, None],
                                    n=dn, axis=0), axes=0)

    range_m = beat_to_range(np.fft.rfftfreq(rn, 1 / fs), S)
    f_d = np.fft.fftshift(np.fft.fftfreq(dn, 1 / f_prf))
    vel = f_d / HZ_PER_MPS
    return range_m, vel, 20 * np.log10(np.abs(rd) + 1e-12)


def clutter_notch(rd_db, vel, width_mps=None, depth_db=60.0):
    """Attenuate the zero-Doppler rows of a range-Doppler map.

    Walls, the bench and the module's own feedthrough are enormous next to any
    target; untouched, the map is one bright stripe at 0 m/s and nothing else.
    """
    out = np.array(rd_db, dtype=float, copy=True)
    dv = float(abs(vel[1] - vel[0])) if len(vel) > 1 else 0.0
    w = width_mps if width_mps is not None else 1.5 * dv
    out[np.abs(vel) <= w, :] -= depth_db
    return out


def vco_linearity(v_points, f_points, order=3):
    """Fit frequency-vs-voltage, then invert it to pre-distort the ramp.

    A linear voltage ramp does not give a linear frequency ramp, and FMCW range
    accuracy depends entirely on the frequency sweep being linear. Returns
    (f_of_v, v_of_f, rms_error_hz): feed v_of_f a linear frequency ramp to get
    the voltage waveform that produces it.

    Until this is applied, expect range peaks smeared and slightly offset --
    that is expected, not a bug.
    """
    v = np.asarray(v_points, dtype=float)
    f = np.asarray(f_points, dtype=float)
    if v.size < order + 1:
        order = max(1, v.size - 1)
    cf = np.polyfit(v, f, order)
    f_of_v = np.poly1d(cf)
    cv = np.polyfit(f, v, order)
    v_of_f = np.poly1d(cv)
    rms = float(np.sqrt(np.mean((f_of_v(v) - f) ** 2)))
    return f_of_v, v_of_f, rms
