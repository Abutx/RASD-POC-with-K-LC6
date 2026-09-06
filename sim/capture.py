"""Render a clean simulated signal into what the AD2 would actually record.

A drone signature that ignores the front end is a lie in the direction that
matters: everything hard about this project is between the IF pin and the
sample buffer. So the chain here is the real one, in order --

    module IF noise  ->  optional external IF amp  ->  DC pedestal
      ->  mains comb  ->  AD2 analog front end  ->  336 uV quantiser  ->  clip

Output is written in exactly the repo's capture format (.npz of raw volts plus
a .json sidecar, source="synth"), so simulated captures flow through
dataset.manifest, dataset.loader and scripts/analyze.py with no special-casing.
That is deliberate: a detector that only works on synthetic data is worthless,
and the only way to catch that is to make the two indistinguishable to the
tooling.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from . import klc6

# Mains comb, measured from out/baseline/...empty_baseline_60s.npz at 0.763 Hz
# bins: dB by which each 60 Hz harmonic stands over the local median floor.
# The shape matters more than any single value -- it dies out by ~6 kHz, which
# is what puts a Mini 3's 7.8 kHz blade tips in clean spectrum and what put the
# fan's 2.4-4.0 kHz tips right in the middle of the junk (FINDINGS 8.1).
MAINS_HZ = 60.0
MAINS_COMB_DB = {1: 26.6, 2: 12.0, 3: 20.2, 4: 9.0, 5: 14.4, 6: 6.0, 7: 5.0,
                 8: 4.5, 10: 3.9, 12: 3.0, 15: 2.8, 20: 2.5, 25: 2.0, 30: 1.5,
                 40: 3.0, 44: 4.5, 48: 6.7, 52: 5.0, 56: 4.5, 58: 4.0,
                 60: 3.8, 70: 2.0, 80: 1.3, 90: 1.0, 100: 0.9}


def shaped_noise(n, fs, density_fn, rng):
    """Gaussian noise with a prescribed one-sided density profile, V/rtHz."""
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    # white noise of unit variance has density sqrt(2/fs); rescale per bin
    W *= density_fn(f) / np.sqrt(2.0 / fs)
    return np.fft.irfft(W, n=n)


def mains_comb(t, fs, floor_fn, rng, comb_db=None, bin_hz=0.763):
    """Deterministic 60 Hz harmonic comb at the measured levels."""
    comb_db = MAINS_COMB_DB if comb_db is None else comb_db
    out = np.zeros_like(t)
    for h, excess_db in comb_db.items():
        f0 = MAINS_HZ * h
        if f0 >= fs / 2.0:
            continue
        # A tone of amplitude A in a bin of width bin_hz stands
        # 20log10(A / (n*sqrt(2*bin_hz))) dB over a floor of density n.
        amp = floor_fn(f0) * np.sqrt(2.0 * bin_hz) * 10 ** (excess_db / 20.0)
        out += amp * np.sqrt(2.0) * np.cos(2 * np.pi * f0 * t
                                           + rng.uniform(0, 2 * np.pi))
    return out


def render(signal, fs, if_gain_db=0.0, dc_offset_v=0.013, seed=0,
           mains=True, quantise=True, lsb_v=klc6.AD2_LSB_V,
           fullscale_v=klc6.AD2_FULLSCALE_V, iq=False):
    """Complex IF signal in volts -> (n_ch, n) array as the AD2 would see it.

    `signal` is complex I+jQ at the module output, BEFORE any external gain.
    Returns real (1, n) unless iq=True, in which case (2, n) as [I, Q] --
    matching what TODAY.md Block 1 wires up.

    dc_offset_v defaults to the +13 mV the bench actually measures (README
    "expected healthy CH1 reading"), not the datasheet's +/-200 mV limit.
    """
    rng = np.random.default_rng(seed)
    signal = np.asarray(signal)
    n = signal.shape[-1]
    t = np.arange(n) / fs
    g = 10 ** (if_gain_db / 20.0)

    def module_density(f):
        return np.full_like(np.asarray(f, dtype=float), klc6.IF_NOISE_V_RTHZ)

    # The measured floor ALREADY contains the quantiser -- it was measured
    # through one. Since np.round() below puts quantisation noise back in for
    # real, the injected analog noise has to be the measured floor with the
    # quantiser's share taken out in quadrature, or the render comes out ~1.8 dB
    # hot and every SNR derived from it is quietly optimistic.
    q_density = (klc6.AD2_QUANT_RMS_V / np.sqrt(fs / 2.0)) if quantise else 0.0

    def ad2_density(f):
        floor = klc6.measured_floor_v_rthz(f)
        return np.sqrt(np.maximum(floor ** 2 - q_density ** 2, (0.05 * floor) ** 2))

    chans = []
    for k, comp in enumerate((np.real(signal), np.imag(signal))):
        if k == 1 and not iq:
            break
        # in front of the amp: target return + the module's own noise
        x = comp + shaped_noise(n, fs, module_density, rng)
        x = x * g
        # after the amp: everything the AD2 contributes, plus the room's mains
        x = x + shaped_noise(n, fs, ad2_density, rng)
        if mains:
            x = x + mains_comb(t, fs, ad2_density, rng)
        x = x + dc_offset_v
        x = np.clip(x, -fullscale_v, fullscale_v)
        if quantise:
            x = np.round(x / lsb_v) * lsb_v
        chans.append(x)
    return np.array(chans)


def save(path, data, fs, meta, channel_names=None):
    """Write .npz of raw volts + .json sidecar, per klc6_capture_spec.md section 3."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = channel_names or (["I", "Q"][:data.shape[0]])
    npz = path.with_suffix(".npz")
    np.savez_compressed(npz, data=data, fs=fs,
                        channel_names=np.array(names),
                        timestamp=_dt.datetime.now().astimezone().isoformat(),
                        metadata=json.dumps(meta))
    sidecar = dict(meta)
    sidecar.setdefault("source", "synth")
    sidecar.setdefault("mode", "cw")
    sidecar.setdefault("if_gain_db", 0)
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return npz


def realised_floor_v_rthz(x, fs, lo_hz, hi_hz, nperseg=65536):
    """Median PSD density of a rendered capture in a band -- for self-checking.

    Round-trips the render: feed it back the same measurement used to build the
    model and the answer should match klc6.measured_floor_v_rthz to a fraction
    of a dB. If it does not, the renderer is lying somewhere.
    """
    from scipy import signal as sig
    nperseg = int(min(nperseg, len(x)))
    f, p = sig.welch(x - np.mean(x), fs=fs, nperseg=nperseg)
    m = (f >= lo_hz) & (f < hi_hz)
    return float(np.sqrt(np.median(p[m])))
