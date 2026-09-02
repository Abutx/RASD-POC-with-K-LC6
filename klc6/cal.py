"""Calibration per TODAY.md Block 2 — compute the non-static cal files and
apply them on load.

Layout: cal/<date>/{dc.json, spurs_noamp.json, iq.json, background_cw.npz,
manifest.json}. Compute with scripts/calibrate.py; consume with
apply_calibration().
"""

import json
from pathlib import Path

import numpy as np
from scipy import signal as sig

CAL_ROOT = Path(__file__).resolve().parent.parent / "cal"


def cal_dir(date=None):
    if date is None:
        dirs = sorted(d for d in CAL_ROOT.glob("*") if d.is_dir())
        if not dirs:
            raise FileNotFoundError(f"no calibration under {CAL_ROOT}")
        return dirs[-1]
    return CAL_ROOT / date


# ---------------- compute (Block 2.1-2.4) ----------------

def compute_dc(x):
    """x: (2, n) still-room capture -> per-channel DC."""
    return {"I": float(np.mean(x[0])), "Q": float(np.mean(x[1]))}


def compute_spurs(x, fs, nperseg=65536, med_hz=80.0, thresh_db=10.0):
    """Welch -> median-filtered floor -> lines > thresh_db over it.

    Mains and switching spurs are coherent narrow tones; at coarse resolution
    their energy dilutes into the bin and disappears under the threshold (an
    8192-pt FFT put the 60 Hz line at only 7.8 dB vs the 35.9 dB FINDINGS §3
    measured at 0.095 Hz bins). Use fine bins and a frequency-defined median
    window so the floor estimate ignores the narrow spikes it's meant to find.
    """
    nperseg = min(nperseg, (len(x[0]) // 2) | 1 + 1)
    f, p = sig.welch(x[0] - np.mean(x[0]), fs=fs, nperseg=nperseg)
    p_db = 10 * np.log10(p + 1e-24)
    med_bins = max(11, int(round(med_hz / (f[1] - f[0]))) | 1)
    floor = sig.medfilt(p_db, med_bins)
    excess = p_db - floor
    idx = np.flatnonzero(excess > thresh_db)
    # collapse adjacent bins to one spur at the local max
    spurs = []
    for grp in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if grp.size:
            i = grp[int(np.argmax(excess[grp]))]
            spurs.append({"hz": round(float(f[i]), 2),
                          "excess_db": round(float(excess[i]), 1)})
    return {"fs": fs, "nperseg": nperseg, "thresh_db": thresh_db,
            "source": "amp-less baseline", "spurs": spurs}


def compute_iq(x):
    """Blind Gram-Schmidt estimate from a broadband (walking-person) capture."""
    I = x[0] - np.mean(x[0])
    Q = x[1] - np.mean(x[1])
    p_i, p_q, c = np.mean(I**2), np.mean(Q**2), np.mean(I * Q)
    sin_phi = float(c / np.sqrt(p_i * p_q))
    return {"sin_phi": sin_phi,
            "gain_ratio": float(np.sqrt(p_q / p_i)),
            "phase_error_deg": float(np.degrees(np.arcsin(sin_phi))),
            "verified": False}


def compute_background(x, fs, nperseg=8192):
    f, p = sig.welch(x[0] - np.mean(x[0]), fs=fs, nperseg=nperseg)
    return f, p


# ---------------- apply ----------------

def apply_calibration(x, fs, date=None, notch_q=30.0):
    """(2, n) raw volts -> calibrated complex I + jQ.

    DC removal -> Gram-Schmidt I/Q correction -> spur notches. The CW
    background reference is a spectrum, not a filter — consumers subtract it
    at display/detection time (load_background()).
    """
    d = cal_dir(date)
    dc = json.loads((d / "dc.json").read_text())
    I = x[0] - dc["I"]
    Q = x[1] - dc["Q"]

    iq_p = d / "iq.json"
    if iq_p.exists():
        iq = json.loads(iq_p.read_text())
        sin_phi = iq["sin_phi"]
        cos_phi = float(np.sqrt(1 - sin_phi**2))
        Q = (Q / iq["gain_ratio"] - I * sin_phi) / cos_phi
    z = I + 1j * Q

    sp_p = d / "spurs_noamp.json"
    if sp_p.exists():
        for s in json.loads(sp_p.read_text())["spurs"]:
            if 0 < s["hz"] < fs / 2 * 0.98:
                b, a = sig.iirnotch(s["hz"], notch_q, fs=fs)
                z = sig.lfilter(b, a, z)
    return z


def load_background(date=None):
    d = np.load(str(cal_dir(date) / "background_cw.npz"))
    return d["f"], d["p"]
