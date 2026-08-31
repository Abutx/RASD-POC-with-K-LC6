"""Query the manifest and emit training-ready arrays.

  from dataset import loader
  rows = loader.query(label="fan_on", mode="cw")
  train, test = loader.session_split(rows, test_frac=0.3)
  for spec, label in loader.spectrogram_windows(train, win_s=2.0):
      ...
"""

import sqlite3
from pathlib import Path

import numpy as np
from scipy import signal as sig

DB_PATH = Path(__file__).resolve().parent / "manifest.sqlite"


def query(label=None, mode=None, source=None, session=None, min_seconds=None,
          include_unlabeled=False):
    """Return capture rows (as dicts) matching the filters."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    clauses, args = [], []
    for col, val in (("label", label), ("mode", mode),
                     ("source", source), ("session", session)):
        if val is not None:
            clauses.append(f"{col} = ?")
            args.append(val)
    if min_seconds is not None:
        clauses.append("seconds >= ?")
        args.append(min_seconds)
    if not include_unlabeled and label is None:
        # non-training labels are opt-in only (query them explicitly by name)
        clauses.append("label NOT IN ('unlabeled', 'derived', 'bench_check')")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return [dict(r) for r in
            con.execute(f"SELECT * FROM captures {where} ORDER BY path", args)]


def session_split(rows, test_frac=0.3, seed=0):
    """Split BY SESSION (rule 3 in the README): windows from one capture must
    never straddle train and test."""
    sessions = sorted({r["session"] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)
    n_test = max(1, round(len(sessions) * test_frac)) if len(sessions) > 1 else 0
    test_s = set(sessions[:n_test])
    train = [r for r in rows if r["session"] not in test_s]
    test = [r for r in rows if r["session"] in test_s]
    return train, test


def load_signal(row):
    """Return (x, fs): 1-D signal for CW rows, 2-D (chirps, samples) for FMCW."""
    path = Path(row["path"])
    if path.suffix == ".wav":
        import soundfile as sf
        x, fs = sf.read(str(path), always_2d=True)
        if x.shape[1] >= 2:                       # stereo I/Q -> complex
            return x[:, 0] + 1j * x[:, 1], fs
        return x[:, 0], fs
    z = np.load(str(path), allow_pickle=True)
    fs = row["fs"]
    if "data" in z.files:
        return np.asarray(z["data"])[0], fs       # row 0 = I (see docs/DATA.md)
    if "chirps" in z.files:
        return np.asarray(z["chirps"]), fs
    if "cpis" in z.files:
        c = np.asarray(z["cpis"])
        return c.reshape(-1, c.shape[-1]), fs     # stack CPIs into chirp rows
    raise ValueError(f"unknown capture format: {path}")


def spectrogram_windows(rows, win_s=2.0, hop_s=1.0, nfft=4096, hp_hz=20.0):
    """Yield (log-mag spectrogram float32 (freq, time), label) per window.

    CW only — FMCW rows are skipped (range-Doppler featurization is a
    different tensor; add it when the model needs it).
    """
    for row in rows:
        if row["mode"] != "cw":
            continue
        x, fs = load_signal(row)
        if hp_hz and fs:
            sos = sig.butter(4, hp_hz, "highpass", fs=fs, output="sos")
            x = sig.sosfilt(sos, x - np.mean(x))
        step, width = int(hop_s * fs), int(win_s * fs)
        for start in range(0, len(x) - width + 1, step):
            seg = x[start:start + width]
            two_sided = np.iscomplexobj(seg)
            f, t, z = sig.stft(seg, fs=fs, nperseg=nfft,
                               noverlap=int(nfft * 0.75),
                               return_onesided=not two_sided)
            if two_sided:
                z = np.fft.fftshift(z, axes=0)
            spec = 20 * np.log10(np.abs(z) + 1e-12).astype(np.float32)
            yield spec, row["label"]
