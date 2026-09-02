"""1D range tracker on FMCW movers — TODAY.md Block 5.

Pipeline per CPI: order-3 polynomial detrend per chirp (FINDINGS 5.4) ->
range FFT -> consecutive-chirp MTI -> Doppler FFT -> zero-Doppler notch ->
2D OS-CFAR -> detections. Tracker: constant-velocity Kalman, 3-sigma gating,
confirm 5 hits in 8 CPIs, drop after 5 misses.
"""

from collections import deque

import numpy as np

C_LIGHT = 299_792_458.0
LAMBDA = C_LIGHT / 24.125e9


def process_cpi(chirps, fs, ramp_hz, bw_hz, notch_mps=0.25,
                guard=2, train=8, q=0.75, offset_db=10.0):
    """(n_chirps, spc) volts -> list of detections (range_m, vel_mps, snr_db)."""
    n_ch, spc = chirps.shape
    t = np.arange(spc)
    x = chirps - np.array([np.polyval(np.polyfit(t, row, 3), t)
                           for row in chirps])
    rng = np.fft.rfft(x * np.hanning(spc)[None, :], axis=1)
    mti = np.diff(rng, axis=0)                       # consecutive-chirp MTI
    rd = np.fft.fftshift(np.fft.fft(mti * np.hanning(n_ch - 1)[:, None],
                                    axis=0), axes=0)
    mag_db = 20 * np.log10(np.abs(rd) + 1e-12)

    slope = bw_hz * ramp_hz
    r_axis = np.arange(spc // 2 + 1) * (fs / spc) * C_LIGHT / (2 * slope)
    v_axis = np.fft.fftshift(np.fft.fftfreq(n_ch - 1, 1 / ramp_hz)) * LAMBDA / 2

    mag_db[np.abs(v_axis) < notch_mps, :] = -np.inf   # zero-Doppler notch

    dets = []
    for (i, j) in zip(*np.where(mag_db > -np.inf)):
        if r_axis[j] < 0.4 or r_axis[j] > 12.0:
            continue
        i0, i1 = max(0, i - guard - train), min(mag_db.shape[0], i + guard + train + 1)
        j0, j1 = max(0, j - guard - train), min(mag_db.shape[1], j + guard + train + 1)
        block = mag_db[i0:i1, j0:j1]
        gi, gj = i - i0, j - j0
        mask = np.ones(block.shape, bool)
        mask[max(0, gi - guard):gi + guard + 1,
             max(0, gj - guard):gj + guard + 1] = False
        ring = block[mask & np.isfinite(block)]
        if ring.size < 8:
            continue
        noise = np.quantile(ring, q)
        snr = mag_db[i, j] - noise
        if snr > offset_db:
            dets.append((float(r_axis[j]), float(v_axis[i]), float(snr)))
    # cluster detections within ~1.5 range bins: keep the strongest of each
    dets.sort()
    clustered = []
    for r, v, s in dets:
        if clustered and r - clustered[-1][0] < 1.3:
            if s > clustered[-1][2]:
                clustered[-1] = (r, v, s)
        else:
            clustered.append((r, v, s))
    return clustered


class Track:
    _next_id = 1

    def __init__(self, r, v, dt):
        self.id = Track._next_id
        Track._next_id += 1
        self.x = np.array([r, v], float)
        self.P = np.diag([0.83**2, 1.0])
        self.dt = dt
        self.hits = deque([1], maxlen=8)
        self.misses = 0
        self.age = 0
        self.confirmed = False
        self.v_hist = [v]

    def predict(self, q_accel=2.0):
        dt = self.dt
        F = np.array([[1, dt], [0, 1]])
        G = np.array([0.5 * dt**2, dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.outer(G, G) * q_accel**2
        self.age += 1

    def gate_sigma(self, r_var):
        return float(np.sqrt(self.P[0, 0] + r_var))

    def update(self, r, v, r_var, v_var=0.05**2):
        # measure both range and Doppler velocity (unaliased within +-3.1 m/s)
        H = np.eye(2)
        S = self.P + np.diag([r_var, v_var])
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ (np.array([r, v]) - self.x)
        self.P = self.P - K @ H @ self.P
        self.hits.append(1)
        self.misses = 0
        self.v_hist.append(v)
        if sum(self.hits) >= 5:
            self.confirmed = True

    def miss(self):
        self.hits.append(0)
        self.misses += 1

    def features(self):
        v = np.abs(self.v_hist)
        return {"mean_abs_v": float(np.mean(v)), "v_var": float(np.var(v)),
                "hover_frac": float(np.mean(v < 0.2))}


class Tracker:
    def __init__(self, dt=0.139, r_var=(0.83**2) / 12, gate_n=3.0, q_accel=6.0):
        self.dt, self.r_var, self.gate_n = dt, r_var, gate_n
        self.q_accel = q_accel
        self.tracks = []

    def step(self, detections):
        for tr in self.tracks:
            tr.predict(self.q_accel)
        used = set()
        for tr in self.tracks:
            gate = self.gate_n * tr.gate_sigma(self.r_var)
            cand = [(abs(r - tr.x[0]), k, r, v) for k, (r, v, s) in
                    enumerate(detections) if k not in used
                    and abs(r - tr.x[0]) < gate]
            if cand:
                _, k, r, v = min(cand)
                tr.update(r, v, self.r_var)
                used.add(k)
            else:
                tr.miss()
        for k, (r, v, s) in enumerate(detections):
            # spawn only if no existing track is plausibly responsible
            if k not in used and all(abs(r - t.x[0]) >
                                     2 * self.gate_n * t.gate_sigma(self.r_var)
                                     for t in self.tracks):
                self.tracks.append(Track(r, v, self.dt))
        self.tracks = [t for t in self.tracks if t.misses < 5]
        return [t for t in self.tracks if t.confirmed]
