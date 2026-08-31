"""Positive control: can a difference-of-averaged-spectra + measured-split-half-floor
pipeline (the same statistical machinery I will apply to the fan CW data) find the
known moving person in raw_chirps_moving.npz?"""
import sys, numpy as np
sys.path.insert(0, r'C:/dev/klc6')
from klc6 import process as P

OUT = r'C:/dev/klc6/out/fan/analysis'
mv = np.load(r'C:/dev/klc6/out/fmcw/raw_chirps_moving.npz')
st = np.load(r'C:/dev/klc6/out/fmcw/raw_chirps_box_out.npz')
fs = float(mv['fs']); B = 180e6; ramp = float(mv['ramp']); S = B*ramp
print(f'fs={fs} ramp={ramp} S={S:.3g} Hz/s  moving{mv["chirps"].shape} static{st["chirps"].shape}')

def detrend_poly(x, order=3):
    n = x.shape[-1]; t = np.linspace(-1,1,n)
    V = np.vander(t, order+1)
    coef, *_ = np.linalg.lstsq(V, x.T, rcond=None)
    return x - (V@coef).T

def spec_blocks(chirps, mti=True, order=3):
    """-> (range_m, block power spectra in dB, one row per chirp/pair)"""
    x = np.asarray(chirps, float)
    x = detrend_poly(x, order)
    if mti:
        x = np.diff(x, axis=0)
    n = x.shape[1]; w = np.hanning(n)
    sp = np.abs(np.fft.rfft(x*w, axis=1))**2
    f = np.fft.rfftfreq(n, 1/fs)
    return P.beat_to_range(f, S), sp

def avg_db(sp, idx):
    return 10*np.log10(sp[idx].mean(axis=0) + 1e-30)

def analyse(mti):
    r, spm = spec_blocks(mv['chirps'], mti)
    _, sps = spec_blocks(st['chirps'], mti)
    Km, Ks = spm.shape[0], sps.shape[0]
    dif = avg_db(spm, np.arange(Km)) - avg_db(sps, np.arange(Ks))
    # measured floor: split the MOVING condition into disjoint Km-vs-Ks-sized halves
    rng = np.random.default_rng(0); nulls=[]
    for _ in range(400):
        p = rng.permutation(Km)
        a, b = p[:Km//2], p[Km//2:Km//2+min(Ks, Km-Km//2)]
        nulls.append(avg_db(spm,a) - avg_db(spm,b))
    nulls = np.array(nulls)
    floor = nulls.std(axis=0)
    band = (r>0.5)&(r<8.0)
    z = np.where(floor>0, dif/np.maximum(floor,1e-9), 0.0)
    i = np.argmax(np.where(band, z, -1e9))
    print(f'\n--- MTI={mti}  Kmov={Km} Kstat={Ks} ---')
    print(f'  measured split-half floor (median over 0.5-8 m): {np.median(floor[band]):.3f} dB')
    print(f'  best bin: R={r[i]:.2f} m  rise={dif[i]:+.2f} dB  z={z[i]:.1f} sigma')
    for rr in [1.0,1.5,2.0,3.0,4.0,7.0]:
        j = np.argmin(abs(r-rr))
        print(f'   R={r[j]:.2f} m  d={dif[j]:+6.2f} dB  floor={floor[j]:.2f}  z={z[j]:+6.1f}')
    # cross-chunk reproducibility: two independent halves of the moving capture
    h0, h1 = np.arange(Km//2), np.arange(Km//2, Km)
    s0, s1 = np.arange(Ks//2), np.arange(Ks//2, Ks)
    d0 = avg_db(spm,h0)-avg_db(sps,s0); d1 = avg_db(spm,h1)-avg_db(sps,s1)
    rr = np.corrcoef(d0[band], d1[band])[0,1]
    print(f'  cross-half correlation of difference spectrum (0.5-8 m): r={rr:+.3f}')
    return r, dif, z, floor, band

for mti in (True, False):
    analyse(mti)
