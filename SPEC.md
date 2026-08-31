# K-LC6 Capture & Processing — Implementation Spec

**For Claude Code.** Build a Python toolchain that drives the Digilent Analog Discovery 2 to capture Doppler radar data from an RFbeam K-LC6, then processes it into spectrograms and a labelled dataset.

**Status: the hardware is confirmed working.** Manual test in WaveForms showed clear motion-dependent signal on Channel 1. This spec replaces the manual workflow with a scripted one.

---

## 1. Hardware context

| | |
|---|---|
| Radar | RFbeam K-LC6-RFB-00D, 24.125 GHz CW (VCO pin open) |
| Digitizer | Digilent Analog Discovery 2, SN 210321B5D847 |
| Power | External 5 V into K-LC6 X1 pin 2 |
| Channel 1 (orange 1+) | K-LC6 X1 pin 3 — **IF I** |
| Channel 2 (blue 2+) | K-LC6 X1 pin 1 — **IF Q** (wire this next) |
| W1 (yellow) | K-LC6 X1 pin 5 — **VCO in**, connected. See §10 |
| Ground | X1 pin 4 + AD2 ⏚ + 1− + 2− + supply return, star-tied |

**Physical constant that drives everything:**

```
Doppler shift = 2v / λ,  λ = 12.43 mm at 24.125 GHz
             => 161.0 Hz per m/s
```

| Target | Speed | Doppler |
|---|---|---|
| Hand wave | 0.5–2 m/s | 80–320 Hz |
| Walking human | 1.5 m/s | ~240 Hz |
| Fan blade tips (foiled) | 15–25 m/s | 2.4–4.0 kHz |
| Quad blade tips | 75–95 m/s | **12–15 kHz** |

**Sample at ≥ 100 kSa/s.** Nyquist at 50 kHz clears the 15 kHz blade content with wide margin. Do not go below 48 kSa/s — the blade signature aliases and the module will appear broken when it is not.

---

## 2. Driver: use the `dwf` C API, not WaveForms scripting

```bash
pip install dwfpy numpy scipy matplotlib
```

`dwfpy` wraps Digilent's `dwf` shared library. The alternative is the official `ctypes` wrapper shipped with WaveForms (`dwfconstants.py` + `dwf.dll` / `libdwf.so`) — either is fine, but `dwfpy` is far less code.

**Two constraints that will bite:**

1. **WaveForms and a script cannot hold the device simultaneously.** Close the WaveForms application before running anything. If the open fails with a device-busy error, that is why.
2. **Default buffer is 8192 samples per channel.** At 100 kSa/s that is 82 ms. For captures longer than that you must use **record mode** (`AcqModeRecord`), which streams continuously to the host. Single-shot mode is only useful for quick diagnostics.

---

## 3. Module: `klc6/acquire.py`

### `open_device()`
Open the first AD2, return a handle. Raise a clear error if the device is busy (tell the user to close WaveForms).

### `configure(device, fs=100_000, channels=(0,), range_v=0.05, offset_v=0.0)`
- Enable the requested analog-in channels
- Set range. The AD2 has two hardware ranges: ±2.5 V (low gain) and ±25 V. Anything ≤ 5 V requests the low-gain path. Use `range_v=0.05` for amplified signals, `0.5` for a bare unamplified module.
- Set offset. The K-LC6 IF output sits at a **0.2 V DC offset** (LO leakage appearing as DC). With a DC-coupled front end, pass `offset_v=-0.2` to recentre.
- Set the sample rate

### `record(device, duration_s, fs)`
Streaming capture in `AcqModeRecord`.

- Loop reading available samples until `duration_s * fs` collected
- **Check the record status flags every iteration.** The API reports `lost` and `corrupted` sample counts. If either is non-zero, raise — a silently gap-filled array will produce spectrogram artifacts that look like real Doppler content.
- Return `np.ndarray` of shape `(n_channels, n_samples)`, float64, volts

### `capture_to_file(path, duration_s, fs, metadata)`
Wrap `record`, write an `.npz` containing:

```python
np.savez_compressed(path,
    data=samples,          # (n_channels, n_samples) float64
    fs=fs,
    channel_names=["I"],   # or ["I", "Q"]
    timestamp=iso8601_str,
    metadata=json.dumps(metadata))
```

**Store raw volts, never processed output.** Every reprocessing idea downstream depends on having the original samples.

---

## 4. Module: `klc6/process.py`

### Constants

```python
C = 299_792_458.0
F_CARRIER = 24.125e9
LAMBDA = C / F_CARRIER          # 0.012427 m
HZ_PER_MPS = 2.0 / LAMBDA       # 160.95
```

### `to_complex(data, channel_names)`
If both I and Q are present, return `I + 1j*Q`. If only I, return the real signal. Complex input makes the spectrum one-sided in the correct direction — approaching and receding become distinguishable. Real-only input gives a symmetric spectrum, which is fine for blade detection but cannot resolve direction.

### `preprocess(x, fs, hp_hz=20.0)`
- Remove DC (subtract mean)
- High-pass Butterworth at `hp_hz`, order 4, zero-phase (`sosfiltfilt`) — kills residual offset and mains hum without touching Doppler content
- Return float or complex array

### `spectrogram_mps(x, fs, nperseg=8192, overlap=0.75, vmax_mps=None)`
- `scipy.signal.spectrogram` with a Hann window
- `return_onesided=False` when the input is complex, `fftshift` the frequency axis
- **Convert the frequency axis to velocity: `v = f / HZ_PER_MPS`**
- Return `(t, v, S_db)` where `S_db = 20*log10(|S| + 1e-12)`

At 100 kSa/s with `nperseg=8192`: frequency resolution 12.2 Hz = **0.076 m/s**, time resolution 82 ms per frame with 75% overlap giving a 20 ms hop. Good for both hand waves and blade lines.

### `cadence_velocity_diagram(S, t)`
**This is the highest-value feature in the whole pipeline.** Take a second FFT along the *time* axis of the spectrogram magnitude, per velocity bin, then sum across bins.

Periodic blade flashes collapse into a sharp peak at the blade-pass frequency. A drone shows a strong stable cadence line; a fan shows a different one; a human wingbeat-equivalent is slow and irregular; static clutter has none.

Return `(cadence_hz, cvd_magnitude)`.

### `plot_spectrogram(t, v, S_db, title, vlim=(-6, 6), out_path=None)`
- `pcolormesh`, y-axis labelled `radial velocity (m/s)`, x-axis `time (s)`
- **Fix the colour scale** at `vmin = S_db.max() - 55`, `vmax = S_db.max()`. Auto-scaling per plot makes captures incomparable.
- Horizontal line at v=0
- `vlim=(-6, 6)` for humans and hands; `(-100, 100)` when looking at blade content

---

## 5. Module: `klc6/dataset.py`

### Filename convention

```
YYYYMMDD_HHMMSS_<class>_<distance>m_<aspect>_<nnn>.npz
20260827_143022_drone_02m_hover_003.npz
```

### Metadata schema

```json
{
  "class": "drone",
  "subclass": "toy_quad_3in",
  "distance_m": 2.0,
  "aspect": "hover",
  "throttle": "hover",
  "sample_rate": 100000,
  "channels": ["I"],
  "mode": "cw",
  "vco_v": null,
  "if_gain_db": 0,
  "range_v": 0.5,
  "environment": "indoor, carpet, fan off",
  "notes": ""
}
```

### Classes

`empty` · `human` · `fan_plastic` · `fan_foil` · `drone`

### Vary systematically

| Dimension | Values |
|---|---|
| Distance | 1, 2, 3, 5 m — and keep going until the signature vanishes |
| Aspect | hover, approach, recede, cross |
| Throttle (drone only) | low, hover, high — RPM shifts the blade line and the classifier must see the range |

**Target: 20+ captures per class, 30 s each.**

### `capture_session(class_name, **metadata)`
Interactive CLI: prompt for distance and aspect, count down 3 seconds, capture, save, print the output path. Should be usable one-handed while holding a drone.

---

## 6. Scripts

### `scripts/smoke_test.py`
Replicates the manual WaveForms test. 10 s capture at 100 kSa/s, single channel, prints peak-to-peak in mV and the dominant frequency in the 20 Hz – 1 kHz band, converted to m/s. Passes if peak-to-peak exceeds the empty-room baseline by 3× while moving.

### `scripts/live.py`
Rolling spectrogram, updates ~10 Hz.

- Configure once, stream in record mode, maintain a rolling image buffer
- **Use `im.set_array()` on an existing `imshow`, never a fresh `pcolormesh` per frame** — recreating the mesh is the usual cause of a display that lags further behind the longer it runs
- Fixed colour scale, established from the first 2 seconds and then frozen
- This is the demo view: turn a fan on and watch the smear appear

### `scripts/capture.py`
CLI wrapper over `capture_session`. `python -m scripts.capture --class drone --distance 2 --aspect hover`

### `scripts/analyze.py`
Load an `.npz`, produce a two-panel figure — spectrogram on top, cadence velocity diagram below — and save alongside as `.png`.

### `scripts/compare.py`
Load two or more `.npz` files and plot their spectrograms side by side with a shared colour scale.

**This produces the money shot: `fan_foil` next to `drone`.** That pair is the artifact the whole project rests on. Make it a first-class command.

---

## 7. Acceptance criteria

**Smoke test**
1. Script opens the device without WaveForms running
2. 10 s capture completes with zero lost or corrupted samples
3. Hand motion produces a spectrogram smear at ±0.5 to 2 m/s; still room shows only a line at 0

**Capture pipeline**
4. `capture.py` writes a correctly named `.npz` with complete metadata
5. Re-loading returns identical samples

**Processing**
6. Spectrogram y-axis is in m/s and a known hand speed lands where expected
7. Cadence velocity diagram shows a peak for a foiled fan and none for an empty room

**Live view**
8. Runs 5 minutes without falling behind real time or accumulating lag

---

## 8. Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Device open fails | WaveForms still holding it | Close the application |
| Only 8192 samples returned | Single-shot mode | Use `AcqModeRecord` |
| Lost/corrupted counts non-zero | Host can't keep up | Lower `fs`, or read more often in the loop |
| Trace pegged at rail | 0.2 V DC offset with DC coupling | Set channel offset to −0.2 V |
| Blade lines at implausible frequencies | Aliasing | `fs` below 48 kSa/s |
| Spectrogram all noise | Range set too coarse | `range_v=0.05` for amplified, `0.5` for bare |
| Live view lags progressively | Recreating the plot each frame | `set_array` on an existing image |

---

## 9. Build order

1. `acquire.py` — open, configure, record. Verify against `smoke_test.py`.
2. `process.py` — preprocess and spectrogram. Verify a hand wave lands at the right velocity.
3. `dataset.py` + `capture.py` — get to 20 captures per class.
4. `analyze.py` and `compare.py` — produce the fan-vs-drone pair.
5. `live.py` — last. It is the demo, not the science, and it is the hardest to debug.

**Do not build the live view first.** A rolling display hides whether a fault is in the radio, the acquisition, or the plotting loop.

---

## 10. FMCW mode — range

**Status: W1 is wired to X1 pin 5 (VCO in).** Everything below is additive — CW mode still works with pin 5 disconnected, and the CW path above remains the primary one for classification.

### 10.1 Characterise the VCO before writing any sweep code

Do this first. The datasheet's Note 3 quotes 1.65 V for an open VCO pin, but **that figure is for the 3.3 V variant's internal divider.** The -00D you have is the 5 V version with a 4.7 kΩ pull-up to 5 V, so the open-circuit voltage will be different.

1. Disconnect W1. Measure X1 pin 5 to ground. **Record the actual value.**
2. Reconnect W1. Set it to DC at 2.5 V. Measure pin 5 again.
   - Reads ≈2.5 V → the AD2 is driving the pull-up fine, no buffer needed
   - Reads noticeably higher → add a unity-gain op-amp follower between W1 and pin 5
3. Sweep W1 by hand as DC across the usable range with a stationary target in front. The IF output will wander as the carrier moves — crude confirmation the VCO is tuning at all.

Write the measured open-circuit voltage and the usable voltage span into a config file. Every sweep parameter depends on them.

### 10.2 Two chirp configurations — they do different jobs

Sweep bandwidth is 300 MHz, so range resolution is `c / 2B` = **0.5 m** in both cases.

**Config A — range only (start here)**

| | |
|---|---|
| Ramp | Sawtooth, 50 Hz → 20 ms period |
| Slope `S` | 300 MHz / 20 ms = 1.5 × 10¹⁰ Hz/s |
| Beat at 5 m | 500 Hz |
| Beat at 25 m | 2.5 kHz |
| Samples per chirp @ 100 kSa/s | 2000 |
| Doppler | **Effectively none** — PRF 50 Hz gives ±0.16 m/s unambiguous |

Good for a static range demo: put a corner reflector at a known distance, show the peak lands in the right bin. Useless for moving targets.

**Config B — range-Doppler map**

| | |
|---|---|
| Ramp | Sawtooth, 1 kHz → 1 ms period |
| Slope `S` | 300 MHz / 1 ms = 3 × 10¹¹ Hz/s |
| Samples per chirp @ 100 kSa/s | 100 |
| Max unambiguous range | `(fs/2) · c / (2S)` = 25 m |
| PRF | 1 kHz → ±500 Hz → **±3.1 m/s** |
| CPI | 128 chirps = 128 ms |
| Output | 100 range bins × 128 Doppler bins |

±3.1 m/s covers a walking human and a hovering drone. It does **not** cover blade tips.

### 10.3 The conflict — document this, do not try to solve it on this hardware

Blade Doppler is 12–15 kHz, which needs a PRF above 30 kHz to sample without aliasing. At 100 kSa/s that leaves 3 samples per chirp, which is not enough for a range FFT.

**On the K-LC6 you can have range or blade classification, not both in one waveform.** That is a fundamental time-bandwidth constraint, not a limitation of the implementation.

The production answer is interleaved modes — long chirps to detect and range, then a short-chirp burst or CW dwell on the detected cell to classify. Do not build that here. Just keep the two configs separate and label captures accordingly.

### 10.4 Acquisition changes

Add to `acquire.py`:

**`configure_chirp(device, f_ramp, v_low, v_high, shape="sawtooth")`**
Configure analog-out channel 0 (W1). Sawtooth preferred over triangle — a triangle's down-ramp inverts the beat frequency sign and you have to discard or separately process it.

**Chirp-synchronous triggering is mandatory.** Set the analog-in trigger source to the analog-out channel so every acquisition starts at the same ramp phase. Without it, chirps land at random phase offsets, the slow-time phase progression is meaningless, and the Doppler FFT integrates noise.

In `dwf` terms: set the acquisition trigger source to `trigsrcAnalogOut1`. Verify by capturing 8 consecutive chirps and overlaying them — they should be near-identical.

**`record_chirps(device, n_chirps, samples_per_chirp)`**
Return shape `(n_chirps, samples_per_chirp)` complex or real. Reshape from the flat record stream using the known chirp period. Assert the reshape is exact — a partial trailing chirp silently corrupts the last row.

### 10.5 Processing

Add to `process.py`:

**`beat_to_range(f_beat, S)`**

```python
R = f_beat * C / (2.0 * S)      # S = B / T_chirp, Hz/s
```

**`range_profile(chirp, fs, S, nfft=None)`**
Window (Hann), FFT along fast time, take the magnitude, convert the frequency axis to metres with `beat_to_range`. Return `(range_m, magnitude)`.

**`range_doppler(chirps, fs, S, f_prf)`**
- Range FFT along axis 1 (fast time)
- Doppler FFT along axis 0 (slow time), `fftshift`
- Convert axis 0 to m/s: `v = f_doppler / HZ_PER_MPS`
- Convert axis 1 to metres via `beat_to_range`
- Return `(range_m, velocity_mps, S_db)`

**`clutter_notch(rd_map)`**
Zero or heavily attenuate the zero-Doppler column. Static returns — walls, the bench, the module's own feedthrough — are enormous compared to a target. Without this the map is one bright stripe and nothing else.

**`vco_linearity(v_points, f_points)`**
A voltage ramp does not produce a linear frequency ramp, and FMCW range accuracy depends entirely on linearity. RFbeam publishes a tool called **VCO-Lin** that computes the non-linearity from three known frequency/voltage points. Implement the same idea: fit a polynomial to measured frequency-vs-voltage, invert it, and pre-distort the W1 waveform so the *frequency* sweep is linear.

Until this is done, expect range peaks to be smeared and slightly offset. That is expected, not a bug.

### 10.6 Feedthrough

The datasheet notes the FMCW ramp is visible at the IF outputs from self-mixing — roughly 20 mVpp on the -00x variant. It appears as a large, low-frequency component synchronous with the ramp.

The 20 Hz high-pass in `preprocess` removes most of it. What remains lands in the near-zero range bins, which is why `clutter_notch` also helps here.

### 10.7 Metadata additions

```json
{
  "mode": "fmcw",
  "ramp_shape": "sawtooth",
  "ramp_hz": 1000,
  "vco_v_low": 0.5,
  "vco_v_high": 4.5,
  "sweep_bw_hz": 300e6,
  "chirp_period_s": 0.001,
  "samples_per_chirp": 100,
  "chirps_per_cpi": 128,
  "vco_linearity_corrected": false
}
```

### 10.8 Scripts

**`scripts/range_demo.py`** — Config A. Corner reflector at a measured distance, plot the range profile, print where the peak lands versus where it should. This is the FMCW equivalent of the smoke test.

**`scripts/rd_map.py`** — Config B. Live or single-shot range-Doppler map, range on one axis, velocity in m/s on the other, clutter notched.

### 10.9 Acceptance criteria

1. Eight consecutive chirps overlay near-identically (trigger sync works)
2. A corner reflector at a known distance produces a range peak within ±1 m of truth (one range cell)
3. Moving the reflector moves the peak in the expected direction
4. A walking human produces a track in the range-Doppler map with range decreasing and negative velocity when approaching
5. Zero-Doppler clutter is suppressed enough that a human at 3 m is visible

### 10.10 Build order

Do this **after** the CW dataset is collected. The classification claim is the POC; range is a separate demonstration and it does not gate anything.

1. Characterise the VCO (§10.1) — measurement, no code
2. `configure_chirp` + trigger sync, verified by overlaying chirps
3. `range_profile` + `range_demo.py` with a corner reflector
4. `range_doppler` + clutter notch
5. Linearity correction last — everything works without it, just less precisely

---

## 11. Notes

- Wire Channel 2 to X1 pin 1 (IF Q) as soon as convenient. Complex I/Q resolves approach from recede and roughly doubles the information the classifier sees.
- The K-LC6 does about 7.5 m on a small quad by RFbeam's own formula. Short range here is a hardware limit, not a failure — do not chase it.
- **The measurement that matters most: the distance at which the drone signature disappears.** Capture at increasing range until it does, and record that number. It calibrates the link budget for the real product.
