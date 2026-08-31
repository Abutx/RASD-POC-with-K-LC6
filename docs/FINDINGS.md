# K-LC6 / Analog Discovery 2 — Measured Findings

Everything in this file was measured on the bench, not assumed. Numbers are
reproducible from the scripts named alongside them. Session dates 2026-08-29 → 30.

Hardware: RFbeam **K-LC6-RFB-00D** (24.125 GHz CW, VCO pin wired), Digilent
**Analog Discovery 2 SN 210321B5D847**, external 5 V supply.

---

## 1. Hardware limits found the hard way

### 1.1 The AD2 has two input ranges, not a continuum

`SPEC.md` §3 asks for `range_v=0.05` (amplified) or `0.5` (bare). **Neither
exists.** The device reports exactly two: **5 V and 50 V peak-to-peak**. Both
requested values silently clamp to 5 Vpp.

Consequence: **LSB = 336 µV**, and the bare K-LC6 IF lands on **4–7 distinct ADC
codes** in CW mode. Amplitude resolution is effectively gone; only FFT processing
gain makes CW detection work at all. An IF amplifier is not optional for
static-target work — the `if_gain_db` field in the metadata schema is load-bearing.

### 1.2 Sustained sample-rate ceiling ≈ 30 s at 100 kSa/s

| capture | result |
|---|---|
| 30 s @ 100 kSa/s | 0 lost, 0 corrupted |
| 60 s @ 100 kSa/s | 0 lost, **9,697 corrupted** |
| 300 s @ 100 kSa/s | **147,441 lost, 325,912 corrupted** |
| 60 s @ 50 kSa/s | 0 lost, 0 corrupted |

Plan long captures at 50 kSa/s (still above the spec's 48 kSa/s aliasing floor,
Nyquist 25 kHz clears the 15 kHz blade content) or keep 100 kSa/s runs under 30 s.

### 1.3 Device configuration 1 gives a 16,384-sample buffer

The AD2 exposes 8 selectable configurations. The default caps analog-in at 8,192,
which does **not** fit Config B's 128-chirp CPI (12,800 samples).

```
[0] analog_in  8,192   analog_out  4,096    <- default
[1] analog_in 16,384   analog_out  1,024    <- use this for FMCW
[2] analog_in  2,048   analog_out 16,384
```

`acquire.open_device_cfg()` opens configuration 1. Note `AnalogDiscovery2(...)`
only *selects*; `.open()` must still be called or `device.analog_input` is `None`.

---

## 2. Acquisition traps (all cost real time; all now guarded)

### 2.1 A progress callback destroys the capture

Passing any callback to `recorder.record()` sends dwfpy down a path that never
fills the channel buffer. Measured side by side at 30 s:

| | samples returned | non-zero | rms |
|---|---|---|---|
| no callback | 3,000,000 | 3,000,000 | 176 µV |
| **with callback** | 3,000,000 | **25** | 39 µV |

The callback fired **once**, at the end, and `lost`/`corrupted` both reported
clean. A 300 s baseline came back as 29,999,966 zeros followed by 34 real
samples. **There is no safe way to watch a record-mode capture in progress.**
`acquire.record()` keeps a `progress` parameter for experiments only, with a
warning.

Separately: a callback returning `True` unconditionally records **forever** — a
300 s request ran past 555 s. The file is only written after `record()` returns,
so that capture was lost entirely.

### 2.2 Every record starts with one buffer of zeros

Exactly **8,192 zero samples** (163.8 ms at 50 kSa/s), one contiguous run from
index 0, with `lost`/`corrupted` reporting clean. 8,192 zeros are a broadband
impulse that smears across the whole spectrum. `acquire.record()` trims the
leading run and hard-errors if zeros appear anywhere else.
`scripts/analyze.py --repair` fixes older captures without re-recording.

### 2.3 `lost`/`corrupted` do not detect an unfilled buffer

Both counters read 0 in every failure above. Integrity must be checked from the
samples themselves.

---

## 3. Interference: the empty room is not empty

`out/baseline/20260829_061237_empty_baseline_60s.npz` — 59.8 s, 2,991,808
samples @ 50 kSa/s, 0 lost, 0 corrupted, 0 dropouts, rms 158.4 µV.

**11 of the 20 strongest lines are 60 Hz mains harmonics**, and they land
squarely in the human-motion band:

| Hz | m/s | dB over median |
|---|---|---|
| 59.99 | 0.373 | **35.9** |
| 180.05 | 1.119 | 28.3 |
| 300.03 | 1.864 | 22.7 |
| 240.04 | 1.491 | 15.6 |
| 2880.76 | **17.899** | 12.3 |

60 Hz alone converts to a very plausible-looking **0.373 m/s**, and the 48th
harmonic sits at 17.9 m/s — fan-blade-tip territory. **Any detector that does not
notch 60 Hz multiples will label this empty room as a walking human, and later as
a drone.** `process.band_peak()` and `scripts/live.py` both notch before deciding.

Non-mains content is all below 17 dB and clustered under 0.3 m/s (slow drift).

A transient comb at exactly 1000/2000/3000/4000/5000 Hz (20–28 dB) appeared in
one capture and did not reproduce at other sample rates — something switching in
the room, not the chain. It lands at 6–31 m/s, so watch for it in drone captures.

---

## 4. CW mode — works

Live detector `scripts/live.py`. Single-shot acquisition (not record mode),
8192-point frames, ~12 fps acquisition ceiling.

Measured over a 352 s session with a person moving:

```
real motion        7-15 dB over the empty-room baseline, peak 0.46 m/s median
noise tail         2-3 dB, peak velocity pinned at the 0.30 m/s floor bin
still-room spread  0.83-1.06 dB frame to frame
```

43% of all triggers at a 2 dB threshold sat in the lowest 2–3 dB bin — i.e. noise.
**A 5 dB threshold keeps every genuine event with margin and removes two-thirds
of the false alarms.** Default is 2 dB by explicit request; `--threshold 5` is
what the data supports.

### 4.1 Display performance

matplotlib is the bottleneck, not the radio:

| method | draw time | fps |
|---|---|---|
| `draw_idle` + `pause`, constrained_layout | 625 ms | 1.6 |
| `draw_idle` + `pause`, tight_layout | 265 ms | 3.8 |
| matplotlib blitting | 162 ms | 6.2 |
| Tk + PIL, new `PhotoImage` each frame | 65 ms | 6.5 |
| **Tk + PIL, `PhotoImage` reused via `paste()`** | **2–4 ms** | **11.2** |

Constructing a fresh `ImageTk.PhotoImage` per frame re-converts the bitmap into
Tk's internal format — that alone was ~45 ms. Reuse one and `paste()` into it.

---

## 5. FMCW mode

### 5.1 VCO tune line — verified good

`scripts/vco_check.py --probe` (CH2 clipped to X1 pin 5, 2− grounded):

```
  W1 cmd  pin5 meas    error
   0.500     0.5438   +0.0438
   2.500     2.4773   -0.0227
   4.500     4.4760   -0.0240     max error 0.044 V
```

Pin 5 tracks W1 across the full 0.5–4.5 V. **W1 drives the 4.7 kΩ pull-up fine —
no op-amp buffer needed.** SPEC §10.1 step 2 passes.

> Trap: measuring pin 5 on the 5 Vpp range centred at 0 only spans ±2.75 V, so
> everything above reads back as a flat 2.7434 V and looks exactly like the VCO
> pin being clamped. Offset the channel to +2.5 V. This produced a wrong
> "add a buffer" verdict before it was caught.

IF DC responds monotonically to tune voltage: 15.8 mV → 3.3 mV over 0.5–4.5 V
(~3.1 mV/V), confirming the VCO tunes.

### 5.2 Chirp sync — verified, and the naive test is misleading

Within one acquisition, chirps overlay at **+0.997 correlation / 25.7 dB shape
SNR whether or not the trigger is enabled** — because `fs/ramp = 100000/1000` is
exactly 100 samples and both clocks come off the same AD2 oscillator. Chirp-to-
chirp coherence is automatic.

The trigger's real job is the **start phase**, visible only across acquisitions:

```
triggered:     chirp-0 correlation +0.989   ramp peak at [97,96,97,96,96,96]
free-running:  chirp-0 correlation -0.098   ramp peak at [51,37,61,85,75,3]
```

`scripts/chirp_check.py` tests both so it cannot give a false pass.
SPEC §10.9.1 **PASS**.

### 5.3 Sweep bandwidth ≈ 180 MHz, not 300 MHz

Calibrated against an operator standing at a known **2.5 m**, whose MTI peak
landed at **1.5 m**. Since `R_meas/R_true = B_true/B_assumed`, that gives
**B ≈ 180 MHz** and true range resolution **0.83 m** (not 0.50 m).

One data point, eyeballed distance — treat as ±30%. `--bw` is a flag everywhere.
A proper VCO-Lin style characterisation (SPEC §10.5) is still outstanding.

### 5.4 Ramp feedthrough dominates

Self-mixing puts **16.8 mV pk-pk / 3.6 mV rms** into the IF — roughly **100× the
entire CW signal level**. Sweeping wider makes it worse and pushes all energy to
low frequency:

| span V | IF rms | spectral centroid | 99% rolloff |
|---|---|---|---|
| 1.0 | 991 µV | 3037 Hz | 46250 Hz |
| 2.0 | 1839 µV | 814 Hz | 34200 Hz |
| 4.0 | 3586 µV | 236 Hz | **150 Hz** |

At the 4 V span used, **99% of IF energy is below 150 Hz** (< 1.5 m equivalent).

`process.range_profile()` only removed the mean — a linear ramp minus its mean is
still a linear ramp. Polynomial detrending per chirp is required; order 2–3 drops
the profile dynamic range from 48 dB to ~10 dB by removing the ramp itself.

### 5.5 Static targets: NOT detectable. Moving targets: strongly detectable.

Foiled cardboard corner reflector (~2.7 m² RCS) at 3–5 m, 320 chirps averaged
per condition, box-in vs box-out:

```
best rise      +0.82 dB at 3.00 m
noise floor     0.76 dB (split-half of the same condition)
significance    1.1 sigma          -> NO DETECTION
correlation between two independent box-present runs: +0.020
```

A real target appears in the same bin in both runs. It did not. The 2.0–2.5 m
feature seen in single captures is **residual VCO nonlinearity** — identical with
the box in and out (+0.03, +0.15 dB).

Same hardware, same day, moving person via consecutive-chirp MTI:

```
 range m   moving   static   difference
    1.00    -28.9    -35.0        +6.07
    1.50    -29.4    -36.1        +6.69   <- 12 sigma
    2.00    -33.3    -36.3        +3.03
    4.00    -40.5    -41.2        +0.65
    7.00    -41.6    -41.5        -0.09
```

**MTI is worth ~30 dB.** Consecutive chirps cancel everything stationary —
feedthrough, walls, bench, VCO nonlinearity — because none of it moves. This is
why the PPI scope notches zero-Doppler.

SPEC §10.9: #1 PASS, #2 blocked (needs IF gain), #3 blocked (same), #4 PASS in
range, #5 PASS.

### 5.6 Why "CW works but the box is invisible" is not a contradiction

CW never detected a static object either. Every CW detection this session was
*motion*. A stationary box returns at exactly 0 Hz Doppler — the same bin as the
wall behind it and the transmitter's own leakage. CW is physically incapable of
separating it. CW proves the transmitter, antennas, receiver and IF path work;
it says nothing about the sweep, which CW does not use.

---

## 6. Live PPI scope

`scripts/ppi.py`. Config B, 12,800-sample CPI in 139 ms → **7.2 CPI/s**, 51 range
bins × 128 Doppler bins, ±3.11 m/s unambiguous.

**There is no bearing information in this data.** One Tx, one Rx: a target at
+10° and one at −10° produce identical samples. Detections are drawn as arcs
spanning the antenna beamwidth, which is what an azimuth-unresolved target
genuinely looks like. Collapsing the arc to a dot requires multiple RX channels
and beamforming — a hardware change, not a processing one.

Verified live: 578 CPIs over 79.5 s, operator visible while moving.

---

## 7. Open items

1. **VCO linearity correction** (SPEC §10.5) — not implemented. Expect smeared
   and slightly offset range peaks until it is.
2. **Sweep bandwidth** — 180 MHz is a single-point estimate. Needs a proper
   frequency-vs-voltage characterisation.
3. **IF amplifier** — the 336 µV LSB is the wall for static-target range. Nothing
   in software fixes it.
4. **Q channel** — X1 pin 1 is still unwired. I/Q would resolve approach from
   recede and roughly double the information available to a classifier.
5. **CW dataset** — only the `empty` class exists. `human`, `fan_plastic`,
   `fan_foil`, `drone` are all outstanding.

---

## 8. Fan detection (2026-08-30)

A household fan was placed ~2.5 m from the module, room otherwise empty. Datasets
in `out/fan/`, inventory in [`DATA.md`](DATA.md).

### 8.1 Duration-matched CW test — the fan's excess is MAINS, not blades

`scripts/fan_matched_test.py`. Two 60 s fan-on captures at 50 kSa/s against the
59.8 s empty baseline at the same rate, so block counts are identical. The two
fan-on captures give a **measured same-condition floor** rather than an assumed one.

```
same-condition floor (fan_on_0 - fan_on_1):  1.37 dB
fan minus empty:                             std 1.21 dB
excess variance beyond the floor:            0.00 dB
```

The fan-vs-empty difference is **no more structured than two fan-on captures
differ from each other**. There is no broadband blade signature.

Of the 20 strongest difference lines, **13 are exact 60 Hz harmonics against 0.7
expected by chance** (~19x over), clustered at harmonics 44–60 (2.6–3.6 kHz):

| Hz | m/s | fan−empty | σ | origin |
|---|---|---|---|---|
| 3479.00 | 21.62 | +6.92 | 5.0 | MAINS ×58 |
| 2879.33 | 17.89 | +6.41 | 4.7 | MAINS ×48 |
| 3359.22 | 20.87 | +6.24 | 4.5 | MAINS ×56 |
| 9769.06 | 60.70 | +4.80 | 3.5 | — (60.7 m/s is implausible for a fan) |

**The fan is detectable only as an electrical load.** Its motor raises supply hum;
the blades produce nothing measurable. This is the same trap documented in §3 —
and it is why any off/on comparison, on its own, cannot distinguish a fan's
Doppler from a fan's power draw.

### 8.2 FMCW MTI — controlled null against a known true positive

Identical processing applied to fan and to the known-good moving person:

```
             fan−empty     σ        person−empty
  2.50 m       -0.07     -0.2          +6.69      <- the fan is AT 2.5 m
  best         +0.44      1.2          +6.69  (14.8 sigma)
```

At exactly the fan's range the difference is −0.07 dB. The same MTI that resolves
a person at 14.8σ resolves nothing for the fan.

### 8.3 MTI blind speeds make FMCW the wrong tool for blades

Consecutive-chirp cancellation has nulls at Doppler = n × PRF, so the FMCW null
above is **partly a method artifact** and must not be read as pure target strength:

| | PRF | unambiguous | blind speeds | blade tips 15/20/25 m/s alias to |
|---|---|---|---|---|
| Config A | 50 Hz | ±0.16 m/s | every 0.31 m/s | **0.09 / 0.12 / 0.15 m/s** |
| Config B | 1 kHz | ±3.11 m/s | every 6.21 m/s | 2.57 / 1.36 / 0.15 m/s |

Blade Doppler (2414–4023 Hz) aliases into the clutter notch that MTI itself
creates. **CW is the correct modality for blade work** — no PRF, therefore no
blind speeds, and unambiguous to ±310 m/s at 100 kSa/s. The CW result in §8.1 is
therefore the load-bearing evidence, and it is a clean null.

### 8.4 Most likely physical cause

Plastic fan blades have very low radar cross-section at λ = 12.4 mm. SPEC §9 says
so explicitly and recommends taping aluminium foil to the blades; that was done
for the earlier 2.4 GHz work but **not** for this fan. A foiled fan is the single
cheapest experiment that would change the answer.
