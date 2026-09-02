# TODAY.md — Lab Session, Single K-LC6, No Amp

**Constraints:** one K-LC6, one AD2, no IF amplifier, no fan. Drone available. Room with real walking space. Everything here is gated on what `docs/FINDINGS.md` measured — moving targets work, static ones don't.

**Goal for the session:** leave with I/Q wired, the non-static calibration files written, a `human` dataset, a first `drone` capture at close range, and a working 1D tracker. That's a real day.

**Rule:** if something isn't working after a few honest attempts, log it in FINDINGS.md and move on. Don't burn the session on one blocker.

---

## Wiring — exact, do not deviate

Verify the K-LC6 pin 4 by continuity to the shield can before applying power. Only pin 4 beeps. Pin 5 is the end pin next to it; count away from there. Your earlier photo was the back view, so pin 1 is on the right in that orientation.

| From | To | Wire colour | Notes |
|---|---|---|---|
| TI board **5 V / VBUS** | breadboard **red rail** | — | verify 5 V, not 3.3 V, with a meter first |
| breadboard **red rail** | K-LC6 X1 **pin 2** (Vcc) | — | 85 mA |
| TI board **GND** | breadboard **blue rail** | — | |
| breadboard **blue rail** | K-LC6 X1 **pin 4** (GND) | — | the star point |
| AD2 **⏚** | breadboard **blue rail** | black | |
| AD2 **1−** | breadboard **blue rail** | orange/white | scope channel 1 negative |
| AD2 **2−** | breadboard **blue rail** | blue/white | scope channel 2 negative |
| AD2 **1+** | K-LC6 X1 **pin 3** (IF I) | orange | scope channel 1 |
| AD2 **2+** | K-LC6 X1 **pin 1** (IF Q) | blue | scope channel 2 — **new today** |
| AD2 **W1** | K-LC6 X1 **pin 5** (VCO) | yellow | FMCW only. Leave connected; set W1 to DC 2.5 V for CW blocks |
| K-LC6 **X2** (3-pin) | — | — | leave floating |
| AD2 V+, V−, W2, T1, T2, DIO | — | — | unused |

**No amplifier in the chain.** IF outputs go straight to the AD2. Add the 100 nF decoupling cap across the rails at pin 2 if the trace looks noisy; otherwise skip it.

**K-LC6 X1 pinout for reference:** 1 = IF Q · 2 = Vcc · 3 = IF I · 4 = GND · 5 = VCO in.

**AD2 configuration:** open with device configuration **1** (16,384-sample analog-in buffer) per FINDINGS §1.3. Sample at **50 kSa/s** for anything over 30 s.

---

## Block 1 — Wire Q and verify

**Hardware:** per the wiring table above. The only new connection is AD2 2+ (blue) → X1 pin 1, with 2− (blue/white) to the ground rail.

**Software**

`acquire.configure()` with `channels=(0, 1)`. `process.to_complex()` now returns `I + jQ`.

**Verify**

`scripts/smoke_test.py` with both channels. Hand wave at 30 cm. Both channels show motion. Plot I against Q for a 1 s window during steady motion — you should see something roughly circular or elliptical, not a line. A line means Q is dead or wired to the same pin as I.

**Gate:** both channels respond to motion, I-vs-Q trace is not degenerate.

---

## Block 2 — Calibration that doesn't need a static target

### 2.1 DC offset

10 s capture, room still, both channels. `dc = mean(x)` per channel. Save `cal/<date>/dc.json`.

### 2.2 Spur map, amp-less baseline

60 s at 50 kSa/s, both channels, room still, module pointed at a wall with nothing in front of it.

Welch, `nperseg=8192`, median filter 101 bins, threshold 10 dB. Save `cal/<date>/spurs_noamp.json` with the frequency list and a `source` field. FINDINGS §3 already has the mains lines — confirm they're still there and note anything new from the lab environment. Lab has different mains loads than home.

**This file is the reference for when the amp arrives.** Anything the amp adds shows up as a diff against it.

### 2.3 I/Q imbalance, blind estimate

Gram-Schmidt needs a broadband signal, not a single tone. **Have someone walk back and forth at 2–4 m for 30 s.** That's plenty.

```python
I -= I.mean(); Q -= Q.mean()
P_I, P_Q, C = mean(I**2), mean(Q**2), mean(I*Q)
sin_phi = C / sqrt(P_I * P_Q)
cos_phi = sqrt(1 - sin_phi**2)
gain_ratio = sqrt(P_Q / P_I)
```

Save `cal/<date>/iq.json` — `sin_phi`, `gain_ratio`, `phase_error_deg`, and `verified: false`. **Verification (IRR before/after on a single tone) waits for a clean single-tone target.** The drone at hover might give you one — see Block 4.

**Sanity check now:** apply the correction to the walking capture, form `I + jQ`, spectrogram. A person walking *toward* should show energy on one side of zero; walking *away* on the other. **If energy appears symmetrically on both sides, the correction is wrong or the channels are swapped.** Before correction you'll see some mirror leakage — that's expected.

### 2.4 Background reference, CW

30 s still-room capture. Average the spectrum. Save `cal/<date>/background_cw.npz`.

### 2.5 Manifest

`cal/<date>/manifest.json` listing the four files, each with `status: pass` or `partial`. Mark `iq` as `partial — unverified`. Static-target steps (VCO linearity, range axis, background_fmcw) listed as `blocked — no IF gain`.

`apply_calibration()` now runs DC → I/Q → spur notch → background on every load.

---

## Block 3 — `human` dataset

All captures through `apply_calibration`. 30 s each, 50 kSa/s, both channels. Filename and metadata per `klc6_capture_spec.md` §5.

| Capture | Distance | Aspect | Count |
|---|---|---|---|
| walking toward | 5 → 1 m | approach | 5 |
| walking away | 1 → 5 m | recede | 5 |
| walking across | 3 m | cross | 5 |
| standing, arms moving | 2 m | fidget | 3 |
| standing still | 2 m | static — expect nothing | 2 |

The last row is deliberate. A person standing still is invisible in CW and it's worth having the capture that proves it.

For each: `scripts/analyze.py` → spectrogram in m/s + cadence velocity diagram. Look at the approach captures with I/Q applied — torso line at ~1.5 m/s on one side of zero, limb micro-Doppler around it. A human gait has a cadence peak at step frequency, roughly 1.5–2 Hz. That's your first real CVD.

**Gate:** 20 labelled `human` captures on disk, torso line on the correct side of zero for approach vs recede, cadence peak at step rate.

---

## Block 4 — Drone at close range

The big one. Everything about geometry matters here.

### Setup

- **Radar at prop height.** A quad's props spin in a horizontal plane. Tip motion is toward and away from the radar only when the radar looks *horizontally* at the disc edge. From below looking up, tip motion is tangential and you see almost nothing. Put the module on a stand at the altitude the drone will hover.
- **Start at 1 m.** FINDINGS puts a person at 3 m at 7–15 dB over baseline. A small quad is ~100× smaller RCS but at 1 m you get 81× back from R⁴. Net roughly −1 dB from the person-at-3 m case. Detectable. Move out from there.
- **Nothing else moving in the beam.** The 80° azimuth fan is wide. Clear it.
- **Prop guards on. Someone else flying while you run captures.**

### Captures

| Capture | Distance | Aspect | Notes |
|---|---|---|---|
| hover | 1 m | hover, prop height | 5 captures, 20 s each |
| hover, throttle high | 1 m | hover | 3 captures — RPM up, blade line moves |
| hover | 2 m | hover | 3 captures |
| hover | 3 m | hover | 2 captures — may be marginal, record it anyway |
| approach | 3 → 1 m | approach | 3 captures |
| recede | 1 → 3 m | recede | 3 captures |
| props spinning, drone held stationary | 1 m | static body | 3 captures — **this isolates the blade signature from body motion** |

That last row is the important one. Hold the drone still with props spinning (carefully — prop guards, gloves). The body has zero Doppler; only the blades move. The spectrogram should show blade lines with no body line. That's the cleanest blade signature you can get, and it's the direct comparison to a bird.

### What to look for

At 24.125 GHz, Doppler is 161 Hz per m/s. A toy quad's tips run ~90 m/s → **~14 kHz**. A DJI-class ~75 m/s → **~12 kHz**. At 50 kSa/s your Nyquist is 25 kHz, so it fits.

In the spectrogram: a body line near zero (hovering) with **sidebands spread out to ±12–15 kHz**, and periodic flashes if you zoom the time axis. In the CVD: a peak at blade-pass frequency — RPM × blade count / 60. A 4-inch prop at 20,000 RPM with 2 blades is ~670 Hz. That number moves with throttle; record throttle setting in the metadata every time.

**If you see nothing at 1 m:** check the module is actually at prop height, check nothing else in the room is moving, then try 50 cm. If still nothing, log it — it's a real finding about the amp-less floor — and move on.

**Gate:** drone detected at ≥1 m. Blade sidebands visible in at least the held-stationary captures. Cadence peak present and moving with throttle.

### I/Q verification bonus

A hovering drone at fixed range with props at steady RPM is close enough to a single-tone source for an IRR check. Form `I + jQ` before and after the Block 2.3 correction. If the mirror image at −v drops by ≥15 dB, mark `iq.json` as `verified: true`.

---

## Block 5 — 1D tracker on FMCW

FINDINGS §5.5: MTI on consecutive chirps gives a walking person at 12σ in range. That's a working range measurement on movers, right now, no amp.

### Pipeline

Config B — 1 kHz PRF, 128-chirp CPI, ±3.1 m/s. Device configuration 1 for the 16k buffer.

1. Polynomial detrend per chirp, order 3 (FINDINGS §5.4 — mean removal is not enough)
2. Range FFT
3. Consecutive-chirp MTI
4. Doppler FFT → range-Doppler map
5. Zero-Doppler notch
6. OS-CFAR, 2D, guard cells 2, training cells 8, ordered statistic at 75th percentile
7. Detection list: (range, velocity, SNR) per CPI

### Tracker — `klc6/track.py`

State: `[range, range_rate]`. Constant-velocity Kalman.

- **Predict** each CPI (139 ms per FINDINGS §6)
- **Gate:** accept a detection if within 3σ of the predicted range
- **Update** with the nearest gated detection
- **Confirm** after 5 hits in 8 consecutive CPIs
- **Drop** after 5 consecutive misses
- Output per confirmed track: id, range, range_rate, age, hit_count, and kinematic features — mean |v|, v variance, hover fraction (|v| < 0.2 m/s)

Use the **measured 180 MHz** for the range axis, flagged as uncalibrated. Everything shifts once the amp lets you run VCO linearity.

### Test

Person walks from 5 m to 1 m and back. Expect one confirmed track, range decreasing then increasing, velocity sign flipping at the turn. Print the track to console; overlay on the PPI.

Then the drone at 1–2 m if Block 4 worked: hover fraction should be > 0.8.

**Gate:** walking person produces one confirmed track with correct range trend. No tracks form on an empty room.

---

## Block 6 — Write it down

Append to `docs/FINDINGS.md`:

- I/Q wired, sanity result from Block 2.3
- Spur map diff between home and lab
- Drone detection floor without amp — the distance where it dropped out
- Tracker result on the walking person

This session's numbers are the baseline everything after the amp gets compared against. Write them while they're fresh.

---

## What's deferred to the amp

| Item | Why |
|---|---|
| Static corner reflector | 336 µV LSB, invisible |
| VCO linearity | needs the reflector |
| Range axis calibration | needs linearity |
| `background_fmcw` | needs a clean static profile |
| Drone beyond ~3 m | SNR floor |
| Second module | nothing until single-module range is calibrated |

Order the amp before the session starts so it's in transit while you work.

---

## Ordering

1 → 2 → 3 → 4 → 5 → 6. If time runs short, **do 4 before 3** — the drone is the finding, the human dataset can be collected any day. But don't skip 1 and 2; every later block loads the calibration.
