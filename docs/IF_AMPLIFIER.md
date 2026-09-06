# IF Amplifier — Specification and What to Buy

The single change that moves every range in this project by a factor of four.
`FINDINGS.md` open item 3 has called it the wall since day one;
`docs/DEMO_TWO_RADAR.md` quantifies it at **25.5 dB** of lost sensitivity.

Derived, not guessed — regenerate with the calculations in
`scripts/demo_budget.py`. **Read §1 before buying anything**, because the
requirement that eliminates most candidate parts is not the one people expect.

---

## 1. The specification

| # | parameter | requirement | why |
|---|---|---|---|
| 1 | **Gain** | **30 dB** (×31.6), ideally adjustable 20–40 | 30 dB recovers 24.2 of the 25.5 dB lost. 40 dB adds only 1.1 dB more. |
| 2 | **Input-referred noise** | **≤ 20 nV/√Hz** | The K-LC6's own IF noise is 45 nV/√Hz. 20 nV/√Hz costs 0.8 dB; 15 costs 0.5 dB. |
| 3 | **Coupling** | **AC, corner ≤ 10 Hz** | **Mandatory.** See below. |
| 4 | **Bandwidth** | flat to **25 kHz** min, 50 kHz preferred | Mini 3 max-throttle blade tips are 13 742 Hz; Nyquist at 50 kSa/s is 25 kHz. |
| 5 | **Input impedance** | **≥ 10 kΩ** | The IF output is a 50 Ω source. A 50 Ω-terminated input halves the voltage — throws away 6 of the 30 dB. |
| 6 | **Output swing** | ±2.5 V is enough | AD2 full scale. Nothing but a person inside 0.5 m gets near it. |
| 7 | **Channels** | **2 minimum, 4 to be safe** | See below. |

**On channel count — 2 is enough for the demo as specified.** One AD2 has
exactly two analog inputs, so the two-radar positioning demo digitises two
channels (one IF I per module) and no more. You only need 4 amplifier channels
if you add I/Q (which requires a second AD2) or the audio-interface path in
Option C. A dual op-amp gives 2 channels per chip, so buying 4 costs almost
nothing extra in Option B and doubles the price in Option A.

### Requirement 2 is a LOW bar — do not overspend on it

The module's own IF noise is 45 nV/√Hz, and the amplifier's noise adds in
quadrature with that:

| amp noise | total | degradation |
|---|---|---|
| 4 nV/√Hz | 45.2 | 0.03 dB |
| 10 nV/√Hz | 46.1 | 0.21 dB |
| 20 nV/√Hz | 49.2 | 0.78 dB |
| 45 nV/√Hz | 63.6 | 3.01 dB |

**An ordinary audio op-amp (1–8 nV/√Hz) is already 20× better than needed.**
Ultra-low-noise lab preamps are solving a problem we do not have. Spend the
money on channel count instead.

### Requirement 3 is the one that kills candidates

The K-LC6 IF output sits on a DC offset of up to **±0.2 V** (datasheet p.2),
and in FMCW it *moves* with VCO voltage through self-mixing:

| gain | 0.2 V offset becomes | AD2 range |
|---|---|---|
| 20 dB | 2.0 V | ±2.5 V |
| **30 dB** | **6.3 V** | ±2.5 V — **saturated** |
| 40 dB | 20.0 V | ±2.5 V — **saturated** |

A DC-coupled amplifier at 30 dB slams into the rail before it sees a single
target. **AC coupling is not optional.** Corner ≤ 10 Hz keeps everything the
existing `process.preprocess(hp_hz=20)` already passes.

### Do NOT buy the K-LC6_V2 as the fix

It is the same module with a 20 dB IF amp built in, which sounds like exactly
what we want. Its IF bandwidth is **10 Hz – 15 kHz**. A Mini 3 at full throttle
puts blade tips at **13.7 kHz** — on the shoulder — and anything faster is
attenuated. It also gives 20 dB, not 30, and it is not upgradable. Wrong buy.

---

## 2a. Zero-build option — Behringer XENYX 802 mixer (~$50–60 new)

For the POC the priority is the DSP pipeline, not hardware, so this is the
recommended buy: a small analog mixer whose two mono **line inputs** are
exactly the amplifier §1 specifies, with nothing to solder or modify. Specs
from the XENYX 502/802/1002/1202 manual, §4:

| §1 requirement | XENYX 802 mono channel, LINE input | |
|---|---|---|
| Gain 30 dB, adjustable | GAIN knob **−10 dB to +40 dB** | ✅ |
| Input noise ≤ 20 nV/√Hz | E.I.N. −129 dB @150 Ω (≈2 nV/√Hz) | ✅ |
| AC coupled, ≤ 10 Hz | response **<10 Hz – 150 kHz** (−1 dB) | ✅ |
| Bandwidth ≥ 25 kHz | 150 kHz | ✅ |
| Input Z ≥ 10 kΩ | **≈10 kΩ unbalanced** (20 kΩ bal.) | ✅ |
| Output ±2.5 V | main out to +22 dBu — plenty; back the gain off if `pk > 2.7 V` | ✅ |
| 2 channels | two mono channels (1 and 2) | ✅ |

Hook-up (radar → mixer → AD2), no changes to the radar wiring:

1. K-LC6 A IF I (X1 pin 3) + GND → **channel 1 LINE IN** (¼" TS, tip = IF I).
   K-LC6 B → **channel 2 LINE IN**. Use ¼" screw-terminal plugs, no soldering.
2. Channel 1 **PAN hard left**, channel 2 **PAN hard right**. EQ knobs
   centred (flat), channel faders and MAIN MIX at 0 dB, FX/aux at minimum.
   Phantom power **OFF** (it only reaches the XLR jacks, but leave it off).
3. **MAIN OUT L → AD2 1+**, **MAIN OUT R → AD2 2+**, sleeves to AD2 ground
   (¼" TS to bare-wire or ¼" → BNC adapters onto the AD2 BNC board).
4. Set both GAIN knobs to the same position around **+30 dB** (three-quarter
   turn). Then **measure** the actual gain with the AD2's W2 into the line
   input (100 mV, 1 kHz) — the knob is not calibrated — and put the measured
   number into `if_gain_db`. Re-check it whenever the knob is touched.

Not an audio interface (it has no ADC), so the AD2 still samples and the
FMCW ramp-synchronous triggering is unchanged. The one thing to watch: it is
mains-powered, so the mixer's ground meets the AD2's USB ground — if 60 Hz
rises relative to the floor after installing it, run the AD2 laptop on
battery for the capture or move the mixer's wall adapter to the same outlet
strip.

Buy the **802** (two mono channels), not the 502 (one). Links:
<https://www.amazon.com/Behringer-802-Premium-8-Input-Preamps/dp/B000J5XS3C>,
<https://www.guitarcenter.com/Behringer/XENYX-802-Mixer-1275776902518.gc>.
The 802S (USB) also works but costs more for a USB port you will not use.

## 2. What to buy — the $60 build (if you would rather solder)

**Budget: $60. This build is ~$30 and beats every commercial option on the two
specs that matter here** (exactly-known gain, and a built-in anti-alias
roll-off). The commercial units in §2b are listed for completeness; none of
them fit $60 and none of them do this job better.

### Power it from the AD2 — no PSU needed

`TODAY.md` lists **AD2 V+, V−** as unused. They are programmable supplies,
+0.5 to +5 V and −0.5 to −5 V. Two NE5532s draw ~16 mA total against a budget
of ~250 mA on USB power. **That is the whole power supply problem solved for
free**, and `dwfpy` can enable the rails in the same script that opens the
device.

### Circuit, per channel (one half of a dual op-amp)

```
K-LC6 X1 pin 3 (IF I) ──────┬──── + IN  ┐
                            │           │   1/2 NE5532P
                     (the 50 Ω source   │
                      provides the DC   │
                      bias path — no    │
                      resistor needed)  │
                                        │
             ┌──── R_f 30.1k 1% ────────┤ ── OUT ──> AD2 1+
             │                          │
             │      ┌─ C_f 100 pF ─┐    │
             └──────┴──────────────┴────┤
                                        │
                         − IN ──────────┘
                            │
                       R_g 1.00k 1%
                            │
                       C_g 22 µF BIPOLAR
                            │
                           GND (star point)
```

| | measured value | requirement |
|---|---|---|
| Gain | 1 + 30.1k/1.00k = 31.1 = **29.9 dB** | 30 dB ✅ |
| Input noise | NE5532: **5 nV/√Hz** | ≤20 ✅ (9× margin) |
| High-pass | 1/(2π·1k·22µ) = **7.2 Hz** | ≤10 Hz ✅ |
| Low-pass | 1/(2π·30.1k·100p) = **52.9 kHz** | ≥25 kHz ✅ |
| Input Z | ~10⁵ Ω (op-amp + input) | ≥10 kΩ ✅ |
| Output swing | ±3.5 V on ±5 V rails | ±2.5 V ✅ |

**Why the coupling cap is in the feedback leg, not the input.** At DC, C_g is
an open circuit, so R_g is disconnected and the stage is a unity-gain buffer.
The module's ±0.2 V offset therefore passes through at **0.2 V, not 6.3 V** —
no saturation — while everything above 7.2 Hz gets the full 31×. It also means
no input coupling capacitor sitting across the 50 Ω source.

**C_g must be BIPOLAR (non-polarised).** Both sides of it sit at ~13 mV DC, so
a polarised electrolytic runs at essentially zero bias and will distort.

### BOM — 4 channels

| qty | part | example P/N | ~unit | ~total |
|---|---|---|---|---|
| 2 | Dual op-amp, DIP-8 | **NE5532P** (TI) | $1.20 | $2.40 |
| 4 | Resistor 30.1 kΩ 1% metal film | any | $0.10 | $0.40 |
| 4 | Resistor 1.00 kΩ 1% metal film | any | $0.10 | $0.40 |
| 4 | Cap 22 µF **bipolar** electrolytic 16 V | Nichicon **UES1C220MEM** | $0.60 | $2.40 |
| 4 | Cap 100 pF C0G ceramic | any | $0.15 | $0.60 |
| 4 | Cap 100 nF ceramic (rail decoupling) | any | $0.10 | $0.40 |
| 2 | Cap 10 µF ceramic/tant (bulk) | any | $0.30 | $0.60 |
| 2 | 8-pin DIP socket | any | $0.25 | $0.50 |
| 1 | Perfboard / prototyping board | any | $5.00 | $5.00 |
| 1 | Screw terminals / headers | any | $3.00 | $3.00 |
| 2 m | Shielded cable for the inputs | any thin coax | $4.00 | $8.00 |
| 1 | Metal enclosure (or an Altoids tin) | — | $8.00 | $8.00 |
| | | | **total** | **≈ $32** |

Digi-Key or Mouser will have all of it in one order. **You have room in the
budget for the better op-amp if you want it:**

| op-amp | noise | DIP-8 | ~price | degradation vs ideal |
|---|---|---|---|---|
| **NE5532P** | 5 nV/√Hz | yes | $1.20 | 0.05 dB |
| LM4562NA | 2.7 nV/√Hz | yes | $3.50 | 0.02 dB |
| OPA2134PA | 8 nV/√Hz (JFET) | yes | $5.00 | 0.14 dB |

**All three are indistinguishable in this application** — the module's own
45 nV/√Hz swamps every one of them. Buy the NE5532 and spend the difference on
a proper enclosure, which will matter far more.

### Build notes that actually affect the result

At 30 dB, **layout beats part selection**. `FINDINGS §3` measured 60 Hz mains
at +35.9 dB over the floor with *no* gain in the chain; a sloppy input lead at
×31 will make that much worse and you will have amplified your worst enemy.

1. Mount the amp **within a few cm** of the K-LC6 — short, shielded input lead.
2. Keep the **star ground** from `TODAY.md`. Cable shield to the star point at
   one end only.
3. Put it **in a metal box**, bonded to the star ground.
4. Keep the amp away from the AD2's switching supplies and any USB cable run.
5. **Measure the actual gain** with a signal generator before the first
   capture, and record that number — not 30 — in `if_gain_db`.

---

## 2b. Commercial alternatives (all over budget, listed for completeness)

### Option A — Koheron AMP200-10k ×4 · ~€275 each · lowest risk

<https://www.koheron.com/photonics/amp200-amplifier/>

| spec | value | vs requirement |
|---|---|---|
| Gain | 5–100, adjustable | ✅ set to 31.6 |
| Input noise | 2.4 nV/√Hz | ✅ 20× margin |
| Coupling | AC/DC selectable, **0.72 Hz** corner | ✅ |
| Bandwidth | 13 MHz | ✅ (more than needed — see caveat) |
| Input Z | **10 kΩ** | ✅ |
| Output | ±8.5 V hi-Z | ✅ |
| Supply | **±12 V**, 100 mA/rail | ⚠️ extra PSU needed |

Buy the **-10k** variant, not the -1M: better noise (2.4 vs 4.7 nV/√Hz) and
10 kΩ still loads a 50 Ω source by nothing.

**Caveat:** 13 MHz of bandwidth in front of a 50 kSa/s sampler is an aliasing
risk — broadband noise from 25 kHz to 13 MHz folds into the band. Use the AD2's
`filter="average"` mode (already the default in `acquire.configure`) and
consider a simple RC low-pass at ~30 kHz on the output.

**Cost:** ~€1100 for 4 channels, plus a ±12 V supply.

### Option B — 24-bit audio interface · ~$270 · fixes the ADC too

**MOTU M4** (~$270, −129 dBu EIN) or **Focusrite Scarlett 4i4 4th gen**
(~$280, −127 dBu EIN). 4 channels, 24-bit/192 kHz.

This is a different and in some ways better idea: it replaces the amplifier
**and** the 14-bit ADC that caused the problem in the first place.

- Mic-preamp EIN of −129 dBu ≈ **1.9 nV/√Hz** ✅ negligible
- Gain adjustable to +50 dB ✅
- AC coupled ~10–20 Hz ✅
- 24-bit at ±1 V → LSB ~0.1 µV vs the AD2's **336 µV** — the quantisation wall
  simply disappears
- **The entire Mini 3 blade signature (≤13.7 kHz) fits inside ordinary
  20 kHz audio bandwidth.** This is the fact that makes the idea work.

**Two hard limits:**

1. ⚠️ **PHANTOM POWER MUST BE OFF.** 48 V onto the K-LC6's IF pin will destroy
   the module. Check before every connection.
2. **It cannot drive the FMCW ramp.** Chirp-synchronous triggering needs the
   ADC and the ramp DAC on one clock (`FINDINGS §5.2`: correlation +0.989
   triggered vs −0.098 free-running). So this covers the **CW / micro-Doppler**
   path only; FMCW positioning stays on the AD2 and still needs Option A or B.

### Recommendation

**Build §2 now** — ~$32, two hours, exactly-known gain set by two resistors.
That last point is not cosmetic: `if_gain_db` goes into every sidecar and
`dataset/README.md` rule 2 says clips at different gain are not comparable
without it. A resistor ratio is more trustworthy than a front-panel knob.

The commercial options only become interesting later: the audio interface if
you want the 14-bit wall gone too, and the Koheron only if nobody has time to
solder — it is 30× the cost for a noise specification the physics says you
cannot use.

---

## 3. After it is installed

1. **Re-measure the floor.** Run a 60 s empty-room capture and compare against
   `out/baseline/20260829_061237_empty_baseline_60s.npz`. The broadband floor
   at 7–8 kHz should drop from 845 nV/√Hz toward the module's 45 nV/√Hz.
   `sim/klc6.py:measured_floor_v_rthz()` has the reference curve; update it.
2. **Record the real gain**, measured, into `if_gain_db` on every capture. Not
   the nominal — measure it with a signal generator.
3. **Re-run `python scripts/demo_budget.py`** with the new floor. Every range
   in `DEMO_TWO_RADAR.md` should move by ~4×.
4. **Watch for clipping** near targets: a person inside 0.5 m will saturate at
   30 dB. `scripts/session.py` already flags `pk > 2.7 V` as CLIPPING.
5. **Check the mains comb did not get worse.** Gain amplifies pickup as
   readily as signal; if 60 Hz rises relative to the floor, the input lead or
   the ground is the problem, not the amplifier.
