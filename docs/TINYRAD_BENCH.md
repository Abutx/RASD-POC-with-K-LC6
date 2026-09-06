# TinyRad Bench Rig — Analysis of `BOM_tinyrad_bench.md`

EV-TINYRAD24G replacing the K-LC6 + AD2 chain: 2 Tx (ADF5901), 4 Rx
(ADF5904), 16-bit 4-ch ADC (ADAR7251), PLL ramp (ADF4159), Blackfin
forwarding raw ADC over USB-C.

**Reproduce every number with `python scripts/tinyrad_analysis.py --bench`.**
Model in `sim/tinyrad.py`. Every hardware number below is transcribed from
the EV-TINYRAD24G user guide **UG-1709 rev 0** or the ADF5901 / ADF5904 /
ADAR7251 datasheets. Nothing is estimated except the DJI Mini 3's RCS, which
is the same extrapolation `DEMO_TWO_RADAR.md` §3 already carries. Field-rig
analysis in `docs/TINYRAD_FIELD.md`.

---

## 1. VERDICT

**TinyRad removes the one limit that has bounded every K-LC6 measurement,
and introduces one new hard limit the BOM does not mention.**

| BOM claim | verdict |
|---|---|
| "IF amplifier not needed — own IF chain with PGA" | **survives** — ADC costs 0.15 dB at ADI's own setting vs 25.5 dB on the AD2 |
| "Range on small drone ~30–50 m" | **survives as a floor** — model says 38 m (conservative) to 113 m (textbook); ADI's own data supports the textbook end |
| "100 m range" | **survives** — ADI states it for 1 m²; the *waveform*, not the link, caps range |
| "Position from a single radar, no baseline" | **survives** — beats the two-K-LC6 rig beyond ~5 m |
| "Angle resolution ~20°" | **survives** — geometry gives 14.5°; 20° is ADI's conservative figure |
| Carry-over table | **survives**, with three additions (§7) |
| *Implicit:* blade micro-Doppler carries over unchanged | **fails** — see §4 |

The failure is the important one. **The chirp engine has a hard floor of
N × 1 µs + 20 µs per chirp** (UG-1709 p.13). Blade Doppler at hover needs
PRF ≥ 15.5 kHz, so **≤44 samples per chirp and a max range of ~13 m**; full
throttle needs ≤16 samples and ~5 m. The blade signature that was
sensitivity-limited on the K-LC6 is *waveform*-limited on TinyRad — to about
the same 4–10 m. Range and blade classification still do not coexist in one
waveform. The good news: UG-1709 says N and the chirp period are freely
configurable from the stock API, so the interleaved detect/classify scheme is
reachable without firmware work.

---

## 2. Validation — the model against ADI's own numbers

**Virtual array.** From UG-1709 Table 2 (Tx1 −18.654, Tx2 0, Rx 31.014 /
37.231 / 43.449 / 49.666 mm): Rx pitch 6.2175 mm = λ/2, Tx pitch 18.654 mm =
3λ/2. Forming Tx+Rx gives **7 distinct positions at 6.218 mm with one doubled**
— exactly the UG's "virtual, seven element array with a spacing of λ/2. Two
elements overlap." Reproduced.

**Link budget.** ADI measured a 78 mm corner cube (~1 m²) at 5 m with N=256,
1 MHz, 300 µs period, ADC gain 21 dB, and states 100 m for RCS = 1 m². Since
SNR falls as R⁻⁴, that implies **≥66 dB SNR at 5 m** for one 256 µs chirp (at
5σ). The model gives **62 dB with a 6 dB implementation loss, 43 dB with
RFbeam's 25 dB**. The 6 dB convention is within 4 dB of ADI's own claim; the
25 dB one is off by 23. **RFbeam's derating was specific to their module and
does not transfer** — the textbook convention is the supported one here.

(ADI's reference sweep was 23.95–24.25 GHz — 300 MHz, 50 MHz outside the ISM
band. All figures below use 250 MHz.)

---

## 3. Sensitivity — the AD2 problem is gone

ADAR7251 datasheet Table 1 gives input-referred noise by gain *and frequency*
— it is a 1/f profile, so the penalty depends on where the signal sits in the
IF. Referred to the ADF5904's output noise (NF 10 dB, gain 22 dB):

| PGA gain | @100 kHz (FMCW beat) | @1 kHz | @100 Hz (CW body line) |
|---|---|---|---|
| 9 dB | 1.13 dB | 2.57 dB | 8.63 dB |
| 15 dB | 0.36 dB | 0.93 dB | 4.40 dB |
| **21 dB** ← ADI's config | **0.15 dB** | 0.38 dB | 2.33 dB |
| 27 dB | 0.10 dB | 0.25 dB | 1.63 dB |
| 45 dB | 0.08 dB | 0.20 dB | 1.36 dB |

At ADI's own 21 dB the ADC costs **0.15 dB** on an FMCW beat. Even the worst
corner (9 dB gain, 100 Hz) is under 9 dB. **The 336 µV quantisation wall does
not exist on this board.** The BOM's "no IF amplifier needed" is correct; the
$32 amp in `IF_AMPLIFIER.md` is for the K-LC6 only.

Set the PGA as high as the FMCW feedthrough allows without clipping. Full
scale at 21 dB is 0.498 V p-p; at 45 dB, 31 mV p-p (ADAR7251 Table 1). The
K-LC6's feedthrough was 16.8 mV pk-pk (`FINDINGS.md` §5.4); TinyRad's will
differ and must be measured before choosing.

---

## 4. Link budget — DJI Mini 3

EIRP per Tx: 8 dBm + **12.6 dBi** (UG-1709 Table 1) = **+20.6 dBm**, 2.6 dB
above the K-LC6; four coherent receivers add 6 dB. NF 10 dB → −164 dBm/Hz.

| target | T_coh | 6 dB loss (supported) | 25 dB loss (RFbeam) | BOM |
|---|---|---|---|---|
| **Mini 3 body, moving** | 50 ms | **113 m** | 38 m | 30–50 m |
| Mini 3 body, moving | 200 ms | 160 m | 53 m | |
| **Mini 3 blades** (spread, fast-chirp) | 50 ms | **9 m** | 3 m | |
| Mini 3 blades | 200 ms | 12 m | 4 m | |
| person, 1 m² | 50 ms | 300 m | 101 m | |
| car, 10 m² | 50 ms | 534 m | 179 m | 100 m |

§2 shows ADI's own measurement supports the 6 dB column. **The BOM's 30–50 m
is therefore conservative by 2–3× — defensible as a floor, not as an
expectation.** The remaining uncertainty is the Mini 3 RCS (±5 dB, extrapolated
from a Phantom 4): one corner reflector at 4 m and one drone at 5 m settle it.

Blade rows carry the 14 dB spread penalty (775 Doppler bins in a 50 ms
fast-chirp dwell) and assume the fast-chirp mode in §5. **In a long-chirp frame
the blades alias into the clutter notch and are unavailable at any range**
(`FINDINGS.md` §8.3).

---

## 5. Waveform limits — the new hard constraint

UG-1709 p.13: *"the sampling rate is fixed to 1 MHz … The framework requires
that N × tS + 20 µs is smaller than the PWM period."* PRF is set by how many
range samples you take. Sweep 250 MHz:

| N | period | PRF | ±Doppler | ±v | max range | bins | blades? |
|---|---|---|---|---|---|---|---|
| 8 | 28 µs | 35.7 k | 17.9 kHz | 111 m/s | 2.4 m | 4 | hover + max |
| **16** | **36 µs** | **27.8 k** | **13.9 kHz** | 86 | **4.8 m** | 8 | **hover + max** |
| 24 | 44 µs | 22.7 k | 11.4 kHz | 71 | 7.2 m | 12 | hover only |
| **32** | **52 µs** | **19.2 k** | **9.6 kHz** | 60 | **9.6 m** | 16 | **hover only** |
| 48 | 68 µs | 14.7 k | 7.3 kHz | 46 | 14.4 m | 24 | aliases |
| 64 | 84 µs | 11.9 k | 6.0 kHz | 37 | 19.2 m | 32 | aliases |
| 128 | 148 µs | 6.8 k | 3.4 kHz | 21 | 38 m | 64 | aliases |
| 256 | 276 µs | 3.6 k | 1.8 kHz | 11 | 77 m | 128 | aliases |
| 512 | 532 µs | 1.9 k | 0.9 kHz | 6 | 153 m | 256 | aliases |

Mini 3 blade tips: 7 757 Hz hover, 13 742 Hz max. **Hover needs N ≤ 44 → max
range ~13 m. Full throttle needs N ≤ 16 → ~5 m.** At N=16 the 20 µs dead time
is 56% of the period — it, not the ADC rate, is the villain.

This is `SPEC.md` §10.3 with real numbers. Two things UG-1709 establishes that
make the interleaved detect/classify scheme practical:

- *"In range doppler mode, Np and N are configurable, and arbitrary uniform
  range doppler measurements are possible."* Every row above is reachable
  through the stock API (`fStart`, `fStop`, `tRampUp`, `Period`, `N`, `Seq`,
  `FrmSiz`, `FrmSizMeas`).
- N is independent of ramp duration: sample past the ramp and you get the
  flyback too; sample less and you get the first part of the ramp only.

**CW mode is never mentioned.** The ADF4159 runs as a single sawtooth
triggered by the DSP's PWM. The API exposes `fStart` and `fStop`; setting them
**equal** is the obvious experiment, and whether the PLL and frame engine
tolerate a zero-bandwidth ramp is **the first thing to try with the board.**
If it works, K-LC6-style micro-Doppler at 1 MSPS comes for free.

Under the 15.245 regulatory sweep (100 MHz — `TINYRAD_FIELD.md` §2) every
max-range figure above rises 2.5× and every range cell widens to 1.5 m.

---

## 6. Angle — what 4 Rx and 2×4 MIMO actually buy

Rx at λ/2 → unambiguous FOV ±90°, consistent with the 75° beam. The 2×4 MIMO
forms a **7-element** virtual array with one deliberate overlap (UG: *"to allow
the implementation of motion compensation algorithms"*).

| array | 3 dB beamwidth | ADI claims |
|---|---|---|
| 4 Rx only | 25.4° | |
| **2 Tx × 4 Rx = 7 virtual** | **14.5°** | 20° |

ADI's 20° is conservative against the geometry's 14.5°; the six-patch
amplitude taper and calibration residuals eat some of the gap. **Expect 15–20°
measured.**

**Resolution** (separating two targets) is the beamwidth. **Accuracy**
(locating one) is σ_θ ≈ BW / (1.6 √(2·SNR)) and sets cross-range error
σ_x = R·σ_θ:

| range | SNR | σ_θ, 4 el | σ_x, 4 el | σ_θ, 7 el | σ_x, 7 el |
|---|---|---|---|---|---|
| 5 m | 14 dB | 2.24° | 0.20 m | 1.28° | 0.11 m |
| 10 m | 14 dB | 2.24° | 0.39 m | 1.28° | 0.22 m |
| 10 m | 24 dB | 0.71° | 0.12 m | 0.40° | 0.07 m |
| **30 m** | **14 dB** | 2.24° | 1.17 m | 1.28° | **0.67 m** |
| 30 m | 24 dB | 0.71° | 0.37 m | 0.40° | 0.21 m |
| 50 m | 24 dB | 0.71° | 0.62 m | 0.40° | 0.35 m |

Down-range error is the 0.6 m cell / √(2·SNR) — a few centimetres.

**Against the two-K-LC6 baseline rig** (`DEMO_TWO_RADAR.md` §6: 0.35 m at 2 m,
0.75 m at 8 m, degrading as range²): TinyRad's cross-range error grows
*linearly* and stays under 0.5 m to ~30 m at decent SNR. **One TinyRad beats
the two-module baseline beyond ~5 m and needs no second radar, no baseline, no
carrier-offset trick.** What it adds that the baseline rig could not do at all:
separate two targets at the same range.

---

## 7. Update rate, data, compute, carry-over

Detect frame N=256 × 128 chirps → **35 ms, 28 Hz** by the chirp rule; ADI
states 50 ms → **20 Hz** in practice. Raw ADC: 4 × 16 bit × 256 × 128 =
**0.26 MB/frame → 7.4 MB/s** continuous. Range-Doppler-angle: ~10 MFLOP/frame,
**0.19 GFLOP/s at 20 Hz**.

The BOM's carry-over table is correct on every line. Three additions:

- **`sim/` carries over almost whole.** `drone.py`, `human.py`, `geometry.py`,
  `capture.py` are front-end-independent; `klc6.py` is replaced by
  `tinyrad.py`. Synthetic Mini 3 captures for the new board are a one-line
  change — build the detector before the board arrives.
- **The measured floor and mains comb do not carry over.** TinyRad has its own
  regulation (ADP5024 bucks + LT1963 LDO per its BOM) and is USB-powered; the
  60 Hz picture will differ, possibly with USB switching spurs instead.
  Re-measure day one.
- **RFbeam's 25 dB derating does not transfer** (§2). If the 4 m corner
  reflector lands more than ~6 dB below the 6 dB-loss prediction, look for a
  cable, calibration or NF problem — not a modelling one.

---

## 8. Classification on TinyRad

The layered classifier from `DEMO_TWO_RADAR.md` §9 survives and gains a layer:

| layer | signal | on TinyRad |
|---|---|---|
| 1 | gait cadence on the bulk line | any frame, full range |
| 2 | kinematics (`hover_frac`, `v_var`) | **now 2-D** — much stronger |
| 3 | RCS from range + amplitude | full range |
| 4 | blade micro-Doppler | **fast-chirp frames only, 4–10 m** |
| **5** | **angular extent** | new: a car spans 10°+ at 20 m; a drone is a point |

Blade detection at N=32, 50 ms, spread penalty applied: **8.7 m (6 dB) /
2.9 m (25 dB)**, and the waveform caps it at 9.6 m regardless. Link and chirp
floor land in the same place — 3–6× the K-LC6-with-amp figure (1.5 m), still
short of demo distances.

**The detect-vs-classify gap does not close; it moves.** K-LC6 + amp: track to
18 m, classify to 1.5 m (12×). TinyRad: track to ~113 m, classify to 9.6 m
(~12×). Both ends moved out 6–7×, the ratio held, and the classify limit
changed character — from a soft SNR limit an amplifier could push, to a
**hard waveform cap** that only a different chirp engine (or a working CW
mode) can lift. Plan the product's classification architecture around that.

---

## 9. First tasks on arrival — the BOM's list, with additions

The BOM's seven items are right and in the right order. Insert:

- **0.** Before anything else: set `fStart == fStop` and see whether the
  board produces a usable CW stream. Then confirm per-frame switching between
  a long-chirp and a short-chirp config. §5 and §8 hinge on both.
- **3′.** The BOM's "verify Python on Linux" is already answered by UG-1709:
  the Python tree ships **`usb.dll`**, a Windows library, and the board needs
  a manual Windows driver install. **Expect no.** See `TINYRAD_FIELD.md` §9.
- **5a.** After the range check: 60 s empty-room capture → spur inventory →
  compare to `out/baseline/`. New floor.
- **6a.** After the angle check: fit the measured beamwidth. Nearer 25° than
  15° means the 2×4 MIMO is not being formed — find out why.
- **7′.** "Drone at 3, 5 m, then out until it drops" **must run in both a
  fast-chirp config (blades) and the default config (body)**. They drop out
  at very different ranges. Only the body number replaces the K-LC6's 7.5 m.

---

## 10. Sources

- **UG-1709 rev 0** (2/2020), EV-TINYRAD24G User Guide — retrieved via the
  Internet Archive after ADI's servers timed out; gzip-encoded in the archive.
  Tables 1–7, pp. 4–5, 12–14.
- ADF5901 datasheet rev B; ADF5904 datasheet rev A; ADAR7251 datasheet rev 0
  Table 1 — Analog Devices
- ADI EngineerZone Q&A 589531 (22 µs; UG says 20)
- `docs/DEMO_TWO_RADAR.md`, `docs/FINDINGS.md`, `SPEC.md` §10.3
