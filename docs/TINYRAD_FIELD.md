# TinyRad Field Rig — Analysis of `BOM_tinyrad_field.md`

Outdoor deployment at Harmony School of Endeavor: one EV-TINYRAD24G in an
IP66 box on a pole, Pi 5 edge computer, PoE, streaming to Supabase.

**Reproduce every number with `python scripts/tinyrad_analysis.py --field`.**
Method as in `FINDINGS.md`: each claim in the BOM either survives the
arithmetic or it does not. Hardware numbers from **UG-1709 rev 0** (antenna
gain 12.6 dBi, 75°×15°, 5 V / 780 mA, Windows-only software) and the
ADF5901/ADF5904/ADAR7251 datasheets. Bench-rig analysis in
`docs/TINYRAD_BENCH.md`.

---

## 1. VERDICT

**The field BOM is a sound parts list with four engineering gaps, one of which
changes the plan.** The parts, power, network and data-path choices all
survive. What does not survive is the assumption that one unit at 2–3 m,
tilted down at a pickup lane, will also capture drones.

| BOM claim | verdict | see |
|---|---|---|
| Parts list, PoE budget, data path, compute | **survives** | §6–8 |
| 3 mm polycarbonate radome | **survives** (0.97 dB two-way) | §3 |
| "1D range plus bearing is plenty" for the lane | **survives** | §5 |
| Verify Linux before buying the edge computer | **survives** — and the answer is **no** | §9 |
| **Raspberry Pi 5 as the edge computer** | **fails** — stock Python is Windows-only (`usb.dll`) | §9 |
| **One unit serves vehicle counting AND drone capture** | **fails** — geometry | §4 |
| Regulatory basis for outdoor operation | **missing** — 100 MHz, not 250 | §2 |
| Wet-radome loss | **missing** — the largest outdoor loss | §3 |
| Enclosure thermal | **missing** — 70–80 °C inside on a Texas day | §7 |

Two things change the plan. **§4:** the vehicle-counting placement and the
drone-capture placement are physically incompatible with a single unit at
2–3 m — three ways out are given. **§9:** UG-1709 makes the edge-computer
question answerable from the desk, and the answer is the x86 path unless
someone budgets a libusb port as a real task.

---

## 2. Regulatory — 47 CFR 15.245 vs 15.249 at 24 GHz

Not in the BOM at all, and it moves a hardware number.

| rule | band | limit @ 3 m | as EIRP | TinyRad (est. +21 dBm) |
|---|---|---|---|---|
| 15.249 general | 24.0–24.25 GHz | 250 mV/m | **+12.7 dBm** | **over by 8 dB** |
| 15.245 field-disturbance sensor | **24.075–24.175 GHz** | 2500 mV/m | +32.7 dBm | under by 12 dB |

TinyRad's +8 dBm PA into its 12.6 dBi antenna (UG-1709 Table 1) is
**+20.6 dBm EIRP**. Under the general rule that is 8 dB over. Under the field-disturbance-sensor rule it is fine —
**but that rule allows only 24.075–24.175 GHz, a 100 MHz window, not the
250 MHz ISM band TinyRad sweeps by default.**

**Consequence:** operating as a 15.245 sensor means a 100 MHz sweep and
**range resolution of 1.5 m, not 0.6 m.** Every waveform table in
`TINYRAD_BENCH.md` §3 built on 250 MHz shifts by 2.5× in cell size. (Max
range for a given chirp length goes *up* 2.5×, which helps the lane job.)

This is an evaluation board with no FCC ID of its own. Indoor bench use is the
normal engineering case. A fixed installation radiating over a school pickup
lane is not, and someone should decide deliberately before install day. This
is an engineering reading of the rules, not legal advice.

**Add to the BOM:** a line item — 15.245 at 100 MHz, or seek authorisation.
Do not default to the full sweep outdoors.

---

## 3. Radome — polycarbonate is right; water is the problem

One-way loss of a flat sheet at 24 GHz, normal incidence (Fabry-Pérot slab +
dielectric absorption). Half-wave thickness in polycarbonate (ε_r 2.75) is
**3.75 mm** — the reflectionless thickness.

| material | thickness | one-way | two-way |
|---|---|---|---|
| polycarbonate | 2.0 mm | 1.11 dB | 2.23 dB |
| **polycarbonate (BOM)** | **3.0 mm** | **0.48 dB** | **0.97 dB** |
| polycarbonate | 3.75 mm | 0.11 dB | 0.22 dB |
| acrylic (PMMA) | 3.0 mm | 0.63 dB | 1.26 dB |
| window glass | 3.0 mm | 1.49 dB | 2.97 dB |
| **ABS — the box wall** | 2.5 mm | 0.93 dB | 1.85 dB |

3 mm polycarbonate costs **0.97 dB two-way** — a good choice, and chasing the
3.75 mm optimum saves 0.75 dB, not worth it. The BOM's "not acrylic" is
over-cautious (3 mm PMMA is 1.26 dB); "not glass" is right. Note the ABS box
wall itself is 1.85 dB — the window must be a real cutout, not "shoot through
the lid."

**What the BOM misses: water.** A 0.5 mm film of rain on the window costs
several dB at 24 GHz. A wet radome is the single largest weather loss outdoors
— far above rain in the path, which is negligible:

| rain rate | specific | two-way at 50 m |
|---|---|---|
| 5 mm/hr | 0.68 dB/km | 0.07 dB |
| 25 mm/hr | 3.77 dB/km | 0.38 dB |
| 50 mm/hr | 7.87 dB/km | 0.79 dB |

**Add to the BOM:** tilt the window ≥15° from vertical so water sheets off;
extend the lid as a drip shield; consider a hydrophobic coating. Keep the
6 mm antenna-to-window standoff the BOM already specifies.

---

## 4. Site geometry — the finding that changes the plan

The BOM puts the radar at 2–3 m, tilted slightly *down* at the pickup lane,
and says drone captures "happen wherever they fly — coordinate a schedule."

TinyRad's elevation beam is **15°**. That is a thin horizontal sheet, exactly
the situation `DEMO_TWO_RADAR.md` §6 described for the K-LC6:

| horizontal range | sheet height | drone must be within |
|---|---|---|
| 10 m | 2.6 m | ±1.3 m of radar height |
| 20 m | 5.3 m | ±2.6 m |
| 30 m | 7.9 m | ±3.9 m |
| 50 m | 13.2 m | ±6.6 m |

**A drone at 15–30 m altitude is outside the elevation beam entirely from a
2.5 m mount tilted down.** Tilt up to catch it and two things happen: the
pickup lane leaves the beam, and you are now looking at the rotor disc from
*below* — the cos(elevation) null quantified in `DEMO_TWO_RADAR.md` §4. From
25 m altitude at 30 m horizontal range the radar sees the drone at **40°
elevation**; the simulation put blade energy above 4 kHz at 55% of the
level-view value at 30° and 0.3% at 60°. At 40° you are on the steep part of
that curve.

**The vehicle-counting placement and the drone-capture placement are
incompatible with one unit.** Options:

| | placement | vehicles | drones | cost |
|---|---|---|---|---|
| **(a)** | two units, one per job | ✅ | ✅ level-view | +$400 |
| **(b)** | one unit at 2.5 m; drones fly *in the sheet* | ✅ | ⚠️ low, level, lane empty | $0 |
| **(c)** | one unit **high (roof edge, 8–10 m)** tilted down | ✅ | ✅ drones crossing at 5–15 m pass through the sheet with the radar roughly level | pole → roof mount |

(b) gives useful *dataset* captures but is not the product's overhead-drone
case. **(c) is the only single-unit geometry that serves both**, and the BOM's
pole-mount bracket does not reach it. Decide before ordering mounting parts.

---

## 5. Vehicle counting — the job this placement is good for

| range | angle cell (20°) | range cell, 15.245 | range cell, full ISM |
|---|---|---|---|
| 20 m | 7.0 m | 1.5 m | 0.6 m |
| 30 m | 10.5 m | 1.5 m | 0.6 m |
| 50 m | 17.5 m | 1.5 m | 0.6 m |

Bearing cannot separate two cars side by side at 30 m — the angle cell is
10 m wide. **Range does the work.** A single-file pickup lane is a 1D problem
and 0.6–1.5 m of range resolution resolves 4–5 m cars with margin. Car RCS
≥10 m² detects at hundreds of metres by SNR — the waveform's max range is the
limit, not the link. At 20 Hz and 5 m/s a car moves 0.25 m per frame; the
tracker gets 20+ hits per car.

**The BOM's "1D range plus bearing is plenty" is right.**

**New confuser outdoors: birds.** K-band bird RCS is −20 to −30 dBsm —
comparable to a Mini 3. The separator is cadence: wingbeat 2–10 Hz vs
blade-pass 200+ Hz, two orders of magnitude apart. Seeing it needs the
fast-chirp frame from `TINYRAD_BENCH.md` §3. Pedestrians are handled by the
gait layer already designed.

---

## 6. Power and PoE — survives

| item | power | source |
|---|---|---|
| **TinyRad, all 4 Rx enabled** | **3.9 W** (5 V × 780 mA) | UG-1709 Table 3 |
| Pi 5 under load (or ~15 W for an x86 mini-PC, §9) | ~6 W | |
| **total** | **~10 W** (Pi) / ~19 W (x86) | |
| 802.3at at the PD | 25.5 W | |

**~15 W of margin with a Pi, ~6 W with x86.** The BOM's "under 15 W,
comfortable" is right for the Pi path and tight for the x86 path — if §9 sends
you to x86, re-check the PoE budget and pick an 802.3bt injector.

**USB current:** 780 mA exceeds a USB 2.0 port's 500 mA. A Pi 5 shares 1.6 A
across its USB ports; a mini-PC's USB-C port will be fine. Do not hang
anything else on the same bus.

---

## 7. Enclosure thermal — missing, and it will bite in Texas

~10 W dissipated inside a *sealed* IP66 box (more on the x86 path). A
250×200×100 mm enclosure has ~0.19 m² of surface; at ~5 W/m²/K natural
convection that is a **~10 K rise over ambient before solar gain**, which on a
dark box adds 15–25 K more. A Pi 5 throttles at 85 °C. **On a 40 °C day the
inside reaches 70–80 °C.**

Desiccant handles humidity, not heat. **Add to the BOM:** light-coloured or
white enclosure; a sun shield / second skin; a Pi 5 heatsink or the official
active cooler; **CPU temperature in the one-minute heartbeat** so you see
throttling before it becomes a mystery outage. TinyRad itself carries an
**AD7415 temperature sensor on its PCB** (UG-1709 BOM, U12) — read that too,
it is the radar's own die-adjacent temperature.

---

## 8. Data path and compute — survives

Raw ADC at a plausible detect-frame config (N=256, 128 chirps, 4 ch, 16-bit)
is **0.26 MB/frame → ~7 MB/s** at max frame rate. A school uplink will not
take that; the BOM's "push detections, buffer raw locally" is exactly right.

Range-Doppler-angle per frame is ~10 MFLOP → **0.19 GFLOP/s at 20 Hz**. A
Pi 5 does ~30 GFLOP/s in numpy. An order of magnitude of headroom, even
before the classifier.

The BOM's data-path diagram, 24 h rolling buffer, heartbeat and outbound-only
posture are all sound. The "what IT will want to know" paragraph is the right
instinct — write it before you ask.

---

## 9. The Linux question — answerable from the user guide, and the answer is no

UG-1709 p.1, *Equipment Needed:* **"PC with Windows 7 (or more recent
version)."** p.13, Python contents: *"The DLL directory contains the
`usb.dll` file."* The MATLAB path uses a "USB mex driver" from the older
DemoRad kit. The board enumerates as **"BF707 Bulk Device"** and requires a
manual Windows driver install from `Demo_Driver.zip` (pp. 6–7).

**The stock Python library is Windows-only as shipped.** It calls into a
Windows DLL for USB transport. There is no Linux path in the box.

| path | works day one | cost vs Pi 5 | power | notes |
|---|---|---|---|---|
| **(a) x86 mini-PC, Windows** | **yes** | +$100–120 | +~10 W | 12 V PoE splitter; re-check PoE budget (§6) |
| (b) libusb/pyusb port of `usb.dll` | after dev work | $0 | — | device is a plain bulk endpoint; protocol is whatever the Python class sends; a few days if simple, longer if not. Check EngineerZone for prior art first. |

**The BOM's sequencing is right — do not order field compute until the bench
rig has settled this — but go in expecting (a).** Treat (b) as an upgrade
project with its own line item, not a checkbox in the first-tasks list.

---

## 10. Revised field BOM deltas

| add / change | why | ~cost |
|---|---|---|
| **Edge computer: plan x86 mini-PC + Windows; Pi 5 only if a libusb port is funded** | §9 | +$100–120, 802.3bt injector |
| Regulatory decision: 15.245 (100 MHz) or authorisation | §2 | — |
| Decide placement (a)/(b)/(c) **before** ordering the mount | §4 | (c): roof bracket instead of pole |
| Tilted window, drip shield, hydrophobic coating | §3 | +$15 |
| White/light enclosure or sun shield | §7 | +$0–20 |
| Active cooler for the edge computer | §7 | +$5–15 |
| CPU temp **and TinyRad AD7415 temp** in heartbeat | §7 | — |
| Corner reflector check **at the 15.245 sweep** if that is the mode | §2 | — |

Everything else in the BOM stands.

---

## 11. Sources

- **UG-1709 rev 0** (2/2020), EV-TINYRAD24G User Guide — Table 1 (antenna),
  Table 3 (electrical), pp. 1, 6–7, 13 (software, drivers, `usb.dll`), BOM
  Table 8 (AD7415, regulators). Retrieved via the Internet Archive.
- ADF5901 datasheet rev B; ADF5904 datasheet rev A; ADAR7251 datasheet rev 0 —
  Analog Devices
- ADI EngineerZone Q&A 589531, chirp period ≥ N × 1 µs + 22 µs
- ADI EV-TINYRAD24G product page: 75°×15° beam, ~20° MIMO angle resolution,
  60 cm / 100 m
- 47 CFR 15.245, 15.249 (eCFR)
- ITU-R P.838 rain attenuation coefficients (24 GHz)
- `docs/DEMO_TWO_RADAR.md` §4 (elevation null), §6 (thin-sheet coverage)
- `docs/TINYRAD_BENCH.md` §3 (waveform limits)
