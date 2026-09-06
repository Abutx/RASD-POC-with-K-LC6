# BOM — TinyRad Bench Rig

Second-phase prototype. Replaces the K-LC6 + AD2 chain with a self-contained 4-channel MIMO radar. This is what the DSP pipeline runs on from here until the custom board.

---

## What it is

**EV-TINYRAD24G** — Analog Devices radar evaluation module, 24 GHz ISM band.

| | |
|---|---|
| Transmit | 2 antennas, ADF5901 |
| Receive | 4 antennas, ADF5904 quad downconverter, shared LO |
| ADC | ADAR7251, 16-bit, 4-channel |
| DSP | ADSP-BF706 Blackfin — forwards raw ADC to the host over USB |
| Ramp | ADF4159 PLL, linear FMCW — no VCO linearity calibration needed |
| Range | 100 m, 60 cm resolution |
| Angle | ~20° resolution from 4-channel DBF |
| Power | USB-C, from the host |
| Software | Windows GUI, MATLAB and Python reference code, raw data access |

**One unit gives range, velocity, and bearing.** Position is polar-to-Cartesian from a single radar. No second module, no trilateration, no baseline.

---

## Bill of materials

| Item | Qty | Cost | Source | Notes |
|---|---|---|---|---|
| EV-TINYRAD24G | 1 | ~$400 | DigiKey, Mouser, Arrow | In stock at DigiKey. Confirm live price. |
| Host laptop | — | own | | USB-C port. Windows for the vendor GUI; Python may run cross-platform — verify. |
| **Total** | | **~$400** | | |

**In the box:** board, USB-C cable, board mount, USB stick with drivers and software, **corner reflector.**

---

## What is NOT needed

| Item | Why |
|---|---|
| Analog Discovery 2 | TinyRad has its own 16-bit ADC |
| IF amplifier | Own IF chain with PGA on the ADAR7251 |
| External power | USB-C powered |
| Second radar for position | 4 RX channels give angle directly |
| VCO linearity calibration | PLL-locked ramp |
| Corner reflector | Ships with the kit |

---

## What carries over from the K-LC6 work

| From K-LC6 | To TinyRad |
|---|---|
| `process.py` spectrogram, CVD | Unchanged |
| OS-CFAR | Unchanged, now 2D on range-Doppler |
| Spur map, background subtraction | Redo for the new board |
| 1D Kalman tracker | Add two state dimensions: (x, y, vx, vy) |
| Classifier | Retrain on new captures — different noise floor, different IF chain |
| Dataset format, metadata schema | Unchanged, add `bearing_deg` field |

**New capability:** angle estimation. FFT across the 4 receive channels gives a spatial spectrum — power versus bearing. That step doesn't exist in the K-LC6 code and has to be written.

---

## First tasks on arrival

1. Install the vendor GUI on Windows. Confirm range-Doppler and range-angle plots on the corner reflector.
2. Run the vendor Python example. Confirm raw ADC data cube arrives: shape `(n_rx, n_chirps, n_samples)`.
3. **Verify the Python runs on Linux.** This decides the field rig's edge computer. Check ADI EngineerZone for "TinyRad Linux."
4. Port `acquire.py` to the TinyRad data path.
5. Corner reflector at 1, 2, 3, 4 m — range check. Should land within one cell with no calibration.
6. Corner reflector at −30°, 0°, +30° — angle check.
7. Drone at 3 m, then 5 m, then out until it drops. **This is the range number that replaces the K-LC6's 7.5 m.**

---

## Where TinyRad tops out

| Limit | Value | Why |
|---|---|---|
| Range on small drone | ~30–50 m estimated | Small antennas, ~10 dB NF |
| Angle resolution | ~20° | 4 channels |
| Update rate | Set by chirp config, tens of Hz | |
| Field of view | Wide — front-side antennas | |

Enough to prove the pipeline with real bearing. Not enough for the 500 m product. That's the custom board.
