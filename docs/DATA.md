# Captured Datasets

All raw volts. Nothing here is processed output — every reprocessing idea
downstream depends on having the original samples.

Carrier 24.125 GHz, λ = 12.427 mm, **160.95 Hz per m/s**.

| file | key | shape | fs | condition |
|---|---|---|---|---|
| `out/baseline/20260829_061237_empty_baseline_60s.npz` | `data` | (2, 2991808) | 50 k | Empty room, 59.8 s. **The reference.** 0 lost, 0 corrupted, 0 dropouts. |
| `out/cw/smoke_static.npz` | `data` | (2, 600000) | 100 k | Empty room, 6 s. Only fan-off CW reference at 100 kSa/s. |
| `out/fan/…_fan_on_cw0_100k.npz` | `data` | (2, 2500000) | 100 k | **Fan on**, CW micro-Doppler, 25 s, VCO parked at 2.5 V DC. |
| `out/fan/…_fan_on_cw1_100k.npz` | `data` | (2, 2500000) | 100 k | **Fan on**, independent second 25 s chunk — use as a cross-check. |
| `out/fan/…_fan_on_fmcw_cfgB.npz` | `cpis` | (200, 128, 100) | 100 k | **Fan on**, FMCW Config B, 1 kHz ramp, 200 CPIs over 27 s. |
| `out/fan/…_fan_on_fmcw_cfgA.npz` | `chirps` | (240, 2000) | 100 k | **Fan on**, FMCW Config A, 50 Hz ramp. |
| `out/fmcw/raw_chirps_box_out.npz` | `chirps` | (160, 2000) | 100 k | Empty room, Config A. **Fan-off FMCW reference.** |
| `out/fmcw/raw_chirps_box_in.npz` | `chirps` | (160, 2000) | 100 k | Foiled corner reflector at 3–5 m. Undetectable vs box_out (1.1σ). |
| `out/fmcw/raw_chirps_moving.npz` | `chirps` | (240, 2000) | 100 k | **Positive control** — person moving 1–2.5 m. MTI gives +6.7 dB at 1.5 m, 12σ. |
| `out/fmcw/rangeprof_*.npz` | `mag` | (1001,) | 100 k | Averaged Config A range profiles (reflector / empty). |
| `out/fmcw/vco_check.npz` | — | — | — | W1 command vs measured pin-5 voltage, and IF response. |

## Notes on using these

**Channel 2 is not a signal.** Every `data` array is shape (2, N) but only
row 0 (I, X1 pin 3) is connected to the IF. Row 1 was floating for most captures
and later clipped to the VCO pin for characterisation — it is **not** the Q
channel. X1 pin 1 (IF Q) is still unwired. Always use `data[0]`.

**The positive control matters.** `raw_chirps_moving.npz` is a known-true
detection at 12σ. Any new detector should be validated on it before a null result
elsewhere is treated as informative rather than merely uninformative.

**Unequal averaging.** The fan-on CW is 25 s; the only matching fan-off reference
at the same rate is 6 s. That asymmetry inflates the variance of any difference
spectrum. Either decimate the fan data to 50 kSa/s to use the 59.8 s baseline, or
account for the unequal block counts explicitly.

**Two independent fan-on chunks** (`cw0`, `cw1`) exist specifically so a claimed
spectral line can be checked for reproducibility across captures rather than
trusted from a single record.

See [`FINDINGS.md`](FINDINGS.md) for the measured hardware limits that constrain
what can be extracted from any of this.
