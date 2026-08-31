# K-LC6 24 GHz Radar — Capture, Processing, Live Scopes

RFbeam **K-LC6-RFB-00D** (24.125 GHz) driven by a Digilent **Analog Discovery 2**.
Implements `SPEC.md`. Everything measured on the bench is in
**[`docs/FINDINGS.md`](docs/FINDINGS.md)** — read that before trusting any number
here, including the ones in the spec.

```
klc6/          acquire.py   device open, configure, record, FMCW chirps
               process.py   preprocess, spectrogram, cadence, range-Doppler
scripts/       one job each, see table below
docs/          FINDINGS.md  every measured result and every trap hit
out/           captured data and figures, by topic
```

## Physical constants

```
lambda        = 12.427 mm at 24.125 GHz
Doppler       = 161.0 Hz per m/s
range res     = c / 2B  ->  0.83 m at the MEASURED 180 MHz sweep
                            (0.50 m at the datasheet's 300 MHz)
```

## Scripts

| script | what it does |
|---|---|
| `doctor.py` | **Environment + hardware check. Run first on any new machine.** |
| `smoke_test.py` | Proves the chain streams. Level, dominant Doppler, spectrogram. |
| `baseline.py` | Long empty-room capture + interference inventory in Hz and m/s. |
| `analyze.py` | Load an `.npz` → inventory, spectrogram, cadence diagram. `--repair` trims legacy leading zeros. |
| `live.py` | **Live CW motion detector.** Big banner, rolling spectrogram, metric trace. Tk/PIL, ~11 fps. |
| `vco_check.py` | VCO characterisation (SPEC §10.1). `--probe` needs CH2 on X1 pin 5. |
| `chirp_check.py` | Chirp-sync acceptance (§10.9.1). Tests within *and* across acquisitions. |
| `range_demo.py` | FMCW Config A range profile, with background subtraction and averaging. |
| `ppi.py` | **Live PPI scope.** Range arcs, Doppler colour, waterfall, zero-Doppler notched. |
| `collect_fan.py` | Multi-modal dataset capture (CW micro-Doppler + FMCW Config A and B). |
| `fan_matched_test.py` | Duration-matched fan-on vs empty test with a measured same-condition floor. |
| `fan_ab_test.py` | **Same-session fan ON/OFF/ON A/B.** Run this first — removes the session confound. |

## Setup on a new machine

**Two things are NOT pip-installable and must come first:**

1. **Digilent WaveForms** — https://digilent.com/reference/software/waveforms/waveforms-3/start
   This ships the SDK and `dwf.dll` (`libdwf` on Linux/macOS). `dwfpy` is only a
   wrapper; without WaveForms installed it fails at import. **Close the WaveForms
   application before running anything** — it holds the device exclusively.
2. **tkinter** — bundled with python.org builds. On Debian/Ubuntu:
   `sudo apt install python3-tk`

Then:

```bash
pip install -r requirements.txt
python scripts/doctor.py          # <-- run this FIRST
```

`doctor.py` checks packages, the dwf runtime, device enumeration, device
configuration 1 (the 16,384-sample buffer FMCW needs), and takes a real 0.5 s
capture to confirm the K-LC6 is powered and wired to Channel 1. It prints the
specific fix for whatever fails instead of a stack trace.

### Wiring

| | |
|---|---|
| Power | external 5 V into K-LC6 **X1 pin 2** |
| CH1 (orange 1+) | **X1 pin 3** — IF I |
| W1 (yellow) | **X1 pin 5** — VCO in (FMCW only) |
| Ground | X1 pin 4 + AD2 ⏚ + 1− + 2− + supply return, star-tied |

Expected healthy CH1 reading with the module powered: **~150–180 µV rms**, mean
around +13 mV, 4–7 distinct ADC codes. A flat channel (<30 µV) means no power or
a wiring fault — `doctor.py` says so explicitly.

## Quick start

```bash
python scripts/doctor.py                # environment + hardware check
python scripts/smoke_test.py            # is the chain alive?
python scripts/live.py --threshold 5    # CW motion detector
python scripts/ppi.py --rmax 10         # FMCW radar scope
```

## Things that will bite you

These are all measured, with the numbers in `docs/FINDINGS.md`:

1. **Never pass a progress callback to `recorder.record()`.** It silently returns
   an unfilled buffer while reporting 0 lost and 0 corrupted.
2. **`range_v` only offers 5 or 50 Vpp.** LSB is 336 µV; the bare IF sits on 4–7
   ADC codes. An IF amp is required for static-target range.
3. **Notch 60 Hz harmonics before any detection decision.** The empty room puts
   mains at 0.373, 1.119, 1.491 and 1.864 m/s — dead in the walking band.
4. **100 kSa/s only sustains ~30 s.** Use 50 kSa/s for long captures.
5. **Static targets are not detectable in range; moving ones are.** MTI /
   zero-Doppler notching is worth about 30 dB.
6. **There is no bearing information.** One Tx, one Rx. Targets are arcs, not dots,
   until beamforming hardware exists.
7. **A fan pointed AT the radar is Doppler-invisible by geometry.** Its blades then
   sweep across the line of sight, so radial velocity is ~zero for every blade
   element. Turn it side-on. See `docs/FAN_DETECTION.md`.

## Fan detection study (2026-08-30)

8 independent DSP methods, adversarially verified: **not detected**, ~0.9
confidence, bounded ≥6.4–11.7 dB below a walking person at the same 2.5 m range.
Every >5σ candidate resolved to 60 Hz mains. Full write-up in
[`docs/FAN_DETECTION.md`](docs/FAN_DETECTION.md); per-method scripts, figures and
logs in `out/fan/analysis/`.
