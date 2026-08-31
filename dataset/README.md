# Dataset layer

Storage-for-the-model: every capture is a raw file plus a JSON sidecar, indexed
into one SQLite manifest. Works identically for bench captures, field drone
sessions, and Sionna/synthetic simulation output — the manifest doesn't care
where samples came from, only that they're labeled.

```
<capture>.npz|.wav      raw samples (sacred — never overwrite)
<capture>.json          sidecar: label + conditions (source of truth)
dataset/manifest.sqlite index built FROM sidecars (disposable, rebuild anytime)
```

## Sidecar schema

```json
{
  "label": "fan_on",            // class: empty | human_moving | fan_on | fan_foiled |
                                //        reflector_static | drone_<model> | bird_<species> | unlabeled
                                // non-training: derived (processed output) |
                                //               bench_check (uncontrolled aliveness capture)
  "source": "bench",            // bench | field | sionna | synth
  "mode": "cw",                 // cw | fmcw_chirps | fmcw_cpis   (inferred from npz keys if absent)
  "session": "20260830_bench",  // capture session — REQUIRED, splits happen on this
  "target": "household fan, plastic blades",
  "distance_m": 2.5,
  "aspect": "axis-on",
  "if_gain_db": 0,              // 0 = bare IF straight into the ADC
  "notes": ""
}
```

Extra keys are preserved (Sionna runs: put sim params here).

## Rules

1. **Label at capture time.** An unlabeled clip is a chore; a mislabeled one is poison.
2. **Record `if_gain_db` always** — clips at different gain are not comparable without it.
3. **Train/val/test split by `session`, never by clip.** Windows from one capture
   are near-duplicates; splitting them across sets fabricates accuracy.
4. Raw files stay out of git once they outgrow it — sync `out/` to a bucket
   (Cloudflare R2 / HF dataset), sidecars and manifest code stay in git.

## Use

```bash
python -m dataset.manifest build out/ ../klc6/data/   # scan -> manifest.sqlite
python -m dataset.manifest stats                      # what do we have?
```

```python
from dataset import loader
rows = loader.query(label="fan_on", mode="cw")
train, test = loader.session_split(rows, test_frac=0.3)
for spec, lab in loader.spectrogram_windows(train, win_s=2.0):
    ...  # (freq, time) float32 log-mag + label
```
