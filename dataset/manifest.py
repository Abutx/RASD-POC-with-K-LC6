"""Build / query the capture manifest.

  python -m dataset.manifest build <dir> [<dir> ...]
  python -m dataset.manifest stats

Scans for .npz/.wav capture files, reads the .json sidecar next to each
(files without one are indexed as label='unlabeled' so backfill debt is
visible in stats), infers mode/fs/duration from the file itself, and writes
everything into dataset/manifest.sqlite.
"""

import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent / "manifest.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS captures (
    path        TEXT PRIMARY KEY,
    sha256      TEXT,
    bytes       INTEGER,
    label       TEXT,
    source      TEXT,
    mode        TEXT,
    fs          REAL,
    seconds     REAL,
    shape       TEXT,
    session     TEXT,
    target      TEXT,
    distance_m  REAL,
    aspect      TEXT,
    if_gain_db  REAL,
    extra_json  TEXT
);
"""

SKIP_KEYS = {"label", "source", "mode", "session", "target",
             "distance_m", "aspect", "if_gain_db"}


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _probe(path):
    """Infer (mode, fs, seconds, shape) from the raw file itself."""
    if path.suffix == ".wav":
        import soundfile as sf
        info = sf.info(str(path))
        return "cw", float(info.samplerate), info.frames / info.samplerate, \
            f"({info.frames},{info.channels})"
    z = np.load(str(path), allow_pickle=True)
    keys = set(z.files)
    fs = float(z["fs"]) if "fs" in keys else None
    if "data" in keys:
        n = z["data"].shape[-1]
        return "cw", fs, (n / fs if fs else None), str(z["data"].shape)
    if "chirps" in keys:
        ramp = float(z["ramp"]) if "ramp" in keys else None
        n = z["chirps"].shape[0]
        return "fmcw_chirps", fs, (n / ramp if ramp else None), str(z["chirps"].shape)
    if "cpis" in keys:
        c = z["cpis"].shape
        ramp = float(z["ramp"]) if "ramp" in keys else 1000.0
        return "fmcw_cpis", fs, c[0] * c[1] / ramp, str(c)
    return "other", fs, None, str({k: getattr(z[k], "shape", None) for k in z.files})


def _session_of(path, meta):
    if meta.get("session"):
        return str(meta["session"])
    m = re.match(r"(\d{8})_\d{6}", path.stem)
    return m.group(1) if m else path.parent.name


def build(dirs):
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    n_new = 0
    for d in dirs:
        for path in sorted(Path(d).rglob("*")):
            if path.suffix not in (".npz", ".wav") or not path.is_file():
                continue
            sidecar = path.with_suffix(".json")
            meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
            try:
                mode, fs, seconds, shape = _probe(path)
            except Exception as e:
                print(f"  skip {path}: {type(e).__name__}: {e}")
                continue
            extra = {k: v for k, v in meta.items() if k not in SKIP_KEYS}
            con.execute(
                "REPLACE INTO captures VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(path.resolve()), _sha256(path), path.stat().st_size,
                 meta.get("label", "unlabeled"), meta.get("source", "bench"),
                 meta.get("mode", mode), meta.get("fs", fs), seconds, shape,
                 _session_of(path, meta), meta.get("target"),
                 meta.get("distance_m"), meta.get("aspect"),
                 meta.get("if_gain_db"), json.dumps(extra) if extra else None))
            n_new += 1
    con.commit()
    print(f"indexed {n_new} files -> {DB_PATH}")


def stats():
    con = sqlite3.connect(DB_PATH)
    print(f"{DB_PATH}\n")
    print("by label:")
    for lab, n, sec, mb in con.execute(
            "SELECT label, COUNT(*), SUM(COALESCE(seconds,0)), SUM(bytes)/1e6 "
            "FROM captures GROUP BY label ORDER BY 2 DESC"):
        print(f"  {lab:22s} {n:4d} files  {sec or 0:7.1f} s  {mb:8.1f} MB")
    print("\nby source / mode:")
    for src, mode, n in con.execute(
            "SELECT source, mode, COUNT(*) FROM captures GROUP BY 1,2 ORDER BY 3 DESC"):
        print(f"  {src:8s} {mode:12s} {n:4d}")
    print("\nsessions:", [r[0] for r in con.execute(
        "SELECT DISTINCT session FROM captures ORDER BY 1")])
    n_unlab = con.execute(
        "SELECT COUNT(*) FROM captures WHERE label='unlabeled'").fetchone()[0]
    if n_unlab:
        print(f"\n!! {n_unlab} files have no sidecar -- write .json labels for them")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "build":
        build(sys.argv[2:])
    elif len(sys.argv) == 2 and sys.argv[1] == "stats":
        stats()
    else:
        print(__doc__)
