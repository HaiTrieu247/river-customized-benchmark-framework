"""Real streams with natural drift that River does not ship.

All seven INSECTS variants River ships are already protocol folds, so a check
that the findings hold on drift nobody designed needs streams from elsewhere.
These four are the user's choice (2026-10-23); each is long enough for the
protocol's windows (``h = 1000``) to leave an evaluation tail after window 20.

* **Rialto** (Losing, Hammer and Wersing 2016): 82 250 colour histograms of ten
  buildings along the Rialto bridge in Venice, taken from webcam images over
  twenty consecutive days. Light and weather change continuously, which is the
  drift. The ten classes arrive interleaved (one image per building per
  timestamp), so every window holds all of them.
* **Sensor** (Zhu 2010, from the Intel Berkeley Research Lab data): readings of
  the lab's sensors over five weeks (2004-02-28 to 2004-04-03); temperature,
  humidity, light and voltage are the features and the sensor id is the class.
  Daily cycles, a building that changes during the day and batteries running
  down make the drift. The protocol's 100 000-sample cap keeps about the first
  27 hours.
* **INSECTS incremental_imbalanced, incremental_reoccurring_imbalanced** and
  ``incremental_abrupt_imbalanced`` (Souza et al. 2020): the imbalanced
  counterparts of three protocol folds, 452 044 samples each. River lists
  their URLs but leaves them commented out; the files are still on the
  authors' Google Drive, behind the large-file warning page, so they are
  fetched from Drive's direct download host.

Downloads are cached where River caches its own (``$RIVER_DATA``, default
``~/river_data``), written to a ``.part`` file and renamed only once the size
matches, so an interrupted download never looks like a finished one. Each
loader yields float features as a dict and the label as ``y``, in the order
of the stream, and restarts from the first record on every iteration.

References
----------
Losing, Hammer, Wersing (2016). *KNN classifier with self adjusting memory for
heterogeneous concept drift.* ICDM, 291-300. Data:
https://github.com/vlosing/driftDatasets (realWorld/rialto).

Zhu (2010). *Stream data mining repository.* Built from Intel Berkeley Research
Lab, https://db.csail.mit.edu/labdata/labdata.html (data.txt.gz).

Souza, Reis, Maletzke, Batista (2020). *Challenges in benchmarking stream
learning algorithms with real-world data.* Data Mining and Knowledge Discovery
34(6), 1805-1858.
"""

from __future__ import annotations

import csv
import gzip
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ..base import IterableDataset

# --- downloads ------------------------------------------------------------------------

#: Drive answers a plain ``uc?export=download`` link to a file this large with an
#: HTML "can't scan for viruses" page; its own form posts to this host with
#: ``confirm=t``, which returns the file.
_DRIVE = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"


def data_home() -> Path:
    """River's cache directory: ``$RIVER_DATA``, else ``~/river_data``
    (``river.datasets.base.get_data_home``, without importing River)."""
    return Path(os.path.expanduser(os.environ.get("RIVER_DATA", os.path.join("~", "river_data"))))


def fetch(url: str, dest: Path, size: int | None = None) -> Path:
    """``dest``, downloaded from ``url`` unless it is already there with ``size`` bytes.

    The bytes go to ``dest.part`` and are renamed onto ``dest`` only when the
    download completed with the expected size. A file of the wrong size at
    ``dest`` (an old copy, a manual placement gone wrong) is fetched again.
    """
    dest = Path(dest)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (river-benchmark)"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
    except (urllib.error.URLError, OSError) as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download {url} to {dest}: {exc}. "
            f"Download it by hand and place it at {dest}"
            + (f" ({size} bytes)." if size else ".")
        ) from exc
    got = part.stat().st_size
    if size is not None and got != size:
        head = part.read_bytes()[:200]
        part.unlink(missing_ok=True)
        hint = " (the server returned an HTML page)" if b"<html" in head.lower() else ""
        raise RuntimeError(
            f"download of {url} returned {got} bytes, expected {size}{hint}; "
            f"place the file at {dest} by hand"
        )
    os.replace(part, dest)
    return dest


def _capped(n_samples: int | None, total: int) -> int:
    return total if n_samples is None else min(n_samples, total)


# --- Rialto -----------------------------------------------------------------------------

RIALTO_URL = "https://raw.githubusercontent.com/vlosing/driftDatasets/master/realWorld/rialto/"
#: File name -> size in bytes, as GitHub serves them.
RIALTO_FILES = {"rialto.data": 10_239_344, "rialto.labels": 164_500}
RIALTO_N_SAMPLES = 82_250
RIALTO_N_FEATURES = 27


def load_rialto(n_samples: int | None = None) -> IterableDataset:
    """Rialto bridge timelapse: 82 250 records, 27 features, 10 classes (Losing et al. 2016).

    Each record is a 27-bin normalised colour histogram of one building's image
    region; the label is the building (0-9, an int). The order is the source
    order, which is time order. Nothing is dropped or transformed; four bins
    (``f7``, ``f8``, ``f12``, ``f21``) are 0 on every record, which adds nothing
    to any distance.
    """
    folder = data_home() / "Rialto"
    paths = {name: fetch(RIALTO_URL + name, folder / name, size) for name, size in RIALTO_FILES.items()}
    names = [f"f{i}" for i in range(1, RIALTO_N_FEATURES + 1)]

    def factory():
        n = 0
        with open(paths["rialto.data"], encoding="ascii") as xs, open(paths["rialto.labels"], encoding="ascii") as ys:
            for line, label in zip(xs, ys):
                if n_samples is not None and n >= n_samples:
                    return
                values = line.split()
                if len(values) != RIALTO_N_FEATURES:
                    raise ValueError(f"rialto.data line {n + 1}: {len(values)} values, expected {RIALTO_N_FEATURES}")
                yield dict(zip(names, map(float, values))), int(label)
                n += 1

    return IterableDataset(factory, name="rialto", n_features=RIALTO_N_FEATURES, has_labels=True,
                           length=_capped(n_samples, RIALTO_N_SAMPLES))


# --- Sensor -----------------------------------------------------------------------------

SENSOR_URL = "https://db.csail.mit.edu/labdata/data.txt.gz"
SENSOR_SIZE = 34_422_518
SENSOR_FEATURES = ("temperature", "humidity", "light", "voltage")
#: The deployment's motes; the raw file also holds a few readings tagged with
#: other ids (e.g. 55, 56, 58), which are not among the 54 sensors.
SENSOR_MOTES = range(1, 55)
#: Bump when a cleaning rule changes: the processed cache is named after it.
SENSOR_CACHE_VERSION = 1


def prepare_sensor(raw: Path, out: Path) -> dict:
    """Turn the raw Intel Lab file into the time-ordered Sensor stream, saved as ``out`` (.npz).

    Rules, in order, with what each removes counted in the returned dict:

    1. a line that does not hold all eight fields (date, time, epoch, mote id
       and the four readings) is dropped: the raw file truncates a reading's
       missing trailing values instead of marking them (``dropped_missing``);
    2. a line with a field that does not parse as a number or a timestamp is
       dropped (``dropped_unparsable``);
    3. a reading whose mote id is not one of the 54 sensors is dropped
       (``dropped_mote``);
    4. the rest is sorted by timestamp, stable, so readings with the same
       timestamp keep the file's order. The file is grouped by mote, so without
       this the "stream" would visit one sensor after another.

    Rule 1 removes two whole motes: mote 28 never reports light (its lines hold
    seven fields) and mote 5 has 35 readings, none complete, so the stream
    carries 52 of the 54 sensors. Duplicates are kept: a reading repeated on the
    same mote and epoch is a
    retransmission the network delivered, and dropping it would be a cleaning
    decision about the data rather than a reading of it. Their counts are
    returned (``duplicate_rows``: identical on every field;
    ``duplicate_mote_epoch``: same mote and epoch). Physically implausible
    values (temperatures above 100 C from motes whose battery is failing) are
    kept for the same reason.
    """
    import pandas as pd

    names = ["date", "time", "epoch", "moteid", *SENSOR_FEATURES]
    with gzip.open(raw, "rt", encoding="ascii", errors="replace") as handle:
        lines = handle.read().splitlines()
    n_raw = len(lines)
    split = [line.split() for line in lines]
    del lines
    full = [parts for parts in split if len(parts) == len(names)]
    stats = {"n_raw": n_raw, "dropped_missing": n_raw - len(full)}
    del split

    df = pd.DataFrame(full, columns=names)
    del full
    numeric = ["epoch", "moteid", *SENSOR_FEATURES]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    stamp = pd.to_datetime(df["date"] + " " + df["time"], format="mixed", errors="coerce")
    ok = stamp.notna() & df[numeric].notna().all(axis=1)
    stats["dropped_unparsable"] = int((~ok).sum())
    df, stamp = df[ok], stamp[ok]

    mote_ok = df["moteid"].isin(list(SENSOR_MOTES)) & (df["moteid"] == df["moteid"].round())
    stats["dropped_mote"] = int((~mote_ok).sum())
    df, stamp = df[mote_ok], stamp[mote_ok]

    order = np.argsort(stamp.to_numpy(dtype="datetime64[ns]").astype(np.int64), kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    stats["duplicate_rows"] = int(df.duplicated().sum())
    stats["duplicate_mote_epoch"] = int(df.duplicated(subset=["moteid", "epoch"]).sum())
    stats["n_samples"] = int(len(df))

    X = df[list(SENSOR_FEATURES)].to_numpy(dtype=float)
    y = df["moteid"].to_numpy(dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part.npz")
    np.savez(part, X=X, y=y, **{f"stat_{k}": np.int64(v) for k, v in stats.items()})
    os.replace(part, out)
    return stats


def _sensor_arrays() -> tuple[np.ndarray, np.ndarray, dict]:
    folder = data_home() / "Sensor"
    processed = folder / f"sensor_v{SENSOR_CACHE_VERSION}.npz"
    if not processed.exists():
        prepare_sensor(fetch(SENSOR_URL, folder / "data.txt.gz", SENSOR_SIZE), processed)
    with np.load(processed) as z:
        stats = {k.removeprefix("stat_"): int(z[k]) for k in z.files if k.startswith("stat_")}
        return z["X"], z["y"], stats


def sensor_cleaning() -> dict:
    """What :func:`prepare_sensor` removed and counted, read from the processed cache."""
    return _sensor_arrays()[2]


def load_sensor(n_samples: int | None = None) -> IterableDataset:
    """Intel Lab Sensor: 2 210 084 readings, 4 features, 52 classes (Zhu 2010).

    Features are temperature (C), humidity (%), light (lux) and voltage (V);
    the label is the mote id (an int in 1-54; motes 5 and 28 never appear, see
    :func:`prepare_sensor`); records come in time order. The
    first call downloads the raw file (34 MB) and builds a processed cache next
    to it (see :func:`prepare_sensor` for the rules and
    :func:`sensor_cleaning` for what they removed).
    """
    X, y, _ = _sensor_arrays()
    total = len(y)
    stop = _capped(n_samples, total)

    def factory():
        for i in range(stop):
            yield dict(zip(SENSOR_FEATURES, X[i].tolist())), int(y[i])

    return IterableDataset(factory, name="sensor", n_features=len(SENSOR_FEATURES), has_labels=True,
                           length=stop)


# --- INSECTS, the variants River leaves out -------------------------------------------------

#: The imbalanced variants River's ``Insects`` comments out. The sizes are what
#: Drive serves today (2026-10-23), not the ones in River's comment (140 MB),
#: and all three files share them: the same 452 044 records in another order.
INSECTS_EXTRA_VARIANTS: dict[str, dict] = {
    "incremental_imbalanced": {
        "n_samples": 452_044, "size": 120_798_571, "drive_id": "1K3vp0EjA4FPDeSgffiBe4CMosbB4CKbL",
        "filename": "incremental_imbalanced.csv", "drift": "incremental", "balanced": False,
    },
    "incremental_reoccurring_imbalanced": {
        "n_samples": 452_044, "size": 120_798_571, "drive_id": "1aSqdxvZvug-SwQw5NLY_9nTPhhEjB2ig",
        "filename": "incremental_reoccurring_imbalanced.csv", "drift": "incremental + reoccurring",
        "balanced": False,
    },
    "incremental_abrupt_imbalanced": {
        "n_samples": 452_044, "size": 120_798_571, "drive_id": "1M6QfsernUlM0qvqXdbo9bPYAsHCcfuhb",
        "filename": "incremental_abrupt_imbalanced.csv", "drift": "incremental + abrupt", "balanced": False,
    },
}
INSECTS_N_FEATURES = 33


def load_insects_extra(variant: str, n_samples: int | None = None) -> IterableDataset:
    """One of :data:`INSECTS_EXTRA_VARIANTS` (33 optical-sensor features, 6 classes).

    Read exactly as River reads the variants it ships (``load_insects``):
    features ``f1``-``f33`` as floats, the class as River's ``iter_csv`` gives
    it, a string. Cached in River's own ``Insects`` folder under the file name
    River would use.
    """
    if variant not in INSECTS_EXTRA_VARIANTS:
        raise KeyError(
            f"unknown INSECTS variant {variant!r}; available here: {sorted(INSECTS_EXTRA_VARIANTS)} "
            "(the others: benchmark.datasets.load_insects)"
        )
    cfg = INSECTS_EXTRA_VARIANTS[variant]
    path = fetch(_DRIVE.format(id=cfg["drive_id"]), data_home() / "Insects" / cfg["filename"], cfg["size"])
    names = [f"f{i}" for i in range(1, INSECTS_N_FEATURES + 1)]

    def factory():
        n = 0
        with open(path, newline="", encoding="ascii") as handle:
            for row in csv.reader(handle):
                if n_samples is not None and n >= n_samples:
                    return
                if not row:
                    continue
                if len(row) != INSECTS_N_FEATURES + 1:
                    raise ValueError(f"{path.name} record {n + 1}: {len(row)} fields, expected {INSECTS_N_FEATURES + 1}")
                yield dict(zip(names, map(float, row[:-1]))), row[-1]
                n += 1

    return IterableDataset(factory, name=f"insects_{variant}", n_features=INSECTS_N_FEATURES, has_labels=True,
                           length=_capped(n_samples, cfg["n_samples"]))
