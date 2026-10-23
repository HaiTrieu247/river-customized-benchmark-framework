"""Loaders of the real natural-drift streams River does not ship (X7).

The fast tests run on tiny fixture files placed in a temporary ``$RIVER_DATA``;
the one ``slow`` test reads the real files (downloading them on first use).
"""

from __future__ import annotations

import gzip
from collections import Counter

import pytest

from benchmark.datasets.real import drift_streams as D


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIVER_DATA", str(tmp_path))
    return tmp_path


def _place(path, content: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return len(content)


# --- fetch --------------------------------------------------------------------------------

def test_fetch_downloads_then_reuses_the_cached_file(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"abcdef")
    dest = tmp_path / "cache" / "file.bin"
    assert D.fetch(src.as_uri(), dest, size=6) == dest
    assert dest.read_bytes() == b"abcdef"
    assert not (tmp_path / "cache" / "file.bin.part").exists()
    src.write_bytes(b"changed")  # a second call must not download again
    D.fetch(src.as_uri(), dest, size=6)
    assert dest.read_bytes() == b"abcdef"


def test_fetch_rejects_a_wrong_size_and_leaves_nothing_behind(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"<html>warning page</html>")
    dest = tmp_path / "cache" / "file.bin"
    with pytest.raises(RuntimeError, match="HTML page"):
        D.fetch(src.as_uri(), dest, size=10_000)
    assert not dest.exists()
    assert not (tmp_path / "cache" / "file.bin.part").exists()


def test_fetch_replaces_a_cached_file_of_the_wrong_size(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"abcdef")
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"ab")
    D.fetch(src.as_uri(), dest, size=6)
    assert dest.read_bytes() == b"abcdef"


def test_fetch_failure_names_the_url_and_the_place_to_put_the_file(tmp_path):
    dest = tmp_path / "file.bin"
    with pytest.raises(RuntimeError, match="place it at"):
        D.fetch((tmp_path / "missing.bin").as_uri(), dest, size=3)


# --- Rialto ---------------------------------------------------------------------------------

def _rialto_fixture(home, monkeypatch, n=5):
    rows = [" ".join(f"{(i + 1) * 0.01 + j * 1e-4:g}" for j in range(27)) for i in range(n)]
    data = ("\n".join(rows) + "\n").encode()
    labels = ("\n".join(str(i % 10) for i in range(n)) + "\n").encode()
    sizes = {"rialto.data": _place(home / "Rialto" / "rialto.data", data),
             "rialto.labels": _place(home / "Rialto" / "rialto.labels", labels)}
    for name, size in sizes.items():
        monkeypatch.setitem(D.RIALTO_FILES, name, size)
    monkeypatch.setattr(D, "RIALTO_N_SAMPLES", n)


def test_rialto_reads_27_floats_and_an_int_label_in_order(home, monkeypatch):
    _rialto_fixture(home, monkeypatch)
    ds = D.load_rialto()
    rows = list(ds)
    assert len(rows) == len(ds) == 5
    assert ds.n_features == 27 and ds.has_labels
    x, y = rows[2]
    assert list(x) == [f"f{i}" for i in range(1, 28)]
    assert x["f1"] == pytest.approx(0.03) and isinstance(x["f27"], float)
    assert [y for _, y in rows] == [0, 1, 2, 3, 4]


def test_rialto_is_reiterable_and_capped(home, monkeypatch):
    _rialto_fixture(home, monkeypatch)
    ds = D.load_rialto(n_samples=3)
    assert list(ds) == list(ds)
    assert len(list(ds)) == len(ds) == 3


# --- Sensor ---------------------------------------------------------------------------------

SENSOR_LINES = [
    "2004-02-28 01:10:16.6789 25 1 19.1 38.8 45.08 2.68",       # mote 1, later
    "2004-02-28 00:59:16.02785 3 1 19.9 37.0 45.08 2.69",       # mote 1, earliest
    "2004-02-28 01:06:16.013453 17 1 19.2",                      # missing light and voltage
    "2004-02-28 01:00:00 3 2 18.0 40.0 50.0 2.70",               # mote 2, no fractional seconds
    "2004-02-28 01:00:00 3 2 18.0 40.0 50.0 2.70",               # exact duplicate of the line above
    "2004-02-28 01:01:00 4 58 18.0 40.0 50.0 2.70",              # not one of the 54 motes
    "2004-02-28 01:02:00 5 3 abc 40.0 50.0 2.70",                # unparsable reading
    "2004-02-28 01:03:00 5 3 18.5 41.0 51.0 2.71",               # mote 3, same epoch as above
]


def _sensor_fixture(home, monkeypatch):
    raw = gzip.compress(("\n".join(SENSOR_LINES) + "\n").encode())
    monkeypatch.setattr(D, "SENSOR_SIZE", _place(home / "Sensor" / "data.txt.gz", raw))


def test_sensor_drops_incomplete_unparsable_and_foreign_readings_and_counts_them(home, monkeypatch):
    _sensor_fixture(home, monkeypatch)
    ds = D.load_sensor()
    stats = D.sensor_cleaning()
    assert stats == {"n_raw": 8, "dropped_missing": 1, "dropped_unparsable": 1, "dropped_mote": 1,
                     "duplicate_rows": 1, "duplicate_mote_epoch": 1, "n_samples": 5}
    assert len(ds) == 5 and ds.n_features == 4


def test_sensor_is_in_time_order_with_duplicates_kept(home, monkeypatch):
    _sensor_fixture(home, monkeypatch)
    rows = list(D.load_sensor())
    assert [y for _, y in rows] == [1, 2, 2, 3, 1]
    assert list(rows[0][0]) == list(D.SENSOR_FEATURES)
    assert rows[0][0]["temperature"] == pytest.approx(19.9)
    assert rows[1] == rows[2]


def test_sensor_builds_its_processed_cache_once(home, monkeypatch):
    _sensor_fixture(home, monkeypatch)
    D.load_sensor()
    processed = home / "Sensor" / f"sensor_v{D.SENSOR_CACHE_VERSION}.npz"
    assert processed.exists()
    (home / "Sensor" / "data.txt.gz").unlink()  # the second load never reads the raw file
    ds = D.load_sensor(n_samples=2)
    assert len(list(ds)) == len(ds) == 2
    assert list(ds) == list(ds)


# --- INSECTS --------------------------------------------------------------------------------

def _insects_fixture(home, monkeypatch, variant="incremental_imbalanced", n=4):
    rows = [",".join([f"{i + j * 0.5:g}" for j in range(33)] + [str(2 + i % 3)]) for i in range(n)]
    cfg = D.INSECTS_EXTRA_VARIANTS[variant]
    size = _place(home / "Insects" / cfg["filename"], ("\n".join(rows) + "\n").encode())
    monkeypatch.setitem(cfg, "size", size)
    monkeypatch.setitem(cfg, "n_samples", n)


def test_insects_extra_reads_like_river_33_floats_and_a_string_class(home, monkeypatch):
    _insects_fixture(home, monkeypatch)
    ds = D.load_insects_extra("incremental_imbalanced")
    rows = list(ds)
    assert len(rows) == len(ds) == 4
    assert ds.name == "insects_incremental_imbalanced" and ds.n_features == 33
    x, y = rows[1]
    assert list(x) == [f"f{i}" for i in range(1, 34)] and x["f1"] == 1.0
    assert [y for _, y in rows] == ["2", "3", "4", "2"]


def test_insects_extra_is_reiterable_and_capped(home, monkeypatch):
    _insects_fixture(home, monkeypatch, "incremental_reoccurring_imbalanced")
    ds = D.load_insects_extra("incremental_reoccurring_imbalanced", n_samples=2)
    assert list(ds) == list(ds)
    assert len(list(ds)) == len(ds) == 2


def test_insects_extra_rejects_an_unknown_variant(home):
    with pytest.raises(KeyError, match="load_insects"):
        D.load_insects_extra("abrupt_balanced")


def test_loaders_are_exported():
    import benchmark.datasets as datasets

    for name in ("load_rialto", "load_sensor", "load_insects_extra", "INSECTS_EXTRA_VARIANTS"):
        assert getattr(datasets, name) is getattr(D, name)


# --- the real files ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_files_have_the_documented_shape():
    """Downloads on first use (~400 MB in all)."""
    ds = D.load_rialto()
    rows = list(ds)
    assert len(rows) == D.RIALTO_N_SAMPLES and len(rows[0][0]) == 27
    assert sorted(Counter(y for _, y in rows)) == list(range(10))

    sensor = D.load_sensor()
    assert 2_000_000 < len(sensor) < 2_400_000
    assert set(D.sensor_cleaning()) >= {"dropped_missing", "dropped_mote", "n_samples"}

    for variant, cfg in D.INSECTS_EXTRA_VARIANTS.items():
        rows = list(D.load_insects_extra(variant))
        assert len(rows) == cfg["n_samples"]
        assert len(rows[0][0]) == 33 and len({y for _, y in rows}) == 6
