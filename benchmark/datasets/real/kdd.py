"""KDDCup99 / KDDCup98 loaders, following this repo's README conventions.

* **KDDCup99** — the 34 continuous attributes only (drops 3 nominal + 4 binary
  + the label). Fetched automatically via ``sklearn.datasets.fetch_kddcup99``.
  Label ``y`` = connection type.
* **KDDCup98** — the 56 RFM fields extracted from ``cup98LRN.txt`` (path must be
  provided). Label ``y`` = ``TARGET_B``.

Both stream in file/record order (no shuffling): the record order *is* the
streaming order.
"""

from __future__ import annotations

from ..base import IterableDataset

# --- KDDCup99: the 41 feature names in order (label excluded) --------------
_KDD99_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
# columns dropped per README: 3 nominal + 4 binary -> leaves 34 continuous ones
_KDD99_DROP = {
    "protocol_type", "service", "flag",  # nominal
    "land", "logged_in", "is_host_login", "is_guest_login",  # binary
}
_KDD99_CONTINUOUS = [c for c in _KDD99_COLUMNS if c not in _KDD99_DROP]  # 34


def load_kddcup99(n_samples: int | None = None, percent10: bool = True) -> IterableDataset:
    """Load KDDCup99 as a benchmark dataset (34 continuous features, README).

    Parameters
    ----------
    n_samples:
        Cap the number of streamed records (``None`` => all).
    percent10:
        Use the standard 10% subset (much faster) instead of the full set.
    """

    def factory():
        from sklearn.datasets import fetch_kddcup99

        bunch = fetch_kddcup99(percent10=percent10, as_frame=True)
        df = bunch.frame
        labels = df["labels"]
        rows = df[_KDD99_CONTINUOUS]
        limit = len(df) if n_samples is None else min(n_samples, len(df))
        for i in range(limit):
            x = {c: float(rows.iat[i, j]) for j, c in enumerate(_KDD99_CONTINUOUS)}
            y = labels.iat[i]
            y = y.decode() if isinstance(y, bytes) else str(y)
            yield x, y

    length = None
    return IterableDataset(
        factory, name="kddcup99", n_features=len(_KDD99_CONTINUOUS),
        has_labels=True, length=length,
    )


# --- KDDCup98: the 56 RFM fields (README) ----------------------------------
_KDD98_FIELDS = (
    [f"RAMNT_{i}" for i in range(3, 25)]  # 22 repeated donation amounts
    + [f"RDATE_{i}" for i in range(3, 25)]  # 22 repeated donation dates
    + [  # 12 summary fields
        "RAMNTALL", "NGIFTALL", "MINRAMNT", "MAXRAMNT", "LASTGIFT", "AVGGIFT",
        "CARDPROM", "NUMPROM", "CARDGIFT", "NUMPRM12", "TIMELAG", "CARDPM12",
    ]
)


def load_kddcup98(path: str, n_samples: int | None = None) -> IterableDataset:
    """Load KDDCup98 (charitable donation) from ``cup98LRN.txt`` (README).

    56 RFM feature fields; label ``y`` = ``TARGET_B`` (0/1). Non-numeric or
    missing cells are coerced to ``0.0``.
    """

    def factory():
        import csv

        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            count = 0
            for row in reader:
                if n_samples is not None and count >= n_samples:
                    break
                x = {}
                for f in _KDD98_FIELDS:
                    raw = (row.get(f) or "").strip()
                    try:
                        x[f] = float(raw) if raw else 0.0
                    except ValueError:
                        x[f] = 0.0
                target = (row.get("TARGET_B") or "0").strip()
                try:
                    y = int(float(target))
                except ValueError:
                    y = 0
                yield x, y
                count += 1

    return IterableDataset(
        factory, name="kddcup98", n_features=len(_KDD98_FIELDS), has_labels=True,
    )
