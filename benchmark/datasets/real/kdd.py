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
#: The 3 nominal columns, streamed as strings when include_nominal is set.
#: They need a categorical encoder in the preprocessing chain to be usable.
_KDD99_NOMINAL = ["protocol_type", "service", "flag"]


# Attack type to 5 main clusters (MIT Lincoln Labs / Paper [7, 8])
_KDD99_ATTACK_TO_MAIN = {
    "normal": "normal",
    # Denial of Service (DoS)
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "mailbomb": "dos",
    # Probe
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    # User to Root (U2R)
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
    # Remote to Local (R2L)
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "worm": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
}


def _map_attack_label(label: str) -> str:
    cleaned = label.rstrip(".").lower()
    return _KDD99_ATTACK_TO_MAIN.get(cleaned, cleaned)


def load_kddcup99(
    n_samples: int | None = None,
    percent10: bool = True,
    include_nominal: bool = False,
    map_to_5_classes: bool = True,
) -> IterableDataset:
    """Load KDDCup99 as a benchmark dataset.

    Parameters
    ----------
    n_samples:
        Cap the number of streamed records (``None`` => all).
    percent10:
        Use the standard 10% subset (much faster) instead of the full set.
    include_nominal:
        Also stream the 3 nominal columns (``protocol_type``, ``service``,
        ``flag``) as strings, giving 37 features instead of 34.
    map_to_5_classes:
        Map specific attack names to the 5 main clusters (Normal, DoS, Probe,
        U2R, R2L) as defined in MIT Lincoln Lab / KDD Cup benchmark [7, 8].
    """

    columns = _KDD99_CONTINUOUS + (_KDD99_NOMINAL if include_nominal else [])

    # Fetched once and cached on the loader, so repeated settings/seeds re-stream
    # the identical records without re-reading the archive.
    cache: dict = {}

    def _fetch():
        if not cache:
            from sklearn.datasets import fetch_kddcup99

            bunch = fetch_kddcup99(percent10=percent10, as_frame=True)
            df = bunch.frame
            limit = len(df) if n_samples is None else min(n_samples, len(df))
            # Materialise once as a float matrix + label list. Going through
            # `.iat` per cell instead costs 34 pandas lookups per record, which
            # dominates the wall clock and adds nothing to what is measured.
            cache["X"] = df[_KDD99_CONTINUOUS].to_numpy(dtype=float)[:limit]
            raw_labels = [_as_text(v) for v in df["labels"].to_numpy()[:limit]]
            if map_to_5_classes:
                cache["y"] = [_map_attack_label(v) for v in raw_labels]
            else:
                cache["y"] = raw_labels
            cache["nominal"] = (
                [
                    [_as_text(v) for v in df[col].to_numpy()[:limit]]
                    for col in _KDD99_NOMINAL
                ]
                if include_nominal
                else []
            )
        return cache["X"], cache["y"], cache["nominal"]

    def factory():
        X, y, nominal = _fetch()
        n_num = len(_KDD99_CONTINUOUS)
        for i, (row, label) in enumerate(zip(X, y)):
            x = dict(zip(columns[:n_num], row.tolist()))
            for name, values in zip(_KDD99_NOMINAL, nominal):
                x[name] = values[i]
            yield x, label

    def length_fn():
        return len(_fetch()[1])

    return _LazyLenDataset(
        factory,
        name="kddcup99_mixed" if include_nominal else "kddcup99",
        n_features=len(columns),
        has_labels=True,
        length_fn=length_fn,
    )


def _as_text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class _LazyLenDataset(IterableDataset):
    """Dataset whose length is only known once the archive has been fetched."""

    def __init__(self, factory, name, n_features, has_labels, length_fn):
        super().__init__(factory, name=name, n_features=n_features, has_labels=has_labels)
        self._length_fn = length_fn

    def __len__(self) -> int:
        return self._length_fn()


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
