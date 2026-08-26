"""Adapters for the datasets River ships with.

River already downloads and caches a number of the streams the literature uses,
so wrapping them costs nothing and widens the benchmark considerably.

By default the adapter keeps only features it can read as numbers, which is what
the coordinate-based clusterers need. That is a *default*, not an assumption
about clustering: pass ``numeric_only=False`` for algorithms that consume
something else — ``TextClust`` works on term-frequency dictionaries built from a
raw text field, which a numeric filter would strip away entirely (see
:func:`load_sms_spam`).

The important addition is **INSECTS**. Souza et al. (2020) built it precisely
because the field's usual benchmarks are unsatisfactory: their drift is
undocumented, so a drop in score cannot be attributed to a distribution change
rather than to underfitting or noise. INSECTS comes from optical sensors reading
flying insects, with drift induced by controlled temperature changes, so the
*type* and *timing* of each change are known in advance. Its variants cover
abrupt, gradual, incremental and incremental-reoccurring drift, each in balanced
and imbalanced form. For a clustering benchmark that means the drift scenarios
can be compared against a documented ground truth instead of guessed at.

References
----------
Souza, Reis, Maletzke, Batista (2020). *Challenges in benchmarking stream
learning algorithms with real-world data.* Data Mining and Knowledge Discovery
34(6), 1805-1858. USP DS repository: https://sites.google.com/view/uspdsrepository
"""

from __future__ import annotations

import numbers

from ..base import IterableDataset

#: INSECTS variants, with the sample counts River reports for each. Drift type
#: is documented by the authors rather than inferred, which is the whole point.
INSECTS_VARIANTS: dict[str, dict] = {
    "abrupt_balanced": {"n_samples": 52_848, "drift": "abrupt", "balanced": True},
    "abrupt_imbalanced": {"n_samples": 355_275, "drift": "abrupt", "balanced": False},
    "gradual_balanced": {"n_samples": 24_150, "drift": "gradual", "balanced": True},
    "gradual_imbalanced": {"n_samples": 143_323, "drift": "gradual", "balanced": False},
    "incremental_balanced": {"n_samples": 57_018, "drift": "incremental", "balanced": True},
    "incremental_abrupt_balanced": {
        "n_samples": 79_986, "drift": "incremental + abrupt", "balanced": True,
    },
    "incremental_reoccurring_balanced": {
        "n_samples": 79_986, "drift": "incremental + reoccurring", "balanced": True,
    },
}


def _as_float(value):
    """Coerce a feature value to float, or return ``None`` if it is categorical.

    Coercing rather than type-checking matters here: several River datasets
    (INSECTS among them) yield their numeric columns as *strings*, so an
    ``isinstance(v, Number)`` filter silently discards every feature and leaves
    an empty stream.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, numbers.Number):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def from_river_dataset(
    dataset,
    name: str,
    n_samples: int | None = None,
    numeric_only: bool = True,
) -> IterableDataset:
    """Wrap any ``river.datasets`` object as a benchmark stream.

    Features that cannot be read as numbers are dropped when ``numeric_only``
    is set (the default), because a coordinate-based clusterer has no distance
    for a categorical string. Turn it off for algorithms that consume raw
    fields — the text clusterers need the string itself.

    The underlying River dataset is re-iterated on every pass, so repeated
    settings and seeds all consume the identical record order.
    """

    def factory():
        n_yielded = 0
        for x, y in dataset:
            if n_samples is not None and n_yielded >= n_samples:
                break
            if numeric_only:
                coerced = {k: _as_float(v) for k, v in x.items()}
                x = {k: v for k, v in coerced.items() if v is not None}
            if not x:
                continue
            yield x, y
            n_yielded += 1

    # Probe one record so n_features is known before the stream is consumed.
    probe = next(iter(factory()), None)
    if probe is None:
        raise ValueError(
            f"river dataset {name!r} yielded no usable records; "
            "every feature was dropped as non-numeric"
        )
    n_features = len(probe[0])

    declared = getattr(dataset, "n_samples", None)
    length = declared if n_samples is None else (
        declared if declared is None else min(n_samples, declared)
    )

    return IterableDataset(
        factory, name=name, n_features=n_features, has_labels=True, length=length,
    )


def load_insects(
    variant: str = "abrupt_balanced",
    n_samples: int | None = None,
) -> IterableDataset:
    """Load one INSECTS variant (33 optical-sensor features, 6 classes).

    Downloaded and cached by River on first use (6-95 MB depending on variant).

    Parameters
    ----------
    variant:
        One of :data:`INSECTS_VARIANTS`. The drift type each one contains is
        documented by the dataset authors, so a scenario can be chosen on
        purpose rather than hoped for.
    n_samples:
        Cap the number of streamed records.
    """
    if variant not in INSECTS_VARIANTS:
        raise KeyError(
            f"unknown INSECTS variant {variant!r}; available: {sorted(INSECTS_VARIANTS)}"
        )
    from river import datasets

    return from_river_dataset(
        datasets.Insects(variant=variant), name=f"insects_{variant}", n_samples=n_samples
    )


def load_shuttle(n_samples: int | None = None) -> IterableDataset:
    """NASA Shuttle: 49,097 records, 9 numeric features.

    River ships the outlier-detection framing, so the label is **binary**
    (nominal vs anomalous) rather than the original 7 cover classes. Two
    heavily imbalanced classes make a weak clustering target on their own — the
    value here is speed and the purity/ARI disagreement that severe imbalance
    produces, which is a useful check that both metrics are being read. Do not
    make it a headline result.
    """
    from river import datasets

    return from_river_dataset(datasets.Shuttle(), name="shuttle", n_samples=n_samples)


def load_keystroke(n_samples: int | None = None) -> IterableDataset:
    """Keystroke dynamics: 20,400 records, 31 features, 51 users.

    Many classes relative to its length, which stresses algorithms that cap the
    number of micro-clusters.
    """
    from river import datasets

    return from_river_dataset(datasets.Keystroke(), name="keystroke", n_samples=n_samples)


def load_sms_spam(n_samples: int | None = None) -> IterableDataset:
    """SMS Spam: 5,574 short messages, labelled spam/ham. **Text, not numeric.**

    Included because not every River clusterer works on coordinate vectors:
    ``TextClust`` consumes term-frequency dictionaries and compares them by
    tf-idf cosine. Pair it with the ``bag_of_words`` preprocessor, which turns
    the ``body`` field into token counts::

        Setting("TextClust", scaler="bag_of_words", scaler_params={"on": "body"})

    The raw text field is kept (``numeric_only=False``); a numeric filter would
    strip it and leave an empty stream. Two classes make this a weak external
    target, so read it as an exercise of the text path rather than a ranking.
    River also ships ``TREC07`` (75k emails) if a larger text stream is wanted.
    """
    from river import datasets

    return from_river_dataset(
        datasets.SMSSpam(), name="sms_spam", n_samples=n_samples, numeric_only=False,
    )


def load_electricity(n_samples: int | None = None) -> IterableDataset:
    """Elec2: 45,312 records, 8 features, 2 classes.

    Included for continuity with the literature, but treat it with suspicion:
    Souza et al. show its labels are so autocorrelated that predicting "same as
    last" scores 85% where independence would give 51%. With only two classes it
    is also a weak clustering target. Use it as a secondary check, never as the
    headline result.
    """
    from river import datasets

    return from_river_dataset(datasets.Elec2(), name="electricity", n_samples=n_samples)
