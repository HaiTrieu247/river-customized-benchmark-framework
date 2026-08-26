"""Forest CoverType loader.

After KDDCup99, Forest CoverType is the most frequently used real dataset in the
stream-clustering literature (StreamKM++, D-Stream, EDMStream, DBSTREAM all
report on it), which is the main reason to include it: it makes results here
comparable with published numbers.

581,012 observations of a 30x30m forest cell, labelled with one of 7 cover
types. Of the 54 attributes only the **first 10 are quantitative**; the other 44
are one-hot indicators for wilderness area and soil type. This loader keeps the
10 quantitative ones by default, matching how the clustering literature uses the
dataset and staying consistent with the KDDCup99 loader here, which likewise
drops nominal and binary columns. Euclidean distance over one-hot columns is not
meaningful, so feeding all 54 to a distance-based clusterer would mostly measure
the encoding.

**Caveat worth knowing before reading any score.** Souza et al. (2020) show the
records carry strong sequential dependence, almost certainly because the rows
are in geographic order and neighbouring cells share a cover type. That ordering
is undocumented and no geolocation metadata ships with the data. For this
benchmark the consequence is concrete: evaluation windows tend to hold a single
class, which makes per-window external metrics degenerate. The runner measures
exactly this and reports it as ``window_label_diagnostics`` — read that before
reading the time-aware table.

References
----------
Blackard & Dean (1999). *Comparative accuracies of artificial neural networks
and discriminant analysis in predicting forest cover types.*
Souza, Reis, Maletzke, Batista (2020). *Challenges in benchmarking stream
learning algorithms with real-world data.* DMKD 34(6).
"""

from __future__ import annotations

from ..base import IterableDataset

#: The 10 quantitative attributes, in dataset order. The remaining 44 columns
#: are one-hot wilderness-area (4) and soil-type (40) indicators.
COVTYPE_QUANTITATIVE = [
    "elevation",
    "aspect",
    "slope",
    "horizontal_distance_to_hydrology",
    "vertical_distance_to_hydrology",
    "horizontal_distance_to_roadways",
    "hillshade_9am",
    "hillshade_noon",
    "hillshade_3pm",
    "horizontal_distance_to_fire_points",
]


def load_covtype(
    n_samples: int | None = None,
    quantitative_only: bool = True,
) -> IterableDataset:
    """Load Forest CoverType as a benchmark stream.

    Downloaded once via ``sklearn.datasets.fetch_covtype`` (~11 MB compressed)
    and cached by scikit-learn afterwards.

    Parameters
    ----------
    n_samples:
        Cap the number of streamed records (``None`` => all 581,012).
    quantitative_only:
        Keep only the 10 quantitative attributes (default). Set ``False`` to
        stream all 54, including the one-hot indicators.
    """
    cache: dict = {}

    def _fetch():
        if not cache:
            from sklearn.datasets import fetch_covtype

            bunch = fetch_covtype()
            X, y = bunch.data, bunch.target
            limit = len(X) if n_samples is None else min(n_samples, len(X))
            if quantitative_only:
                X = X[:, : len(COVTYPE_QUANTITATIVE)]
                columns = list(COVTYPE_QUANTITATIVE)
            else:
                columns = list(COVTYPE_QUANTITATIVE) + [
                    f"wilderness_area_{i}" for i in range(4)
                ] + [f"soil_type_{i}" for i in range(40)]
            cache["X"] = X[:limit].astype(float)
            cache["y"] = [int(v) for v in y[:limit]]
            cache["columns"] = columns
        return cache["X"], cache["y"], cache["columns"]

    def factory():
        X, y, columns = _fetch()
        for row, label in zip(X, y):
            yield dict(zip(columns, row.tolist())), label

    n_features = len(COVTYPE_QUANTITATIVE) if quantitative_only else 54

    class _CovType(IterableDataset):
        def __len__(self) -> int:
            return len(_fetch()[1])

    return _CovType(
        factory,
        name="covtype" if quantitative_only else "covtype_full",
        n_features=n_features,
        has_labels=True,
    )
