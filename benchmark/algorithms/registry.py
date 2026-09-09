"""Registry of River clusterers, their presets, and the preprocessing factory.

The registry covers **every** clusterer in ``river.cluster``, not just the
distance-based ones, so it has to record what each algorithm actually supports
rather than assume a single protocol. Two of the seven break the usual
"assign this sample to a cluster" shape:

* **ODAC** clusters *variables*, not samples: it builds a hierarchy over the
  feature set, and its ``predict_one`` raises ``NotImplementedError`` by design.
  Sample-level external metrics (ARI/NMI/AMI/purity) are undefined for it. What
  it does expose is a structure — cluster count, active-cluster count, height —
  so it is benchmarked on structure and resources instead.
* **TextClust** consumes *token-count dictionaries*, not numeric vectors, and
  compares them by tf-idf cosine rather than a metric on coordinates. It needs a
  bag-of-words preprocessor and a text stream; standardising term frequencies
  would be meaningless.

:class:`AlgorithmSpec` carries that information so the runner can adapt rather
than crash, and so a report never quietly prints a quality score that the
algorithm cannot produce.

Presets are sensible *starting points* (several drawn from the original papers).
Override any of them per benchmark setting.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from .adapter import ClustererAdapter, build_river_path, make_adapter
from .transformers import OneHotCategorical, safe_standard_scaler

# Ensure the sibling river tree is importable before we touch river.*.
build_river_path()

from river import (  # noqa: E402
    base,
    cluster,
    compose,
    feature_extraction,
    feature_selection,
    preprocessing,
)

#: What an algorithm consumes.
NUMERIC = "numeric"
TOKENS = "tokens"


@dataclass(frozen=True)
class AlgorithmSpec:
    """What a registered clusterer is and what protocol it supports.

    Attributes
    ----------
    factory:
        Returns ``(model, resolved_params)``.
    partitions_samples:
        Whether ``predict_one`` assigns *the sample being processed* to a
        cluster. ``False`` for algorithms that cluster something else (ODAC
        clusters variables), which means external quality metrics do not apply
        and the run is scored on structure and resources alone.
    input_kind:
        :data:`NUMERIC` for coordinate vectors, :data:`TOKENS` for
        term-frequency dictionaries.
    default_preprocessor:
        The preprocessing step that makes sense for this algorithm when the
        caller does not choose one.
    note:
        One line for help text and reports.
    provenance:
        Where the implementation came from and how far it has been verified.
        A benchmark table that mixes estimators shipped by River with ones
        written for this project is only readable if each row says which it is,
        so this string is printed in the report next to the parameters.
    """

    factory: Callable[..., Any]
    partitions_samples: bool = True
    input_kind: str = NUMERIC
    default_preprocessor: str = "none"
    note: str = ""
    provenance: str = "river upstream"


# --- clusterer factories: name -> (model, resolved_params) -----------------
def _clustream(**kw):
    params = dict(n_macro_clusters=5, max_micro_clusters=100, time_gap=100, seed=42)
    params.update(kw)
    return cluster.CluStream(**params), params


def _dstream(**kw):
    params = dict(
        grid_width=0.1,
        decaying_factor=0.998,
        dense_threshold=3.0,
        sparse_threshold=0.8,
        beta=0.3,
        n_grids=100,
    )
    params.update(kw)
    return cluster.DStream(**params), params


def _birch(**kw):
    params = dict(
        threshold=0.0,
        branching_factor=50,
        leaf_capacity=50,
        max_entries=500,
        distance_measure="d2",
        threshold_measure="diameter",
    )
    params.update(kw)
    return cluster.BIRCH(**params), params


def _improved_birch(**kw):
    params = dict(
        initial_threshold=0.5,
        modifying_factor=2.0,
        branching_factor=50,
        leaf_capacity=50,
        max_entries=500,
        distance_measure="d2",
        threshold_measure="diameter",
    )
    params.update(kw)
    return cluster.ImprovedBIRCH(**params), params


def _denstream(**kw):
    params = dict(
        decaying_factor=0.25, beta=0.5, mu=2.5, epsilon=0.5, n_samples_init=1000,
        stream_speed=100,
    )
    params.update(kw)
    return cluster.DenStream(**params), params


def _dbstream(**kw):
    params = dict(
        clustering_threshold=1.0, fading_factor=0.01, cleanup_interval=2,
        intersection_factor=0.3, minimum_weight=1.0,
    )
    params.update(kw)
    return cluster.DBSTREAM(**params), params


def _streamkmeans(**kw):
    params = dict(chunk_size=50, n_clusters=5, seed=42)
    params.update(kw)
    return cluster.STREAMKMeans(**params), params


def _kmeans(**kw):
    params = dict(n_clusters=5, halflife=0.5, seed=42)
    params.update(kw)
    return cluster.KMeans(**params), params


def _odac(**kw):
    params = dict(confidence_level=0.9, n_min=100, tau=0.1)
    params.update(kw)
    return cluster.ODAC(**params), params


def _repstream(**kw):
    params = dict(k=5, alpha=1.5, decay_rate=0.99, max_vertices=100, repository_fraction=0.5)
    params.update(kw)
    return cluster.RepStream(**params), params


def _sostream(**kw):
    params = dict(
        alpha=0.1, min_pts=2, merge_threshold=1.0, fading_factor=0.01,
        fade_threshold=0.1, fading_interval=100,
    )
    params.update(kw)
    return cluster.SOStream(**params), params


def _textclust(**kw):
    # `real_time_fading` defaults to True upstream, which makes `learn_one`
    # require a wall-clock `t` on every call and raise a TypeError without it.
    # The adapter supplies the stream position as `t` when a model accepts one,
    # but defaulting to the counter-based mode keeps the common case working
    # with no timestamps in the data.
    params = dict(radius=0.3, fading_factor=0.001, tgap=100, real_time_fading=False)
    params.update(kw)
    return cluster.TextClust(**params), params


#: name -> spec. Covers every clusterer exported by ``river.cluster``.
ALGORITHMS: dict[str, AlgorithmSpec] = {
    "CluStream": AlgorithmSpec(
        _clustream, default_preprocessor="standard",
        note="micro-clusters with cluster features; offline k-means on top",
    ),
    "BIRCH": AlgorithmSpec(
        _birch, default_preprocessor="standard",
        note=(
            "CF-tree with a memory budget; threshold grows on rebuild. Phase 3 is OFF by "
            "default so predict_one labels leaf entries. Setting n_macro_clusters turns it "
            "on, and it then re-runs agglomerative clustering on every predict_one"
        ),
        provenance=(
            "added for this project from Zhang et al. (1996); verified by branch tests "
            "against the paper pseudocode, no reference implementation compared"
        ),
    ),
    "ImprovedBIRCH": AlgorithmSpec(
        _improved_birch, default_preprocessor="standard",
        note=(
            "BIRCH with one threshold per leaf entry instead of one per tree. A rejected "
            "entry raises its own threshold by a fixed step before opening a new entry, so "
            "modifying_factor sets the step and initial_threshold sets where every entry "
            "starts. Ablation pair with BIRCH: same tree, same offline phase, threshold "
            "policy is the only difference. initial_threshold=0 freezes the mechanism and "
            "max_entries then stops being enforced"
        ),
        provenance=(
            "added for this project from Ismael et al. (2014); verified by branch tests "
            "against the paper pseudocode, no reference implementation compared. The paper "
            "gives no formula for the modifying factor and no memory-budget policy, so both "
            "were chosen here and are recorded in the estimator docstring"
        ),
    ),
    "DStream": AlgorithmSpec(
        _dstream, default_preprocessor="standard",
        note="density-and-grid based with time decay; arbitrary shaped clusters",
    ),
    "DenStream": AlgorithmSpec(
        _denstream, default_preprocessor="standard",
        note="density-based with time decay; epsilon must match the data scale",
    ),
    "DBSTREAM": AlgorithmSpec(
        _dbstream, default_preprocessor="standard",
        note="density-based using shared density between micro-clusters",
    ),
    "STREAMKMeans": AlgorithmSpec(
        _streamkmeans, default_preprocessor="standard",
        note="chunk-based k-means; no centre until the first chunk completes",
    ),
    "KMeans": AlgorithmSpec(
        _kmeans, default_preprocessor="standard",
        note="incremental k-means with a decaying learning rate",
    ),
    "ODAC": AlgorithmSpec(
        _odac,
        partitions_samples=False,
        default_preprocessor="none",
        note="hierarchical clustering OF VARIABLES; no per-sample assignment",
    ),
    "RepStream": AlgorithmSpec(
        _repstream, default_preprocessor="standard",
        note=(
            "two mutual-kNN sparse graphs: one over the points in memory, one over the "
            "representative points chosen from them. Clusters are the connected components "
            "of the density-related links between representatives, so there is no offline "
            "phase and no fixed radius: alpha scales each representative's own local "
            "density. max_vertices is the memory budget; half of it is reserved for the "
            "repository of representatives kept by usefulness rather than by age"
        ),
        provenance=(
            "added for this project from Luhr & Lazarescu (2009); verified by branch tests "
            "against the paper pseudocode, no reference implementation compared. Six places "
            "where the pseudocode contradicts the paper's own prose were resolved in favour "
            "of the prose, and Definition 7 needed one maintenance step the pseudocode "
            "omits; all seven are recorded in the estimator docstring"
        ),
    ),
    "SOStream": AlgorithmSpec(
        _sostream, default_preprocessor="standard",
        note="competitive learning with a k-dist adaptive radius; merges online, no offline phase",
        provenance=(
            "added for this project from Isaksson et al. (2012); verified by branch tests "
            "against the paper pseudocode, no reference implementation compared"
        ),
    ),
    "TextClust": AlgorithmSpec(
        _textclust,
        input_kind=TOKENS,
        default_preprocessor="bag_of_words",
        note="text micro-clusters compared by tf-idf cosine; needs token counts",
    ),
}


# --- preprocessing ---------------------------------------------------------
#: Every unsupervised transformer River offers that is meaningful in front of a
#: clusterer. A clustering benchmark has no ``y`` at learn time, so anything
#: needing a target is excluded by construction; what remains is more than
#: scaling, and the gaps the non-scalers fill are real ones:
#:
#: ============================  ================================================
#: Group                         Why a clustering benchmark needs it
#: ============================  ================================================
#: scalers                       every radius parameter (DenStream's
#:                               ``epsilon``, DBSTREAM's
#:                               ``clustering_threshold``) is read in the units
#:                               the scaler produces
#: imputers                      real streams have missing values, and River's
#:                               convention is that a missing feature is an
#:                               *absent key* -- a clusterer then compares
#:                               points over different key sets
#: categorical encoders          nominal columns otherwise have to be dropped;
#:                               KDDCup99 loses 3 and CoverType 44 that way
#: projections                   distance-preserving dimensionality reduction
#:                               (Johnson-Lindenstrauss), the standard tool for
#:                               high-dimensional stream clustering
#: variance filter               constant features contribute nothing to any
#:                               distance and actively break scalers -- a
#:                               constant column is what drove River's rolling
#:                               ``StandardScaler`` to a negative variance
#: text extractors               TextClust consumes token counts, and LDA turns
#:                               text into a dense vector the numeric
#:                               clusterers can handle
#: kernel approximation          random Fourier features let a centroid method
#:                               separate clusters that are not spherical
#: ============================  ================================================
#:
#: Deliberately absent: ``SelectKBest``, ``PoissonInclusion`` and the ``Target*``
#: transformers need a target; ``PredClipper`` wraps a regressor;
#: ``PolynomialExtender`` multiplies the dimensionality and distorts exactly the
#: distances a clusterer relies on, so it belongs to supervised feature
#: engineering rather than here.
PREPROCESSORS: dict[str, Callable[..., Any]] = {
    "none": lambda: None,
    # -- scaling: what changes is how much of the stream is remembered --------
    "standard": preprocessing.StandardScaler,          # window_size=N => rolling
    # Same scaler, guarded against RIVER_BUGS.md BUG-1: the rolling variant
    # emits complex numbers on a constant feature. Plain  is left
    # as untouched river behaviour so the two can be compared.
    "standard_safe": safe_standard_scaler,
    "minmax": preprocessing.MinMaxScaler,              # window_size=N => rolling
    "maxabs": preprocessing.MaxAbsScaler,              # window_size=N => rolling
    "robust": preprocessing.RobustScaler,              # median + IQR, no forgetting
    "adaptive_standard": preprocessing.AdaptiveStandardScaler,  # the only one that forgets
    "normalizer": preprocessing.Normalizer,            # per-row L-p, stateless
    "binarizer": preprocessing.Binarizer,
    # -- missing values ------------------------------------------------------
    "impute": preprocessing.StatImputer,               # per-feature statistic
    "impute_previous": preprocessing.PreviousImputer,  # last observed value
    # -- categorical ---------------------------------------------------------
    # `one_hot` encodes EVERY key, which shreds numeric columns on a mixed
    # record; `one_hot_categorical` leaves numbers alone. Prefer the latter
    # unless the record really is entirely categorical.
    "one_hot": preprocessing.OneHotEncoder,
    "one_hot_categorical": OneHotCategorical,
    "ordinal": preprocessing.OrdinalEncoder,
    "feature_hasher": preprocessing.FeatureHasher,     # high-cardinality
    "gap_encoder": preprocessing.GapEncoder,           # dirty categories, needs on=
    # -- dimensionality ------------------------------------------------------
    "sparse_projection": preprocessing.SparseRandomProjector,
    "gaussian_projection": preprocessing.GaussianRandomProjector,
    "variance_threshold": feature_selection.VarianceThreshold,
    # -- routing: pick or drop columns before a step that cannot be told apart
    "select": compose.Select,
    "discard": compose.Discard,
    "select_type": compose.SelectType,
    # -- text ----------------------------------------------------------------
    "bag_of_words": feature_extraction.BagOfWords,
    "tfidf": feature_extraction.TFIDF,
    "lda": preprocessing.LDA,                          # text -> topic vector
    # -- kernel approximation -------------------------------------------------
    "rbf_sampler": feature_extraction.RBFSampler,
    "random_trees_embedding": feature_extraction.RandomTreesEmbedding,
}

#: Backwards-compatible alias. The slot holds any preprocessing step, not only
#: scalers, so :data:`PREPROCESSORS` is the accurate name.
SCALERS = PREPROCESSORS

#: Separator for chaining steps in a preprocessor spec, e.g.
#: ``"impute|one_hot|standard"``.
CHAIN_SEP = "|"


def _make_step(name: str, **params):
    """Build a single preprocessing step by name."""
    if name not in PREPROCESSORS:
        raise KeyError(f"unknown preprocessor {name!r}; available: {sorted(PREPROCESSORS)}")
    if name == "none":
        if params:
            raise TypeError(f"preprocessor 'none' takes no params, got {sorted(params)}")
        return None
    try:
        return PREPROCESSORS[name](**params)
    except TypeError as exc:
        raise TypeError(f"bad params for preprocessor {name!r}: {exc}") from exc


def make_preprocessor(spec: str, **params):
    """Build a preprocessing step, or a chain of them, by name.

    ``spec`` is either a single name or several joined by ``|``, applied left to
    right::

        make_preprocessor("standard", window_size=10_000)
        make_preprocessor("impute|one_hot|standard")
        make_preprocessor("variance_threshold|sparse_projection|standard",
                          sparse_projection={"n_components": 16})

    A single step takes its params flat, as keyword arguments. A chain takes
    them nested under each step's name, because otherwise there is no way to say
    which step a parameter belongs to — two steps in the same chain can easily
    share a parameter name such as ``seed``.

    Chaining matters because the useful pipelines are rarely one step: raw data
    usually needs missing values filled and categories encoded *before* scaling
    means anything, and a high-dimensional stream wants constant columns dropped
    and the rest projected down before any distance is computed.

    Params are validated eagerly, so a typo fails at setup rather than silently
    benchmarking a different pipeline than the one that was asked for.
    """
    names = [n.strip() for n in spec.split(CHAIN_SEP) if n.strip()]
    if not names:
        raise KeyError(f"empty preprocessor spec {spec!r}")

    if len(names) == 1:
        return _make_step(names[0], **params)

    unknown = set(params) - set(names)
    if unknown:
        raise KeyError(
            f"params for a chain must be nested under a step name; "
            f"{sorted(unknown)} match no step in {names}"
        )
    steps = []
    for name in names:
        step_params = params.get(name) or {}
        if not isinstance(step_params, dict):
            raise TypeError(
                f"params for step {name!r} must be a dict, got {type(step_params).__name__}"
            )
        step = _make_step(name, **step_params)
        if step is not None:  # 'none' inside a chain is a no-op
            steps.append(step)
    if not steps:
        return None
    chained = steps[0]
    for step in steps[1:]:
        chained = chained | step  # river composes transformers into a Pipeline
    return chained


#: Kept for callers written before the slot took more than scalers.
make_scaler = make_preprocessor


def get_spec(name: str) -> AlgorithmSpec:
    if name not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {name!r}; available: {sorted(ALGORITHMS)}")
    return ALGORITHMS[name]


def get_algorithm(
    name: str,
    display_name: str | None = None,
    scaler: str = "none",
    scaler_params: dict | None = None,
    **overrides,
) -> ClustererAdapter:
    """Instantiate a registered algorithm behind the benchmark adapter.

    ``scaler`` names the preprocessing step; see :data:`PREPROCESSORS`.
    """
    spec = get_spec(name)
    model, params = spec.factory(**overrides)
    scaler_params = dict(scaler_params or {})
    scaler_obj = make_scaler(scaler, **scaler_params)
    return make_adapter(
        display_name or name, model, params,
        scaler=scaler_obj, scaler_name=scaler, scaler_params=scaler_params,
        partitions_samples=spec.partitions_samples,
    )


def accepts_seed(name: str) -> bool:
    """Whether the registered algorithm's constructor takes a ``seed`` kwarg.

    Used by the harness to decide whether to inject the run seed, instead of
    constructing the model and catching ``TypeError`` — which would also swallow
    a genuine parameter mistake.

    A ``**kwargs`` in the signature counts: ``STREAMKMeans`` declares no ``seed``
    of its own but forwards kwargs to the ``KMeans`` it wraps, and that inner
    model is exactly the randomised part that needs seeding.
    """
    get_spec(name)  # validate the name
    cls = getattr(cluster, name, None)
    if cls is None:
        return False
    params = inspect.signature(cls.__init__).parameters
    if "seed" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def algorithms_for(input_kind: str = NUMERIC, partitions_samples: bool | None = True):
    """Names of registered algorithms matching a capability filter.

    The default — numeric input, produces a sample partition — is the set that
    can be compared on external quality metrics. Pass ``partitions_samples=None``
    to include structure-only algorithms such as ODAC.
    """
    return [
        name
        for name, spec in ALGORITHMS.items()
        if spec.input_kind == input_kind
        and (partitions_samples is None or spec.partitions_samples == partitions_samples)
    ]


def discover_clusterers() -> dict[str, type]:
    """All ``river.cluster`` classes deriving from ``base.Clusterer``.

    Useful as a check that :data:`ALGORITHMS` has not fallen behind River.
    """
    out: dict[str, type] = {}
    for attr in dir(cluster):
        obj = getattr(cluster, attr)
        if isinstance(obj, type) and issubclass(obj, base.Clusterer) and obj is not base.Clusterer:
            out[attr] = obj
    return out
