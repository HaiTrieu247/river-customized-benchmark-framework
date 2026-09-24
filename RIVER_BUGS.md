# Suspected bugs in river

Findings from building this benchmark. **The river tree is never modified** (see
[AGENTS.md](AGENTS.md)) — each entry records the problem, a minimal reproduction,
the root cause, what it costs a benchmark, a suggested upstream fix, and the
workaround used here.

BUG-1 and BUG-2 were verified against the river source tree at `../river`,
version `0.26.1` (distribution metadata reports `0.26.0`), Python 3.13.3,
Windows 11.

BUG-3, BUG-4, BUG-5, GAP-1 and GAP-2 were verified against the same tree at
commit `88ca4633` (branch `main`), river `0.26.1`, Python 3.13.3, macOS 26.6.2.

BUG-6, GAP-3 and GAP-4 were verified against the same tree, river `0.26.1`,
Python 3.13.14, macOS 26.0.0, on 2026-09-22.

---

## BUG-1 — `StandardScaler(window_size=...)` can emit complex numbers

**Severity:** high — crashes a run, far from the cause.
**Status:** reproduced on unmodified river.
**Location:** [`river/preprocessing/scale.py:295-314`](../river/river/preprocessing/scale.py)

### Symptom

A run using a rolling standardiser dies inside a clusterer's distance function:

```
TypeError: VectorDict values must be real numbers
  File "river/cluster/dbstream.py", line 411, in predict_one
  File "river/utils/math.py", line 175, in minkowski_distance
```

### Minimal reproduction

```python
from benchmark.datasets import load_kddcup99
from river import preprocessing

ds = load_kddcup99(n_samples=50_000)
sc = preprocessing.StandardScaler(window_size=10_000)
for i, (x, _) in enumerate(ds):
    sc.learn_one(x)
    out = sc.transform_one(x)
    bad = {k: v for k, v in out.items() if isinstance(v, complex)}
    if bad:
        print(i, bad)
        break
```

Observed:

```
sample 14113: feature 'num_shells' -> (2.5169915865732083e-26-4.1105592050306033e-10j)
  rolling var = -2.0095036745715335e-18   raw x = 0.0
```

### Root cause

`transform_one` guards the division with a truthiness test:

```python
result[i] = (xi - m) / v**0.5 if v else 0.0
```

`num_shells` is constant `0.0` across the window, so the incrementally maintained
rolling variance drifts to a tiny **negative** value through floating-point
cancellation. `v` is then non-zero, so the guard passes; and `(-2e-18) ** 0.5` in
Python returns a **complex** number rather than raising. The complex value flows
into the model and only fails much later, inside Minkowski distance — which is why
the traceback points at the clusterer rather than the scaler.

The non-windowed branch on line 312 has the same shape and the same latent
exposure.

### Impact on benchmarking

Any rolling-scaler configuration is unusable on a stream containing a constant
feature. KDDCup99 has several (`num_outbound_cmds` is always 0), so the entire
"does a forgetting scaler track drift better?" comparison cannot be run.

### Suggested upstream fix

Guard on the sign, not on truthiness, in both branches:

```python
result[i] = (xi - m) / v**0.5 if v > 0 else 0.0
```

### Workaround in this framework

Use the `standard_safe` preprocessor, which wraps river's scaler and coerces any
non-real or non-finite output to `0.0`. Plain `standard` remains untouched river
behaviour so the two can be compared.

---

## BUG-2 — `KMeans` results depend on `PYTHONHASHSEED`, so `seed` does not make a run reproducible

**Severity:** high for benchmarking — silently breaks reproducibility.
**Status:** reproduced on unmodified river.
**Location:** [`river/cluster/k_means.py:122-129`](../river/river/cluster/k_means.py)

### Symptom

The same `seed`, the same data, the same code produce different clusterings from
one process to the next. Measured through this framework on a synthetic stream
(`n=3000, d=10, k=5`, STREAMKMeans, `seed=1`):

```
PYTHONHASHSEED=0 -> ARI = 0.2360384106
PYTHONHASHSEED=1 -> ARI = 0.5360306622
PYTHONHASHSEED=7 -> ARI = 0.5784372243
```

The stream itself is identical in all three runs — same labels, same scaled
values. Only the model output differs.

### Minimal reproduction

```bash
for h in 0 1 7; do
  PYTHONHASHSEED=$h python -c "
from benchmark.datasets.synthetic import StreamConfig, SyntheticStream
from benchmark.experiments import BenchmarkConfig, Setting, run_setting
ds = SyntheticStream(StreamConfig(n_samples=3000, n_features=10, n_clusters=5,
                                  separation=1.0, seed=1))
r = run_setting(Setting('STREAMKMeans', params={'n_clusters': 5, 'chunk_size': 500},
                        scaler='standard'), ds,
                BenchmarkConfig(n_checkpoints=3, window_size=1000), seed=1)
print(round(r.quality['ari'], 10))"
done
```

### Root cause

```python
def predict_one(self, x):
    def get_distance(c):
        center = self.centers[c]
        return sum(
            (abs(center[k] - x.get(k, 0))) ** self.p for k in {*center.keys(), *x.keys()}
        )
    return min(self.centers, key=get_distance)
```

Two facts combine:

1. `{*center.keys(), *x.keys()}` is a **set literal**, and set iteration order for
   string keys depends on `PYTHONHASHSEED`, which Python randomises per process.
2. `self.centers` holds `collections.defaultdict(rand_gauss)` where `rand_gauss`
   draws from the seeded RNG ([`k_means.py:98-101`](../river/river/cluster/k_means.py)).
   Touching a key a centre has not seen **draws the next value from that RNG** —
   this is how centroids are lazily initialised.

So the iteration order decides *which random coordinate is assigned to which
feature*. A different order gives different initial centroids, and in k-means an
early difference in assignment cascades into an entirely different clustering.

`STREAMKMeans` inherits this: it builds a fresh `KMeans` per chunk and feeds the
resulting centres into a running `KMeans`.

### Impact on benchmarking

The `seed` parameter advertises reproducibility it does not deliver across
processes. Repeated-seed runs report a mean ± std that mixes genuine seed
variance with hash-order variance, and results cannot be reproduced on another
machine or another day without knowing the hash seed that was in effect.

### Suggested upstream fix

Iterate the key union in a deterministic order. Materialise the centre's keys
before touching new ones, so the defaultdict may grow without mutating what is
being iterated:

```python
keys = list(center)
keys += [k for k in x if k not in center]
return sum((abs(center[k] - x.get(k, 0))) ** self.p for k in keys)
```

Sorting would also work but can raise on heterogeneous key types.

### Workaround in this framework

There is no way to fix this from outside river, so the framework makes it
**visible** instead:

- `capture_environment()` records `PYTHONHASHSEED`, so every report states the
  value a run was produced under.
- `run_benchmark.py` warns when a hash-order-sensitive algorithm is used with an
  unpinned hash seed.
- To get reproducible numbers, pin it before launching:

  ```bash
  PYTHONHASHSEED=0 python run_benchmark.py ...
  ```

A regression test asserts the *pinned* case is reproducible, which is the
guarantee that actually holds.

---

## BUG-3 — clusterers keep a reference to the caller's `x`, so reusing an input dict corrupts the model

**Severity:** high for `DBSTREAM` — silently changes results. Lower elsewhere.
**Status:** reproduced on unmodified river.
**Location:** [`river/cluster/dbstream.py:443`](../river/river/cluster/dbstream.py),
[`river/cluster/streamkmeans.py:95-99`](../river/river/cluster/streamkmeans.py),
[`river/cluster/denstream.py:141`](../river/river/cluster/denstream.py),
[`river/cluster/clustream.py:285`](../river/river/cluster/clustream.py)

### Symptom

A micro-cluster centre *is* the dict the caller passed in. Mutating that dict after
`learn_one` returns rewrites the model's state:

```python
from river.cluster import DBSTREAM

m = DBSTREAM(clustering_threshold=1.0)
x = {'a': 1.0, 'b': 2.0}
m.learn_one(x)
m.micro_clusters[0].center        # {'a': 1.0, 'b': 2.0}

x['a'] = 999.0                    # caller mutates their own dict
m.micro_clusters[0].center        # {'a': 999.0, 'b': 2.0}   <- model corrupted
m.micro_clusters[0].center is x   # True
```

### Minimal reproduction

Reusing one dict across the loop — an ordinary way to avoid an allocation per
event — changes the clustering. Same data, same order, same parameters:

```python
from river import cluster

m1 = cluster.DBSTREAM(clustering_threshold=1.0)
buf = {}
for v in [1.0, 1.1, 1.2, 5.0, 5.1]:
    buf.clear(); buf['a'] = v          # one dict, reused
    m1.learn_one(buf)

m2 = cluster.DBSTREAM(clustering_threshold=1.0)
for v in [1.0, 1.1, 1.2, 5.0, 5.1]:
    m2.learn_one({'a': v})             # a fresh dict each time

[mc.center for mc in m1.micro_clusters.values()]
# [{'a': 1.19559974818331}, {'a': 5.1}]
[mc.center for mc in m2.micro_clusters.values()]
# [{'a': 1.195002990387169}, {'a': 5.09559974818331}]
```

The reused-buffer run reports a centre of exactly `5.1` — the last raw value —
because the stored centre is the buffer, which was overwritten. The correct
centre has been moved toward `x` by the Gaussian factor: `5.0956`.

### Root cause

`DBSTREAMMicroCluster.__init__` assigns without copying:

```python
def __init__(self, x=None, last_update=None, weight=None):
    self.center = x
```

`center` is then read by `_find_fixed_radius_nn`, `predict_one` and `_recluster`,
so the aliasing is fully functional, not cosmetic.

The other three differ in how much it costs:

| Estimator | Where | Functional? |
|---|---|---|
| `DBSTREAM` | `dbstream.py:443`, `self.center = x` | **Yes** — centre is read everywhere |
| `STREAMKMeans` | `streamkmeans.py:95-99`, `self._temp_chunk[i] = x` | **Yes** — the chunk is replayed into `KMeans` once full |
| `DenStream` | `denstream.py:141`, `BufferItem.x = x` | Yes, during the `n_samples_init` buffer phase. `MicroCluster` already copies (`linear_sum = dict(x)`, `denstream.py:414`) |
| `CluStream` | `clustream.py:285`, `self.x = x` | No — `.x` is never read again. Still retains a reference to caller data and trips the check |

`CluStream` additionally uses a mutable default argument on that constructor
(`x: dict = defaultdict(float)`, `clustream.py:280`).

### Why CI does not catch it

`river/checks/common.py` has `check_no_state_aliasing_with_input`, which is
exactly this test — but it never runs on a clusterer. See GAP-1 below.

### Impact on benchmarking

The framework builds a fresh dict per event, so measurements here are unaffected.
It matters for anyone consuming these estimators from a real stream reader, and
it makes "same data, same params" a weaker guarantee than it reads.

### Suggested upstream fix

Copy on the way in. In `DBSTREAMMicroCluster.__init__`:

```python
self.center = dict(x) if x is not None else {}
```

and in `STREAMKMeans.learn_one`, store `dict(x)` into `_temp_chunk`. `DenStream`'s
`BufferItem` needs the same. `CluStream` can simply drop the unused `self.x`, and
replace the mutable default with `x: dict | None = None`.

### Workaround in this framework

None needed — `run_one` passes a freshly built dict per event. Do not "optimise"
that into a reused buffer.

---

## BUG-4 — `CluStream.learn_one` raises `KeyError` when a new feature appears mid-stream

**Severity:** high — crashes the run.
**Status:** reproduced on unmodified river.
**Location:** [`river/cluster/clustream.py:321`](../river/river/cluster/clustream.py)

### Symptom

```
KeyError: 'c'
  File "river/cluster/clustream.py", line 244, in learn_one
    closest_mc.insert(x, w, self._timestamp)
  File "river/cluster/clustream.py", line 321, in insert
    self.var_x[x_idx].update(x_val, w)
```

### Minimal reproduction

```python
from river import cluster

m = cluster.CluStream(seed=1, max_micro_clusters=3, time_window=10)
for i in range(10):
    m.learn_one({'a': float(i), 'b': float(i)})

m.learn_one({'a': 1.0, 'b': 1.0, 'c': 1.0})   # KeyError: 'c'
```

### Root cause

`CluStreamMicroCluster.var_x` is populated once, in `__init__`, from the keys of
the sample that created the micro-cluster:

```python
for k in x:
    v = stats.Var()
    v.update(x[k], w)
    self.var_x[k] = v
```

`insert` then assumes every key of every later sample already has an entry:

```python
def insert(self, x, w, timestamp):
    self.var_time.update(timestamp, w)
    for x_idx, x_val in x.items():
        self.var_x[x_idx].update(x_val, w)
```

A feature that appears after the micro-cluster was created has no `stats.Var`,
so the lookup raises. This contradicts river's documented convention that
features may appear and disappear mid-stream — `check_emerging_features` exists
precisely to enforce it, and does not run here (GAP-1).

### Impact on benchmarking

Any dataset with a sparse or evolving schema kills a `CluStream` run outright.
It also rules out `CluStream` from preprocessing chains that emit variable key
sets, such as one-hot encoding a categorical column whose levels arrive over time
— which is most real categorical streams, KDDCup99 included.

### Suggested upstream fix

Create the accumulator on demand:

```python
def insert(self, x, w, timestamp):
    self.var_time.update(timestamp, w)
    for x_idx, x_val in x.items():
        if x_idx not in self.var_x:
            self.var_x[x_idx] = stats.Var()
        self.var_x[x_idx].update(x_val, w)
    self._center = None
```

Note this makes the new feature's statistics start from a count of 1 while the
cluster's other features have a higher count, so `center` and `radius` mix
different sample sizes. That is a modelling decision worth raising in the issue
rather than deciding silently.

### Workaround in this framework

Put a preprocessing step that emits a fixed key set ahead of `CluStream`, or keep
`CluStream` off datasets with an evolving schema. The registry's `AlgorithmSpec`
is the right place to record the restriction.

---

## BUG-5 — `STREAMKMeans.predict_one` raises `ValueError` before the first chunk is full

**Severity:** medium — violates a documented convention, crashes test-then-train loops.
**Status:** reproduced on unmodified river.
**Location:** [`river/cluster/streamkmeans.py:113`](../river/river/cluster/streamkmeans.py)

### Symptom

```
ValueError: min() iterable argument is empty
  File "river/cluster/streamkmeans.py", line 113, in predict_one
    return min(self.centers, key=get_distance)
```

### Minimal reproduction

```python
from river import cluster

m = cluster.STREAMKMeans(chunk_size=10, n_clusters=2, seed=1)
m.predict_one({'a': 1.0})       # ValueError: min() iterable argument is empty
```

### Root cause

`__init__` sets `self.centers = {}` (`streamkmeans.py:87`) and `learn_one` only
assigns real centres at the end of its body. Until the first `learn_one`,
`centers` is empty and `min()` has nothing to iterate.

River's own convention, quoted from the docstring of
`check_predict_one_before_any_learn`:

> River's online-learning convention is that `predict_one` / `score_one` /
> `transform_one` are callable from the very first event, before any `learn_one`
> has been issued.

This matters for the standard test-then-train protocol, where `predict_one`
is called on the very first event by construction.

### Why CI does not catch it

`check_predict_one_before_any_learn` is gated on `_inference_methods(model)`,
which returns nothing for a clusterer. See GAP-2 below.

### Impact on benchmarking

`run_one` predicts before learning on every event, so the very first event of
every `STREAMKMeans` run would crash if the framework did not tolerate it.

### Suggested upstream fix

Return a default when there is nothing to compare against:

```python
def predict_one(self, x, w=None):
    if not self.centers:
        return 0
    ...
```

`DBSTREAM` already takes this approach — `predict_one` returns `0` when no valid
cluster exists, covered by `test_all_noisy_predict_one` in
`tests/cluster/test_dbstream.py`.

---

## GAP-1 — `check_estimator` runs no behavioural check at all on a `Clusterer`

**Severity:** high — it is why BUG-3, BUG-4 and BUG-5 all survive CI.
**Status:** measured on unmodified river.
**Location:** [`river/checks/__init__.py`](../river/river/checks/__init__.py), `_yield_datasets()`

### Symptom

```python
>>> from river import checks, cluster, anomaly, linear_model
>>> len(list(checks.yield_checks(cluster.DBSTREAM())))
14
>>> len(list(checks.yield_checks(anomaly.HalfSpaceTrees())))
27
>>> len(list(checks.yield_checks(linear_model.LogisticRegression())))
49
>>> len(list(checks._yield_datasets(cluster.DBSTREAM())))
0
```

Not one of the 14 checks that do run feeds the model a single data point.

### Root cause

`_yield_datasets()` dispatches on `Forecaster`, `Ranker`, `MultiTargetRegressor`,
`Regressor`, `MultiLabelClassifier`, `Classifier`, `AnomalyDetector`,
`Transformer` / `SupervisedTransformer`. `base.Clusterer` matches none of them, so
the generator yields nothing, and every dataset-driven check is skipped in
silence: `check_learn_one`, `check_pickling`, `check_shuffle_features_no_impact`,
`check_emerging_features`, `check_disappearing_features`,
`check_radically_disappearing_features`, `check_predict_one_pure`,
`check_predict_one_before_any_learn`, `check_no_state_aliasing_with_input`,
`check_clone_is_independent`, `check_bounded_memory_growth`.

The 14 that survive are all static: `repr`, `str`, `tags`, five clone checks, two
`__init__` parameter checks, `doc`, pickle roundtrip, `get_params`.

### What happens if the branch is added

Running the dataset checks by hand against `datasets.Phishing().take(200)`:

| | aliasing | shuffle features | emerging features | learn_one | predict_one pure |
|---|---|---|---|---|---|
| `DBSTREAM` | **fail** | pass | pass | pass | pass |
| `DenStream` | **fail** | pass | pass | pass | pass |
| `CluStream` | **fail** | pass | **fail** | pass | pass |
| `STREAMKMeans` | **fail** | **fail** | **fail** | pass | pass |
| `KMeans` | pass | **fail** | pass | pass | pass |

`KMeans` failing `check_shuffle_features_no_impact` is BUG-2 surfacing through a
different route.

### Suggested upstream fix

Add a branch to `_yield_datasets()`, in the same shape as the `AnomalyDetector`
one directly above it. A labelled numeric stream is what clustering checks want;
`datasets.Phishing()` already serves the classifier branch and works here.

This will turn CI red until BUG-3, BUG-4 and BUG-5 are addressed, so it should be
filed as an issue first, and the fix split from the estimator fixes rather than
bundled with them.

### Workaround in this framework

Run the dataset checks explicitly against any clusterer under test rather than
trusting `check_estimator` to have covered it.

---

## GAP-2 — `_inference_methods()` does not know clusterers have `predict_one`

**Severity:** medium — two more checks are silent no-ops even after GAP-1 is fixed.
**Status:** measured on unmodified river.
**Location:** [`river/checks/common.py:16-29`](../river/river/checks/common.py)

### Symptom

```python
>>> from river.checks.common import _inference_methods
>>> list(_inference_methods(cluster.DBSTREAM()))
[]
>>> list(_inference_methods(anomaly.HalfSpaceTrees()))
['score_one']
```

Both `check_predict_one_pure` and `check_predict_one_before_any_learn` start with

```python
methods = list(_inference_methods(model))
if not methods:
    return
```

so for a clusterer they return immediately, having asserted nothing.

### Root cause

`_inference_methods()` branches on `AnomalyDetector`, `Classifier`, `Regressor`,
`Transformer` / `SupervisedTransformer`. There is no `Clusterer` branch. Note
that the sibling helper `_infer()` *does* handle clusterers, by falling through to
`model.predict_one(x)` on its last line — so only the two `_inference_methods`-gated
checks are affected.

### Suggested upstream fix

```python
elif isinstance(model, base.Clusterer):
    yield "predict_one"
```

Fixing this is what makes BUG-5 visible to CI. Best filed together with GAP-1,
since alone it changes nothing — no dataset reaches these checks anyway.

---

## Notes

- BUG-1 and BUG-2 were originally patched locally in the river tree during
  development and both patches were reverted, at which point the bugs returned.
  That is the reason for the no-edits rule in [AGENTS.md](AGENTS.md).
- Nothing here has been reported upstream yet. All are worth filing at
  <https://github.com/online-ml/river/issues>.
- Suggested filing order: GAP-1 + GAP-2 as one issue about clusterer test
  coverage, then BUG-3, BUG-4 and BUG-5 as separate issues that the coverage
  issue can reference. BUG-4 is the only one that crashes a run outright.
- BUG-3, BUG-4, BUG-5, GAP-1 and GAP-2 were found while surveying how river tests
  clusterers, ahead of implementing further distance-based algorithms. Each was
  reproduced on the tree recorded at the top of this file before being written up.

---

## BUG-6 — `DenStream.predict_one` returns `-1` when it has no clusters, and the caller cannot tell that apart from a cluster id

**Severity:** high for any evaluation — it silently turns "produced no clustering" into "produced one cluster that scores zero", and makes the non-clustering run look like the fastest one on the board.
**Status:** measured on unmodified river.
**Location:** [`river/cluster/denstream.py:193`](../river/river/cluster/denstream.py) (`_get_closest_cluster_key`), reached from `predict_one` at `:388`.

### Symptom

```python
>>> import random
>>> from river import cluster
>>> model = cluster.DenStream(epsilon=1e-4, mu=2, beta=0.75,
...                           n_samples_init=10, stream_speed=10)
>>> rng = random.Random(0)
>>> out = []
>>> for _ in range(200):
...     x = {"x": rng.random() * 1000, "y": rng.random() * 1000}
...     out.append(model.predict_one(x))
...     model.learn_one(x)
>>> -1 in out
True
```

`-1` is in the same value space as a genuine cluster key, which is an `int`
index into `self.clusters`. Nothing in the return type, and nothing in the
docstring, says it means "no clustering exists".

### Root cause

`_get_closest_cluster_key(point, clusters)` starts with
`min_distance = math.inf` and `key = -1` and returns `key` unchanged when
`clusters` is empty. `predict_one` calls it with the dictionary built from the
p-micro-clusters, which is empty whenever DenStream has not managed to promote
anything — the normal state at a small `epsilon`, where `_initial_dbscan` finds
no point with more than `mu` neighbours inside the radius and no
o-micro-cluster ever reaches weight `beta * mu`.

### What it costs a benchmark

Measured on this pipeline's 50k pilot, DenStream at River's default `epsilon`
on covtype, insects, keystroke and shuttle:

| | value |
|---|---|
| `n_predicted_clusters` | 2 (the sentinel, plus at most one real key) |
| `largest_cluster_share` | 1.000 |
| `homogeneity` | 0.000 |
| ARI | 0.000 |
| throughput | 15,000 – 30,000 samples/s |

Two readings go wrong at once.

1. **Quality.** `0.000` is reported next to a *negative* ARI from a
   configuration that did cluster, and reads as better. It is not better; it is
   not a clustering. A run that assigns everything to one cluster and a run that
   assigns nothing are different failures and only one of them is about the
   hyperparameters.
2. **Cost.** Emitting a constant is nearly free, so the degenerate run tops the
   throughput and memory tables. "River's default DenStream is 200x faster than
   the tuned one" is, on those folds, a statement about a model that produced
   nothing.

### Suggested upstream fix

Return `None` from `predict_one` when `p_micro_clusters` is empty, and say so in
the docstring — `None` is already how this codebase signals "no label for this
sample" elsewhere. Failing that, raise, so the caller has to decide. Keeping a
valid-looking `int` is the one option that cannot be handled by a caller who
does not already know the internals.

### Workaround here

`benchmark/algorithms/adapter.py` declares `SENTINEL_LABELS = {"DenStream": (-1,)}`
and the adapter **always counts** the occurrences in
`n_sentinel_predictions`, which travels into every result row. Suppression is
opt-in via `ClustererAdapter(drop_sentinel_predictions=True)` and is **off by
default**, because the default has to keep measuring River's own behaviour —
that is what a River user gets. Turning it on returns `None`, and the runner
then leaves the sample out of the external metrics.

Independently of the flag, `hpo_compare`'s report section 4i lists every shipped
run that produced no partition, so the degenerate cells cannot be read as
results.

---

## GAP-3 — `DenStream.predict_one` re-runs the whole offline DBSCAN on every call

**Severity:** medium — it is not a wrong answer, but it makes the cost of
DenStream a statement about River's implementation rather than about the
algorithm, and it is the reason quality and speed cannot be traded off in it.
**Status:** measured on unmodified river.
**Location:** [`river/cluster/denstream.py:351-388`](../river/river/cluster/denstream.py) (`predict_one`), with `_query_neighbor` at `:252`.

### Symptom

`predict_one` does not look up a cached clustering. It re-labels every
p-micro-cluster from scratch:

```python
labels = {pmc: None for pmc in self.p_micro_clusters.values()}
for pmc in self.p_micro_clusters.values():
    ...
    seed_queue = self._query_neighbor(pmc)          # O(P) scan
    while seed_queue:
        ...
        neighbor_neighbors = self._query_neighbor(seed_queue[0])   # O(P) again
self.n_clusters, self.clusters = self._generate_clusters_for_labels(labels)
```

`_generate_clusters_for_labels` then `copy.copy`s and merges a micro-cluster per
label. All of it happens again on the next sample.

Compare CluStream, which rebuilds its offline phase once every `time_gap`
samples (`clustream.py:209`), and says so in its docstring.

### What it costs a benchmark

In test-then-train evaluation `predict_one` is called once per sample, so the
run is `O(N x P^a)`. Fitted on this pipeline's 50k pilot, over 5 datasets:

| dataset | exponent a | r |
|---|---|---|
| kdd99 | 1.22 | 0.94 |
| covtype | 1.51 | 0.99 |
| insects | 1.43 | 0.97 |
| keystroke | 1.54 | 0.99 |
| shuttle | 1.30 | 0.98 |

At equal micro-cluster count on covtype, DenStream runs 3.3x slower than
CluStream at P≈15 and 15.6x slower at P≈60.

The consequence for hyperparameter work is the one that matters: `epsilon` is
the only knob that raises ARI, and lowering it multiplies `P`. With cost
superlinear in `P` and no cache, every quality gain is bought at a
superlinearly rising price, so an unconstrained argmax over ARI selects the
slowest configuration on the curve by construction.

### Suggested upstream fix

Cache the labelling and invalidate it when `p_micro_clusters` changes, or adopt
CluStream's approach and recompute on a period. The paper's own framing supports
this: the offline phase runs "when a clustering request arrives", and in a
test-then-train loop every sample is being treated as such a request.

### Workaround here

None in the model — it would change what is being measured. Instead the pipeline
reports cost per micro-cluster rather than raw throughput, and `hpo_compare`
adds a constrained ceiling (report section 3b-iii: best ARI subject to a
throughput floor) so that a quality number bought with compute is visible as
such.

---

## GAP-4 — `CluStream` documents a pyramidal time frame it does not implement

**Severity:** medium — the docstring promises the paper's central contribution,
and a benchmark that cites it is citing something absent.
**Status:** verified by reading the whole class (368 lines).
**Location:** [`river/cluster/clustream.py:15-17`](../river/river/cluster/clustream.py) (docstring).

### Symptom

The docstring states:

> The micro-clusters are stored at snapshots in time following a pyramidal
> pattern. This pattern allows to recall summary statistics from different time
> horizons.

Neither is implemented:

* `"snapshot"` and `"pyramid"` occur **only** in that docstring — nowhere in the code.
* `__init__` allocates one current `self.micro_clusters: dict`; there is no store of past states.
* There is no horizon argument anywhere. `predict_one(self, x)` takes none, and no method accepts a time range.
* The paper's `alpha` (snapshot base) and `l` (snapshots retained per order) do not exist as parameters.
* `time_window` — the only time-related parameter — is used once, at `:151`, as a relevance-stamp threshold for deleting an old micro-cluster. That is the paper's `delta`, not the pyramidal frame.

### What it costs a benchmark

The online/offline split that lets an analyst ask for clusters over an arbitrary
past horizon *is* the CluStream contribution (VLDB 2003, Sec. 3.1). River
implements the online micro-cluster maintenance and an offline k-means over the
**current** micro-clusters only. Benchmarking `river.cluster.CluStream` and
describing it as CluStream therefore overstates what was measured, and any
statement about horizon queries is untestable here.

### Suggested upstream fix

Cheapest correct action: delete the two sentences from the docstring and state
that the implementation keeps the current micro-cluster set only. Implementing
the frame is a larger change and would add `alpha` and `l` to the signature.

### Workaround here

Documentation only: the benchmark says that River's CluStream has no pyramidal
time frame, and makes no horizon claims.
