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

The GAP-1 to GAP-4 workarounds were verified on 2026-10-06 against the same
tree at commit `873fa6e0` (branch `main`), river `0.26.1`, Python 3.13.3 (the
repo's `.venv`), macOS 27.0.1, with `PYTHONHASHSEED=0`. The measurements in
their workaround sections were taken the same day on the same tree, under the
conditions each section states.

GAP-5 was measured on 2026-10-07 against the same tree and environment.

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

This removes the dependence on the hash seed but not on order altogether: new
centre coordinates are still drawn in the caller's key order, so the result
depends on how the caller orders its dict. Measured with the GAP-1 harness,
`check_shuffle_features_no_impact` still fails with this fix for 19 of 20
shuffle seeds on `KMeans` and 20 of 20 on `STREAMKMeans`; iterating in sorted
key order removes it. Sorting is therefore the fix to prefer, with a fallback
(e.g. `key=repr`) for heterogeneous key types, on which a plain sort raises.

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

`benchmark/algorithms/checks.py` runs the checks river skips, without touching
river. `yield_clusterer_checks(model)` yields three things in order:

1. Everything `checks.yield_checks` yields for the model, unchanged and in order.
2. River's own dataset-driven checks, each bound to `clusterer_datasets()`:
   `datasets.Phishing().take(200)`, the stream the table above was measured on.
   Each check gets a fresh one-shot stream, as in river.
3. `check_bounded_memory_growth` on `clusterer_datasets(scale=10)`. Phishing has
   1250 rows, so that stream has 1250 points, the same cap river's classifier
   branch has.

`check_clusterer(model)` is `check_estimator` for a clusterer. It honours
`_unit_test_skips` and raises on the first failure, with a note naming the
check. `run_clusterer_checks(model)` returns one of three values per check name:

- `"pass"`
- `"fail: <exception>: <message> @ file:line"`
- `"skip: listed in _unit_test_skips"`

All three functions take `make_datasets`, a function of the same shape as
`clusterer_datasets`, so a clusterer that does not take numeric vectors can be
checked on its own stream.

There is one adaptation. River lets an inference method answer
`NotImplementedError`: its two predict_one checks skip it, and a classifier's
`_infer` falls back from `predict_proba_one`. But the other dataset checks call
`_infer` unguarded. On a clusterer that predicts nothing they would stop at the
first point and never reach `learn_one`; ODAC is one, since it clusters
variables, not samples. So when `NotImplementedError` escapes a dataset check,
the harness reruns the check on a fresh clone and the same, untouched data,
with `predict_one` answering `None` instead of raising. The rerun model is a
clone of the model with its class swapped for a subclass that overrides only
`predict_one`, so nested estimators come through intact. Rebuilding it from
`_get_params()` would not work, because that encodes a nested estimator as a
`(class, params)` tuple. That is how the ODAC crash below was found.

`tests/test_clusterer_checks.py` pins the rerun with three planted clusterers:

- one whose `predict_one` raises `NotImplementedError` passes every check;
- the same with a nested `StandardScaler` passes every check;
- one whose `learn_one` indexes the first point's features in every later point
  fails exactly the emerging and radically disappearing checks, with
  `KeyError`. River's own checks stop at `NotImplementedError` before reaching
  that code.

The list of dataset checks is a literal inside `yield_checks`, so
`river_dataset_checks()` copies it branch for branch. The same test file pins
the copy in three ways:

- It compares the copy with river's own output, by module, qualified name and
  name, for eight estimator types river does feed: binary classifier,
  multi-class tree, regressor, stochastic anomaly detector, mini-batch
  transformer, multi-label, model selector and forecaster. A check river adds,
  drops, reorders or wraps differently fails the test.
- Per clusterer, it asserts that the harness yields exactly river's checks, then
  the copy with the two GAP-2 swaps on 200-point streams, then the memory check
  on the long stream.
- It asserts every stream is one-shot, and checks this through a learn count:
  `check_radically_disappearing_features` must make 190 `learn_one` calls, not
  210.

The same file asserts that the gap is still upstream: `_yield_datasets` yields
nothing for any river clusterer. It also shows that a clusterer keeping the
caller's `x` as a centre passes `check_estimator` and fails `check_clusterer`
on `check_no_state_aliasing_with_input`, and only there.

The matrix runs every `river.cluster` class and every registry entry, so it
includes the three framework subclasses: `DenStreamPatched` (BUG-7, BUG-8),
`DenStreamPatchedFast` (GAP-3) and `CluStreamPyramidal` (GAP-4). Each is built
from registry defaults. TextClust is fed token counts made by `BagOfWords` over
a fixed two-topic vocabulary, as in its docstring. Every other clusterer is fed
Phishing. Re-measured 2026-10-06 at river `873fa6e0`, `PYTHONHASHSEED=0`, with
`random.seed(0)` before each cell:

| | failing checks | cause |
|---|---|---|
| `DBSTREAM`, `DenStream`, `DenStreamPatched`, `DenStreamPatchedFast` | aliasing | BUG-3 |
| `CluStream`, `CluStreamPyramidal` | aliasing; emerging features (`KeyError`) | BUG-3, BUG-4 |
| `STREAMKMeans` | aliasing | BUG-3 |
| `STREAMKMeans` | pickling, shuffle features, emerging, disappearing, radically disappearing, predict before learn; seeding idempotence when `seed` is unset (`ValueError`) | BUG-5 |
| `ODAC` | emerging features (`TypeError` at `odac.py:480`) | not filed |
| `KMeans` | none under `PYTHONHASHSEED=0`; under other hash seeds, shuffle features for some shuffle seeds | BUG-2 |
| `BIRCH`, `ImprovedBIRCH` | `check_doc`, a static check, so `check_estimator` already fails | no class docstring, not filed |
| `DCUStream`, `DDStream`, `DStream`, `HPStream`, `MRStream`, `RepStream`, `SOStream`, `TextClust` | none | |

Every failure is a strict xfail that pins the exception type and cites its
cause, so a fix upstream turns the test red until the expectation is removed.
KMeans is the exception: under a hash seed other than 0 its cell is a
non-strict xfail. A separate test reruns it in two subprocesses that differ
only in `PYTHONHASHSEED`: hash seed 0 passes and 3 fails, with `random.seed(5)`.

A framework subclass may inherit the failures of the entry it extends, with the
same exception, and add none.
`test_framework_subclasses_fail_only_what_their_reference_fails` asserts this,
so a new failing cell for a subclass is a defect in the subclass, not a cell to
xfail. All three meet it: each fails the cells of its reference with the same
exception type, which the strict xfails pin, and passes the rest. When measured
on 2026-10-06 the exceptions were also raised at the same lines
(`common.py:442`, `clustream.py:321`); no test pins the line.

The aliasing check looks at the first `learn_one` only, where the inherited
BUG-3 already fails it, so it cannot see an alias that a subclass adds. A
probe, not a test, mutated every input after `learn_one` over the whole stream
and found none that changes results:

- under mutation, `DenStreamPatchedFast` predicted exactly what
  `DenStreamPatched` predicted;
- `CluStreamPyramidal`'s horizon answers, snapshots and id lists were the same
  with and without mutation.

The registry pins `seed=42`, and river yields `check_seeding_is_idempotent`
only when `seed` is `None`. So a separate test clones every entry whose
parameters include `seed` with `seed=None`: CluStream, CluStreamPyramidal,
HPStream, KMeans and STREAMKMeans. Only STREAMKMeans fails, through BUG-5.

Differences from the table above:

- **`KMeans`** passes `check_shuffle_features_no_impact` under
  `PYTHONHASHSEED=0` for 20 of 20 shuffle seeds. Under other hash seeds it
  fails for some of them:

  | hash seed | 1 | 3 | 4 | 6 | 8 | 123 |
  |---|---|---|---|---|---|---|
  | failing shuffle seeds (of 20) | 14 | 18 | 14 | 11 | 10 | 18 |

  At the matrix's `random.seed(0)`, only hash seed 4 fails. So the earlier
  failure was BUG-2 under an unpinned hash seed. Running the test file under
  another hash seed usually reports that cell as XPASS.
- **`STREAMKMeans`** fails the shuffle and emerging-feature checks through
  BUG-5's `ValueError` on the first `predict_one`.
  - In the shuffle check BUG-5 masks a second failure. With `predict_one`
    answering 0 while there are no centres, the check still fails for 5 of 20
    shuffle seeds under `PYTHONHASHSEED=0` (seed 0 among them), and for 19 of
    20 under 3.
  - Iterating the inner KMeans's key union in sorted order removes it under
    both, so it is BUG-2 through the KMeans inside STREAMKMeans.
  - BUG-2's suggested upstream fix does not remove it. That fix iterates the
    centre's keys, then `x`'s new keys in `x`'s order, so it draws new centre
    coordinates in the caller's key order. With it, the check fails for 20 of
    20 shuffle seeds under both hash seeds, and plain `KMeans` for 19 of 20.
    It trades hash-order dependence for caller-order dependence.
  - Once BUG-5 is fixed, the shuffle cell will raise `AssertionError` and its
    xfail will turn red.
  - BUG-5 also fails pickling, disappearing, radically disappearing and
    predict-before-learn, which the table did not list.

`ODAC.learn_one` raises `TypeError` when `x` lacks the one series of a
singleton cluster. There, `update_statistics` calls `float(x.get(name))`
without the `None` guard its multi-series branch has. Minimal repro:
`m = cluster.ODAC(); m.learn_one({'a': 1.0}); m.learn_one({'b': 2.0})`. Through
the harness it fails `check_emerging_features` for 4 of 10 shuffle seeds, the
same under hash seeds 0, 1 and 3. River's checks cannot see it, because they
stop at `predict_one`'s `NotImplementedError`.

At registry defaults, some clusterers never leave their initial buffer on 200
points: `DenStream`, `DenStreamPatched` and `DenStreamPatchedFast`
(`n_samples_init=1000`) and `HPStream` (`n_samples_init=2000`). Their passes on
the short-stream checks cover the buffering phase only. For the two DenStream
subclasses this means their overridden `predict_one` never gets past the
not-initialised early return, so this matrix does not run the BUG-7, BUG-8 and
GAP-3 code paths. Those paths are covered by `tests/test_denstream_patched.py`
and `tests/test_denstream_fast.py`.

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

With GAP-1 fixed, this adds the one check that tests BUG-5's convention
directly, `check_predict_one_before_any_learn`. BUG-5 does not depend on it:
GAP-1's fix alone already exposes BUG-5, through the five other dataset checks
that infer before they learn (measured, see below). Best filed together with
GAP-1, since alone it changes nothing — no dataset reaches these checks anyway.

### Workaround in this framework

`benchmark/algorithms/checks.py` defines its own `check_predict_one_pure` and
`check_predict_one_before_any_learn`. They have river's code, with
`predict_one` as the inference method of a `base.Clusterer`; every other
estimator type falls back to river's `_inference_methods`.
`yield_clusterer_checks` (GAP-1) uses them in place of river's under the same
names, so `_unit_test_skips` and reports keep working.

`tests/test_clusterer_checks.py` covers this in four ways:

- It asserts that the gap is still upstream: `_inference_methods(cluster.DBSTREAM())`
  is empty.
- It pins both functions to river's with `ast`: same arguments and statements,
  ignoring docstrings and comments. The only difference is which
  `_inference_methods` the name resolves to. If river changes either body, the
  test fails until the copy follows.
- On a regressor, where river's versions are active, the two agree, both on
  passing and on raising.
- It proves the fix with two planted defects: a `predict_one` that writes into
  the caller's dict, and one that raises before the first `learn_one`. River's
  checks return `None` on both. The framework's raise `AssertionError` and
  `ValueError`.

In the matrix, this adds the one cell that tests river's convention directly:
`STREAMKMeans` fails `check_predict_one_before_any_learn` with `ValueError`.
With river's `_inference_methods` swapped back in, only that one cell changes;
the other five BUG-5 cells come from GAP-1's fix alone.

`check_predict_one_pure` asserts only that the input is unchanged, as river's
does. A `predict_one` that changes the model's own state passes it and every
other check. The test file keeps a planted clusterer of that kind passing, so
this limit stays visible. No such check was added: three candidates were
measured, and none catches BUG-7 without false positives.

- **State unchanged by `predict_one`.** After learning, 9 of the 15 numeric
  clusterers registered at the time (before the GAP-3 and GAP-4 subclasses)
  change their pickled state on the first `predict_one`, because they build a
  lazy macro-clustering. A plain "state unchanged" check would need an
  allowance per algorithm.
- **A repeat call changes nothing.** None of them, DenStream with BUG-7
  included, changes state on an immediate repeat call, so this check would not
  catch BUG-7 either.
- **Predicting before each learn leaves later answers unchanged.** This
  compares a model that predicts before each `learn_one` with one that only
  learns. DenStream (`n_samples_init=50`) gives the same answers either way on
  Phishing, so this check misses BUG-7 too. Meanwhile `MRStream` differs on 137
  of 200 points, because its cached clustering is refreshed only when an
  inspection runs.

BUG-7 is covered by its own tests, in `tests/test_denstream_patched.py`.

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

`benchmark/algorithms/adapter.py` declares `SENTINEL_LABELS = {"DenStream": (-1,)}`,
looked up along the model's class hierarchy so `DenStreamPatched` and
`DenStreamPatchedFast` inherit the entry, and the adapter **always counts** the occurrences in
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

Upstream is still measured as it is. `river.cluster.DenStream` re-runs the scan
on every `predict_one`, and nothing in this framework changes that.
`tests/test_denstream_fast.py` asserts it: exactly P*(P-1) reachability tests
per call, also back to back, so the test fails the day river caches upstream.
`hpo_compare` reports throughput, memory and the micro-cluster count side by
side, and adds a constrained ceiling (report section 3b-iii: best ARI
subject to a throughput floor), so that a quality number bought with compute is
visible as such. `ClustererAdapter.learn_only` lets a warm start skip the
prediction entirely.

For the patched model there is an exact fast variant, `DenStreamPatchedFast`
(`benchmark/algorithms/denstream_fast.py`), registered as
`"DenStreamPatchedFast"` next to `"DenStreamPatched"`. It is a separate class
under its own name; `DenStream` and `DenStreamPatched` are unchanged. It is
`DenStreamPatched` (BUG-7 and BUG-8 fixed) with an incremental offline phase:

- Per micro-cluster object, it keeps the two distance conditions of
  `_is_directly_density_reachable`. In this river, centre and `calc_radius`
  ignore the timestamp.
- It recomputes only the row of a p-micro-cluster that is new or whose
  `(N, last_edit_time)` changed. Each direction of a pair is evaluated in the
  argument order upstream uses and stored separately, because
  `euclidean_distance_dict` is not bit-symmetric when two centres order their
  keys differently.
- It re-evaluates the weights on every call. It finds components by traversing
  the cached edges between heavy p-micro-clusters, over as many hops as a chain
  needs.
- It fills the `_center` memo of the same micro-clusters the reference's scan
  fills: every heavy one when at least two are heavy, and no light one. It
  reads any other centre through a copy. After each call the live
  micro-clusters equal the reference's, memo slot included.
- It rebuilds a merged cluster, with the inherited
  `_generate_clusters_for_labels`, only when one of these changed: the
  component's member tuple, a member's `(N, last_edit_time)`, the merged
  cluster's own `N`, or the first member's `x`. The `x` check exists because of
  BUG-3: a micro-cluster's `x` is the caller's input dict, which can change
  without `N` moving. A reused merged cluster has its centre memo dropped,
  because `centers` hands that dict out and the reference would have built a
  fresh one.
- It tracks objects by identity, never by dict key, because `denstream.py:223`
  and `:231` can reuse the key of a live micro-cluster after a deletion.

It matches `DenStreamPatched` exactly. `tests/test_denstream_fast.py` compares,
with `==` after every step, the return value, `n_clusters`, every merged
cluster (`x` and key order included), and every live p- and o-micro-cluster,
centre memo included. The comparisons cover:

- synthetic streams that exercise merged components, pruning, o->p promotion,
  the key overwrite, light p-micro-clusters, `stream_speed=1`, predict-only and
  learn-only bursts, and shuffled key order;
- crafted chains that need two to four hops from their first member (BUG-8's
  path), and a chain cut and rejoined at a middle link that turns light and
  then heavy again. No natural stream in the suite formed such a chain, and
  neither did 88k randomly fuzzed predictions, so only these fixtures cover
  multi-hop labelling;
- crafted boundaries: edge direction under asymmetric distances; identical
  centres with radius 0; members swapped or reordered while their stamps stay
  the same; a caller editing an input after `learn_one`, inserting into
  `clusters[k]`, or editing a centre that `centers` returned;
- `adapter.process_one`, and `run_once` (identical predictions and scores);
- Bananas, and INSECTS and covtype slices. On INSECTS and covtype with every
  feature, the norm-based radius is 0 (BUG-8), so no pair is ever linked and
  those runs are smoke tests. One- and two-feature projections of both datasets
  do link pairs and merge components;
- pickle, `clone` and `deepcopy`.

Through the registry, `tests/test_algorithms.py` checks that a `run_once` of
`"DenStreamPatchedFast"` scores exactly what `"DenStreamPatched"` scores. In the
clusterer-check matrix of `tests/test_clusterer_checks.py` (GAP-1) it fails
only the cell `DenStreamPatched` fails, the inherited BUG-3 aliasing, with the
same exception type (and, when measured, at the same line).

One thing is not covered. A caller who edits the sums or times of
`clusters[k]` in place without changing its `N` gets the edited object back on
the next call, because merged clusters are reused while the reference builds
fresh ones. Edits to its `N` or `x`, and to a centre returned by `centers`, are
caught.

It is **not** identical to upstream `DenStream`. BUG-8 makes upstream's
labelling a different function, and BUG-7 makes its predictions mutate state,
so caching upstream would change its results. This variant therefore cannot be
used to time upstream DenStream. A speed number from it describes
`DenStreamPatched`.

Measured on 2026-10-06 against vendored river `873fa6e0`, on a shared machine:

- **Test count.** `DenStream` and `DenStreamPatched` both do exactly P*(P-1)
  reachability tests per `predict_one`. The fast variant does P*(P-1) once
  after the initial DBSCAN. After that it does at most 2*(P-1) per
  test-then-train prediction, and none for a repeated prediction.
- **Per `predict_one`**, at P 45-52 on a synthetic 1-D stream: 1.07-1.18 ms
  against 0.075-0.089 ms, a speed-up of 12.9-14.4x. About 0.008 ms of the fast
  time comes from dropping the centre memo of reused merged clusters, because
  `_get_closest_cluster_key` then recomputes those centres, as it does for the
  reference's fresh clusters.

End-to-end `run_once` (predict + learn), with identical predictions in every
case:

| stream | speed-up | peak model memory |
|---|---|---|
| synthetic, P≈50, 3000 samples | 14.2-14.5x | +27% |
| Bananas, `epsilon=0.05`, mean P 37, 5300 samples | 8.7-9.5x | +26% |
| INSECTS abrupt_balanced, `epsilon=0.1`, P≈7, 6000 samples | 2.3-2.4x | +0% |

The memory column is the cache, which memory probes count. Speed-up was
measured only up to P≈53.

Two consequences of registering it:

- Choosing the variant is explicit, and it replaces nothing. A default
  `run_benchmark.py` run (`algorithms_for()`) includes it next to
  `DenStreamPatched`: identical predictions under two names, and the cost paid
  twice. Pass `--algorithms` to run one of them.
- BUG-6's sentinel count covers it. Until 2026-10-06 `adapter.SENTINEL_LABELS`
  was looked up by the exact class name `"DenStream"`, so on the BUG-6 repro,
  where all three classes return `-1` twice in 200 predictions, the adapter
  counted 2, 0 and 0. `adapter.sentinel_labels` now walks the class's MRO;
  `tests/test_sentinel_predictions.py` checks all three. No shipped result used
  `DenStreamPatched` before the change.

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
describing it as CluStream therefore overstates what was measured, and the
benchmark runner never passes a horizon, so a row for either `CluStream` or
`CluStreamPyramidal` measures clustering over the current micro-cluster set
only. Horizon queries can be tested here only through `CluStreamPyramidal`'s
own methods (see Workaround here).

### Suggested upstream fix

Cheapest correct action: delete the two sentences from the docstring and state
that the implementation keeps the current micro-cluster set only. Implementing
the frame is a larger change and would add `alpha` and `l` to the signature.

### Workaround here

`benchmark/algorithms/clustream_pyramidal.py` adds the frame in a subclass,
`CluStreamPyramidal(river.cluster.CluStream)`. It is registered as
`"CluStreamPyramidal"` next to `"CluStream"`, so `"CluStream"` stays River's
class exactly as shipped (AGENTS.md rule 1.3).

* **The online phase is River's code.** `learn_one` and
  `_maintain_micro_clusters` call `super()` and only observe which key River
  replaced or merged into, to keep id lists. A lockstep test over 3,000 samples
  (48 deletions, 445 merges, horizon queries interleaved) finds identical
  micro-cluster statistics, macro-centres and `predict_one(x)` labels. It runs
  twice: with unit weights, as the framework learns, and with sample weights
  drawn from {0.5, 1, 2, 3}, which River accepts. The counters `n_deletions` and
  `n_merges` follow River's two maintenance paths, checked against River's
  relevance-stamp rule. With `max_micro_clusters=1`, River's merge path adds the
  one micro-cluster to itself and replaces it; that counts as a merge.
* **New parameters:** `alpha` (int >= 2), `l` (int >= 1) and
  `snapshot_interval` (learned samples per clock tick). The defaults 2, 1 and
  100 are runnable placeholders picked for a small store, not tuned values.
* **Snapshots.** Every tick stores `(tick, order, {key: (frozenset(ids), copies
  of var_x and var_time)})`. Retention follows the paper: for each order `i`,
  keep the last `alpha**l + 1` ticks divisible by `alpha**i`, and store the
  union once. This is checked at every one of 1,200 ticks, for four
  `(alpha, l)` pairs, against an independent reference.
* **Horizon queries.** `micro_clusters_for_horizon(h)` subtracts the last stored
  snapshot at or before `t_c - h` through id lists. It returns the horizon
  micro-clusters, as the caller's own copies, together with `covered`
  (`t_c - t_s`), `snapshot_time` and `whole_stream`, so the approximation is
  visible. `predict_one(x, horizon=h)` labels the nearest horizon micro-cluster
  with a k-means built like River's offline phase: a fresh
  `KMeans(n_macro_clusters, seed, **kwargs)`, unweighted, one pass over the
  horizon centres. The horizon k-means is fitted on demand. River fits its own
  only at `time_gap` boundaries, and not before the first boundary after
  initialisation; until then River answers 0. A test with a KMeans option that
  changes the labels (`sigma=0.3`) checks the following for a horizon that
  reaches past the stream start:
  - it reproduces River's labels at every `time_gap` boundary once River has
    fitted (10 of 10 boundaries);
  - it already gives several labels while River still answers 0;
  - it can differ in between, where River's k-means is stale.

  `predict_one(x)` without a horizon is River's.
* **Lemma.** On the tick clock, `t_c - t_s <= (1 + 1/alpha**(l-1)) * h` for
  every horizon of at least one tick. Off the tick grid it holds strictly for
  `h >= snapshot_interval * alpha**(l-1)`; below that only
  `t_c - t_s < h + snapshot_interval` holds. This is checked exhaustively over
  every `(t_c, h)` up to `t_c = 240`.
* **Exactness.** A test-side ledger records the points each live micro-cluster
  holds, with their weights and the timestamp River wrote into `var_time`. River
  stamps every point of its initialisation branch with
  `max_micro_clusters - 1`. Against the ledger, the horizon micro-clusters hold
  exactly the points learned after `t_s`, merged micro-clusters included:
  - weight exact;
  - per-feature mean: relative error <= 4e-14 (unit weights) and <= 3e-13
    (weighted); per-feature variance <= 2.5e-9;
  - mean and variance of the timestamps, which `relevance_stamp` reads:
    relative error <= 1e-14 and <= 1.5e-9.

  Snapshots taken inside River's initialisation branch are checked separately,
  at every horizon of every sample of a short stream, past the end of
  initialisation and through merges. River's `Var.__isub__` often leaves a
  second moment slightly below zero, in `var_time` as in `var_x`. When one point
  remains, `Var.get()` hides it. With two identical points remaining, `radius`
  raises in `math.sqrt`. Such values are clamped to 0, and the tests check the
  stored second moment, not only `get()`.
* **Behaviour, and the condition it depends on.** The stream is 6,000 regime-A
  samples, then 2,000 regime-B samples. With `l=3` the horizon of 2,000 is
  covered exactly; with `l=1` it covers 3,200. A horizon only differs from the
  current set while that set still holds the old regime, so the effect depends
  on `time_window`:
  - With `time_window` longer than the stream (no deletions), the horizon
    weight equals the covered span, and only `covered - h` of it lies near A.
    On held-out B points, ARI with the horizon is at least ARI without it on 5
    of 5 seeds, and higher on 3 (+0.57, +0.57, +0.39).
  - With `time_window=3000` the gains are 0, 0, +0.46, +0.12 and 0.
  - At River's default `time_window=1000`, River's own deletions have removed
    every regime-A micro-cluster by the end of the stream on 5 of 5 seeds.
    Nothing is subtracted, the horizon micro-clusters are the current ones, and
    both predictions are identical (gain 0).

  Tests pin the no-deletion case and the default case.

**Not implemented:**

- horizons that end before `t_c` (the paper's evolution analysis);
- the paper's weighted offline k-means;
- the paper's disk-resident snapshot store.

The runner calls `predict_one(x)` without a horizon. A benchmark row for
`CluStreamPyramidal` therefore measures River's CluStream clustering plus the
snapshot cost; any claim about horizons needs explicit
`micro_clusters_for_horizon(h)` or `predict_one(x, horizon=h)` calls. Through
the registry, `tests/test_algorithms.py` checks that a `run_once` of
`"CluStreamPyramidal"` scores exactly what `"CluStream"` scores. In the
clusterer-check matrix of `tests/test_clusterer_checks.py` (GAP-1) it fails
only the two cells `CluStream` fails, BUG-3 aliasing and BUG-4 emerging
features, with the same exception types (and, when measured, at the same
lines).

**Cost** is given as orders of magnitude, not a benchmark: three runs over two
synthetic Gaussian streams, CPython 3.13, a shared machine, placeholder
defaults. With 30,000 samples of 54 features there are 12 snapshots and
14.7-15.7 MB, against 1.8 MB for `CluStream`, and learning is 6-16% slower. Id
lists shrink only through deletions. With a `time_window` longer than the
stream they hold every id ever issued: about 24,800 ids after 30,000 samples of
10 features, in a 14.2 MB model against 0.3 MB.

Tests: `tests/test_clustream_pyramidal.py`, run with `PYTHONHASHSEED=0`. Its
upstream test asserts that the gap is still present in the vendored tree.

---

## BUG-7 — `DenStream.predict_one` rewrites the sums of a live p-micro-cluster whenever two of them share a DBSCAN label

**Severity:** medium — it corrupts the model's own state from a method that is documented as a query, but only on the path that BUG-8 makes rare.
**Status:** reproduced on unmodified river (2026-09-26).
**Location:** [`river/cluster/denstream.py:274-276`](../river/river/cluster/denstream.py) (`_generate_clusters_for_labels`), with `DenStreamMicroCluster.merge` at `:503`.

### Symptom

Force two real p-micro-clusters under one label and call the grouping step
that `predict_one` runs on every sample:

```python
>>> import random
>>> from river import cluster
>>> random.seed(1)
>>> m = cluster.DenStream(decaying_factor=0.01, beta=0.75, mu=2, epsilon=0.8, n_samples_init=100)
>>> for _ in range(800):
...     m.learn_one({"a": random.gauss(0, 0.6), "b": random.gauss(0, 0.6)})
>>> mcs = list(m.p_micro_clusters.items())
>>> k0, k1 = mcs[0][0], mcs[1][0]
>>> before = (m.p_micro_clusters[k0].N, dict(m.p_micro_clusters[k0].linear_sum))
>>> labels = {mc: (0 if k in (k0, k1) else 1 + i) for i, (k, mc) in enumerate(mcs)}
>>> _ = m._generate_clusters_for_labels(labels)
>>> after = (m.p_micro_clusters[k0].N, dict(m.p_micro_clusters[k0].linear_sum))
>>> before[0] == after[0], before[1] == after[1]
(True, False)
```

Measured: `N` stays 145 while `linear_sum` moves from `{a: 77.75, b: 69.08}`
to `{a: 63.22, b: -56.21}` — the centre of a micro-cluster the model keeps
learning into has jumped from (0.54, 0.48) to (0.44, -0.39) because of a
prediction.

### Root cause

```python
cluster = copy.copy(micro_clusters[0])
for mc in range(1, len(micro_clusters)):
    cluster.merge(micro_clusters[mc])
```

`copy.copy` on a `__slots__` object shares every attribute by reference, so the
copy's `linear_sum` and `squared_sum` are the *same dicts* as the first
member's. `merge` then does `linear_sum[key] += val` in place, and adds the
neighbours' sums into the live p-micro-cluster while its `N` is untouched.
Every further `predict_one` adds them again.

### Impact on benchmarking

Any window in which DenStream's DBSCAN puts two p-micro-clusters under one
label degrades the model's own centres, radii and weights on every prediction
that follows — test-then-train calls `predict_one` once per sample. The effect
is invisible in the metrics (nothing raises) and grows with the number of
predictions between learns. In practice the path fires rarely, because of
BUG-8 below, which is the only reason it has not shown up in the matrices.

### Suggested upstream fix

`copy.deepcopy(micro_clusters[0])`, or build the merged cluster from a fresh
`DenStreamMicroCluster` and merge every member into it.

### Workaround here

`DenStreamPatched` (`benchmark/algorithms/denstream_patched.py`, registered
under its own name per AGENTS.md rule 1.3) overrides
`_generate_clusters_for_labels` to deep-copy the first member before merging.
`tests/test_denstream_patched.py` shows that upstream `DenStream` mutates a live
p-micro-cluster and the subclass does not. Upstream stays registered as
`DenStream`, untouched, so the two can be benchmarked against each other;
`DenStreamPatchedFast` (GAP-3) builds on the subclass. `conf_pipeline` records `n_macro_clusters` next to
`n_pred_labels` per window, which is where a run that did take the path would
show.

---

## BUG-8 — the DBSCAN inside `DenStream.predict_one` never expands past a direct neighbour

**Severity:** medium — DenStream's macro-clusters are, in practice, its p-micro-clusters.
**Status:** reproduced on unmodified river (2026-09-26).
**Location:** [`river/cluster/denstream.py:351-389`](../river/river/cluster/denstream.py) (`predict_one`), with `_is_directly_density_reachable` at `:242`.

### Symptom

Five configurations (`epsilon` 0.3–1.5, `mu` 2–3) over one Gaussian blob of
800 points, `n_samples_init=100`, produced 3–10 p-micro-clusters each and, on
every one of them, `n_clusters == len(p_micro_clusters)` after `predict_one`.

### Root cause

Two things, in the seed-queue loop:

```python
neighbor_neighbors = self._query_neighbor(seed_queue[0])
for neighbor_neighbor in neighbor_neighbors:
    if labels[neighbor_neighbor] is not None:   # inverted
        seed_queue.append(neighbor_neighbor)
```

Only *already labelled* neighbours are queued, so a label never travels beyond
the first ring of direct neighbours of the seed. And the reachability test
`distance <= r_p + r_q` uses `calc_radius`, whose norm-based formula (the
paper's) returns 0 for micro-clusters whose sums are not centred at the origin
— on the blob above, two of five radii were exactly 0 — so most pairs never
qualify in the first place.

### Impact on benchmarking

`final_macro_clusters`, `n_macro_clusters` and every "number of clusters"
column reads DenStream's micro-cluster count. Its ARI is the ARI of the
micro-cluster partition. Neither is wrong as a measurement of River; both are
wrong as a description of DenStream.

### Suggested upstream fix

`if labels[neighbor_neighbor] is None:`; and decide whether `calc_radius` is
meant to be the paper's formula (then document that it can be 0) or the
per-dimension variance one that CluStream uses.

### Workaround here

`DenStreamPatched` overrides `predict_one` with the seed expansion fixed
(`if labels[neighbor] is None`); `tests/test_denstream_patched.py` shows a
density-reachable chain becoming one cluster in the subclass and several
upstream. The `calc_radius` question is left as upstream has it. `conf_pipeline`
and the HPO matrix both carry
`n_pred_labels`/`n_macro_clusters` per window so the reading is visible.

---

## GAP-5 — the k-means inside `CluStream` and `STREAMKMeans` starts its centres at N(0, 1), whatever the data's scale

**Severity:** high for any benchmark that does not standardise its streams —
the macro phase of River's CluStream then reports one cluster, and the
algorithm reads as broken when its micro-cluster phase is fine.
**Status:** measured on unmodified river.
**Location:** [`river/cluster/k_means.py:90-101`](../river/river/cluster/k_means.py)
(centres drawn lazily from `gauss(mu, sigma)`, defaults `mu=0`, `sigma=1`),
used by [`clustream.py:257`](../river/river/cluster/clustream.py) and
[`streamkmeans.py:84,101`](../river/river/cluster/streamkmeans.py).

### Symptom

Three well-separated Gaussian blobs, the same layout multiplied by `scale`,
3 000 points learned and 900 held-out points predicted, `seed=1`:

| | scale | `sigma` | labels | ARI |
|---|---|---|---|---|
| `CluStream(n_macro_clusters=3, max_micro_clusters=30)` | 0.25 | default (1) | 1 | 0.000 |
| | 0.25 | 0.025 | 3 | 1.000 |
| | 1 | default | 2 | 0.571 |
| | 1 | 0.1 | 3 | 1.000 |
| | 4 | default | 2 | 0.571 |
| | 4 | 0.4 | 3 | 1.000 |
| `STREAMKMeans(n_clusters=3, chunk_size=100)` | 0.25 | default | 1 | 0.000 |
| | 0.25 | 0.025 | 2 | 0.571 |

With `sigma` proportional to the data's scale, CluStream recovers the blobs at
every scale; at River's default it collapses below scale 1. STREAMKMeans is
lifted off the single label but stays at two on this layout, which is a second,
separate limit of its chunked k-means and is not claimed fixed here.

### Root cause

`KMeans.learn_one` moves only the closest centre, by `halflife` of the way to
the point. The centres start at N(`mu`, `sigma`) in absolute units, so when
the data's spread is small against `sigma`, or the data sit far from `mu`, one
centre is the nearest to every point, absorbs all of them, and the others never
move. CluStream's offline phase is exactly this `KMeans` run over the
micro-cluster centres every `time_gap` samples, and STREAMKMeans runs it per
chunk and over the chunk centres. Neither signature exposes `mu` or `sigma`;
both reach the inner `KMeans` only through `**kwargs`.

### What it costs a benchmark

In the OTB pilot (2026-10-01, no scaler, by protocol) CluStream at River
defaults gave a lagged ARI of 0.001 on `k_drift` while the common reclusterer
on its own micro-clusters (Table A, k-means) read 0.994: the online phase was
fine and the macro phase put everything in one cluster. On the last third of
`scale_drift`, where the features are multiplied by 0.25, every CluStream
configuration the arms ran collapsed: mean lagged ARI 0.002 to 0.014 per arm
there, against 0.68 to 0.92 for the tuned arms on the first two thirds of the
same stream, because none of them could set `sigma`.

### Suggested upstream fix

Initialise the centres from the data, for example from the first
`n_clusters` distinct points seen (or k-means++ over a first buffer), instead
of N(`mu`, `sigma`). Failing that, document in CluStream and STREAMKMeans that
`mu`/`sigma` must match the data's scale and expose them in their signatures.

### Workaround here

`conf_pipeline/spaces.py` tunes `sigma` for CluStream and STREAMKMeans as a
`scaled` parameter, 0.01 to 1 times `s` on a log scale, so it follows the
stream's spread like every other distance parameter. The start stays River's
`sigma=1`, so the `default` arm is unchanged.
`tests/test_conf_pipeline.py::test_sigma_in_units_of_s_lifts_clustream_off_a_single_label_at_small_scale`
shows River's default giving one label and the middle of the range recovering
the blobs.

Measured limits of that workaround, on windows of the pilot folds (probe, not a
test): on `scale_drift` and `k_drift` `sigma` between 0.05 and 0.5 times `s`
works and 1 to 2 times `s` collapses again, which is why the range stops at 1.
On raw `covtype` and INSECTS no scalar `mu` or `sigma` helps: their features
sit on scales two to four orders of magnitude apart, and every coordinate is
drawn from the same N(`mu`, `sigma`). The protocol does not rescale streams, so
those cells keep the collapse and are read as such, not as a property of the
online phase.
