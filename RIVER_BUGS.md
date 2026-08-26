# Suspected bugs in river

Findings from building this benchmark. **The river tree is never modified** (see
[AGENTS.md](AGENTS.md)) — each entry records the problem, a minimal reproduction,
the root cause, what it costs a benchmark, a suggested upstream fix, and the
workaround used here.

Verified against the river source tree at `../river`, version `0.26.1`
(distribution metadata reports `0.26.0`), Python 3.13.3, Windows 11.

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

## Notes

- Both bugs were originally patched locally in the river tree during development
  and both patches were reverted, at which point the bugs returned. That is the
  reason for the no-edits rule in [AGENTS.md](AGENTS.md).
- Neither has been reported upstream yet. Both are worth filing at
  <https://github.com/online-ml/river/issues>.
