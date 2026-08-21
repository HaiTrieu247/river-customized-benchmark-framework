# River Customized Benchmark Framework

A set of **hand-written** benchmark notebooks for River's clustering module
([River](https://riverml.xyz), `river.cluster`). Goal: reproduce and verify
streaming clustering algorithms on real data, following the exact evaluation
protocol of the original papers — rather than relying on the library's minimal
built-in toy examples.

This repo is a **dedicated benchmark playground**: whenever a new aspect of
`river.cluster` needs measuring (cluster quality, execution time, RAM, behavior on
evolving streams...), add a notebook here instead of editing river's source.

---

## Why this repo exists

River ships many streaming clustering algorithms (CluStream, DenStream, DBSTREAM,
STREAMKMeans...), but its bundled evaluation is very minimal. When you need to
answer questions like:

- Does this algorithm reproduce the **cluster purity** the original paper reports?
- How much **time / RAM** does it cost as the stream grows?
- How do preprocessing (scaler) and parameters (`epsilon`, `mu`, `beta`...) affect
  the number of clusters `K` and purity?
- Does river's implementation **match the paper's description**, or has it been
  adapted for the online setting?

...you need a tightly controlled measurement harness: correct features, correct
stream order, correct purity formula, correct parameters. That is what this repo
does.

---

## Core concepts

### Streaming (online) clustering
There is no separate "train" then "deploy" phase. Each incoming data point is first
**assigned to a cluster** (`predict_one`) and then **learned** (`learn_one`),
exactly like a system running on a live data stream. A dataset is turned into a
stream by taking the **record order in the file as the streaming order** (data
input order) — no shuffling.

## Data & preprocessing

Both datasets are processed exactly as described in the streaming clustering papers.

### KDDCup99 — Network Intrusion Detection
- Uses exactly the **34 continuous attributes** out of 42. Dropped:
  - 3 nominal columns: `protocol_type`, `service`, `flag`
  - 4 binary columns: `land`, `logged_in`, `is_host_login`, `is_guest_login`
  - the label column `labels` (used only as ground truth for purity, never fed to
    clustering)
  - → `42 − 3 − 4 − 1 = 34`.
- Label `y` = connection type (`normal` / various attack types).
- Loaded automatically via `sklearn.datasets.fetch_kddcup99` (no local file needed).

### KDDCup98 — Charitable Donation
- Uses **56 fields** extracted from the 481 fields of each record (the RFM donation
  history set): 22 `RAMNT_3..24` (repeated donation amounts) + 22 `RDATE_3..24`
  (repeated donation dates) + 12 summary fields (`RAMNTALL`, `NGIFTALL`, `MINRAMNT`,
  `MAXRAMNT`, `LASTGIFT`, `AVGGIFT`, `CARDPROM`, `NUMPROM`, `CARDGIFT`, `NUMPRM12`,
  `TIMELAG`, `CARDPM12`).
- Label `y` = `TARGET_B` (0/1: whether the recipient donated).
- Read from the local file `cup98LRN.txt` (481 columns). **Path must be adjusted** —
  see below.

> **Note:** the CluStream paper does not explicitly list the 56 fields, so this is
> the RFM field set commonly used to reconstruct the KDD98 stream. If you have a
> different official list, just edit `KDD98_FIELDS` in the loader.

---

## Environment requirements

- Python 3.10+
- `river` — imported directly from the river source tree next to this repo (see
  *Importing river*).
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `ipywidgets`
- Jupyter (notebook / lab / VS Code) — the wizard needs a kernel with `ipywidgets`
  support.

### Importing river
The `benchmark_cluster_initial.ipynb` notebook **does not install river via pip**;
it inserts the river source tree root into `sys.path` to import directly from
source:

```python
_repo_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, _repo_root)
```

Expected directory layout (the benchmark repo sits **next to** the river tree):

```
OnlineML/
├── river/                              # river source tree (contains the river/ package)
└── river-customized-benchmark-framework/
    ├── benchmark_cluster_initial.ipynb
    └── evaluate_river_resource_benchmark.ipynb
```

If your layout differs, adjust `_repo_root` to point at the directory containing the
`river` package. (The resource notebook uses the river installed in the environment,
so it does not need this step.)

---

## How to run

### 1. `benchmark_cluster_initial.ipynb` (cluster quality — with wizard)
1. Edit `KDD98_PATH` in the data loader cell to point at your `cup98LRN.txt` (KDD99
   is fetched automatically).
2. Check that `_repo_root` points at the correct river tree.
3. Run the cells in order until the wizard opens, then use the widget UI:
   - **Step 1 — Choose algorithm:** lists **every** clusterer in `river.cluster`
     (auto-discovered via `base.Clusterer`), pick by number.
   - **Step 2 — Enter params:** input fields auto-generated from the algorithm's
     `__init__`, pre-filled with a preset when available.
   - **Step 3 — Configure scenarios:** each scenario is an independent bundle of
     *dataset · max samples · horizon · stream speed · scaler · chart type (purity /
     exec-time) · marks*.
   - **Step 4 — Run & plot:** only plots the chart type selected per scenario.

   You can add the same algorithm class multiple times with different params
   (suffixed `#2`, `#3`...) to compare configurations.

### 2. `evaluate_river_resource_benchmark.ipynb` (time & RAM)
Run top to bottom. Adjustable:
- `N_SAMPLES` — number of streamed points (`None` = the full 567,498 samples of
  KDDCup99 HTTP).
- `N_CHECKPOINTS` — number of `(time, memory)` measurements per model.

Output: **execution time vs samples**, **RAM vs samples**, and a **summary table**.
`Memory` is measured via `model._raw_memory_usage` (the size of the model object,
not the whole process).

---

## Reference parameters (from the original papers)

Pre-filled in the presets to reproduce the experiments:

- **DenStream** (Cao et al. 2006): `InitN=1000`, `v (stream_speed)=1000`,
  `λ (decaying_factor)=0.25`, `ε (epsilon)=16`, `µ (mu)=10`, `β (beta)=0.2`.
- **CluStream** (Aggarwal et al., VLDB 2003): `q=50` micro-clusters (`=10·k`),
  `t=2`, `k=5`, `stream_speed=200`, macro-clustering every time unit
  (`time_gap=200`).

---

## Things to keep in mind when reading results

- **The scaler matters a lot.** KDD99 has a few features with an extreme range
  (`src_bytes`, `dst_bytes` up to the millions). `MinMaxScaler` divides by the max,
  squashing nearly every point to ~0 → everything merges into a single
  micro-cluster (`K=1`). Use `StandardScaler`.
- **`epsilon` must match the scaler.** After `StandardScaler` (std≈1), an `epsilon`
  that is too large (e.g. 16) makes the **online** phase merge every point into one
  micro-cluster → `K=1` regardless of what the offline phase does. `epsilon` must be
  small enough to create multiple micro-clusters first.
- **River's DenStream is an online-adapted implementation**, not the original
  DBSCAN: the offline phase runs DBSCAN over **micro-cluster centers** with its own
  density-reachability definition (weight > `mu`, distance < `2ε` and ≤ the sum of
  radii). Account for this difference when comparing against the paper.

---

## Extending the repo

Adding a new benchmark = adding a notebook, keeping the same conventions:
- Stream in **data input order**, no shuffling.
- Purity uses the macro-average formula (per-cluster dominant-class ratio, averaged
  over `K`).
- State the scaler + parameters at the top of the notebook so results are
  reproducible.
- Be explicit about what is measured (quality / time / RAM) and what is ignored.
