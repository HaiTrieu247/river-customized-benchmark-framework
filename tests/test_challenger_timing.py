"""The order in which a challenger enters the tuner (supplementary Algorithm S1), checked on the code
itself with models that record every call they receive: created after window t-1 and warm-started on
it with learning calls only, first scored on window t (every point predicted before it is learned),
admitted after window t, incumbent from window t+1; no model learns an observation twice."""

from __future__ import annotations

import numpy as np

from benchmark.algorithms.adapter import make_adapter
from benchmark.tuning.confstream import ConfStreamEnsemble, EnsembleConfig
from benchmark.tuning.space import Param, ParamSpace, SpaceContext
from benchmark.tuning.window import WindowBuffer

SPACE = ParamSpace("KMeans", (Param("n_clusters", "log_int", 2, 8), Param("halflife", "log", 0.01, 0.9)))
H = 20


class _Recorder:
    """Two clusters by the sign of ``a``; every live call is logged as ``(kind, point id)``. Copies (the
    snapshots) log into their own copy of the list, so the log is the live model's alone."""

    def __init__(self):
        self.log: list = []

    def learn_one(self, x):
        self.log.append(("learn", int(x["i"])))

    def predict_one(self, x):
        self.log.append(("predict", int(x["i"])))
        return int(x["a"] > 0)


def _window(w: int) -> WindowBuffer:
    rng = np.random.default_rng(w)
    buf = WindowBuffer(H)
    for k in range(H):
        side = k % 2
        buf.push({"a": (1.0 if side else -1.0) + 0.01 * rng.standard_normal(), "i": float(w * H + k)}, side)
    return buf


def _window_of(i: int) -> int:
    return i // H


def test_a_challenger_is_warm_started_on_t_minus_1_scored_on_t_admitted_after_t_and_serves_t_plus_1():
    cfg = EnsembleConfig(e_size=2, e_test=1, lam=0.05, p_reset=0.0, runtime_factor=None)
    models: dict[int, _Recorder] = {}

    def make(params, seed):
        return make_adapter("rec", _Recorder(), {})

    ens = ConfStreamEnsemble(space=SPACE, ctx=SpaceContext(s=1.0, d=2), make_adapter=make,
                             start_params=[{"n_clusters": 2, "halflife": 0.5}], config=cfg,
                             rng=np.random.default_rng(0), seed=0)
    ens.initialise(_window(0), window_index=0)
    reports = {}
    challenger = None
    for w in range(1, 5):
        if w == 2:
            # the challenger built after window 1 is made to win its match on window 2
            challenger = ens.test[0]
            challenger.outcome_override = 10.0
        reports[w] = ens.process_window(_window(w), window_index=w)
        for m in ens.active + ens.test:
            models[m.id] = m.adapter.model
    born = challenger.born_window
    log = models[challenger.id].log
    learned = [i for kind, i in log if kind == "learn"]
    predicted = [i for kind, i in log if kind == "predict"]

    assert born == 1                                                     # created after window t-1 = 1
    first = [i for i in learned if _window_of(i) == born]
    assert first == list(range(born * H, (born + 1) * H))                # warm start: all of window t-1 ...
    assert not any(_window_of(i) == born for i in predicted)             # ... with learning calls only
    assert min(_window_of(i) for i in predicted) == born + 1             # first predicted (scored) window: t
    scored = [w for w, r in reports.items() if any(o.member_id == challenger.id for o in r.outcomes)]
    assert scored[0] == born + 1                                         # no score on its warm-start window
    # within window t every point is predicted before it is learned
    order = [(kind, i) for kind, i in log if _window_of(i) == born + 1]
    assert order == [step for i in range((born + 1) * H, (born + 2) * H) for step in (("predict", i), ("learn", i))]
    # admitted after window t, incumbent named after window t, so the output it serves is window t+1's
    assert reports[born + 1].incumbent.member_id == challenger.id
    assert reports[born + 1].lag.member_id != challenger.id
    assert reports[born + 2].lag.member_id == challenger.id
    # no model ever learns an observation twice
    for model in models.values():
        ids = [i for kind, i in model.log if kind == "learn"]
        assert len(ids) == len(set(ids))
