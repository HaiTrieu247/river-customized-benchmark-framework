"""A search space in unit coordinates, and the confStream offspring sampler.

Every parameter is declared with a *kind* and a range, and lives on the unit
interval ``u in [0, 1]`` inside the tuner. The kind decides how ``u`` maps onto
the constructor value:

* ``lin``       -- ``lo + (hi - lo) u``
* ``log``       -- ``lo (hi / lo) ** u``, for parameters that live on a
                   multiplicative scale (rates, thresholds)
* ``log_int``   -- ``log`` rounded to an integer (cluster caps, neighbour counts)
* ``choice``    -- an index into a tuple of values
* ``scaled``    -- ``log`` times the data's distance unit ``s`` (a threshold in
                   the data's own units; see ``hpo_pipeline.probe``)
* ``frac_of_d`` -- ``lin`` times the number of features, rounded, at least 1

Sampling happens in ``u``-space, which is what makes the search *fair across
parameters*: confStream places a truncated normal linearly on the raw range, so
a parameter spanning three decades would almost never be sampled near its
small end. Here every parameter is explored at the same resolution on its own
scale, and the same sigma means the same thing for all of them.

The sampler follows confStream (Carnein et al., 2020, Sec. 3): a truncated
normal centred on the parent with a per-parameter standard deviation that
starts at half the range and shrinks by ``2 ** -lam`` in the offspring;
categorical parameters carry a probability vector whose drawn entry grows by
``2 - 2 ** -lam``. A state can be reset to its initial exploration width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import truncnorm

KINDS = ("lin", "log", "log_int", "choice", "scaled", "frac_of_d")
#: Kinds that are sampled with the truncated normal (everything but ``choice``).
NUMERIC_KINDS = ("lin", "log", "log_int", "scaled", "frac_of_d")
#: Sigma every numeric parameter starts at: half the (unit) range.
INITIAL_SIGMA = 0.5


@dataclass(frozen=True)
class SpaceContext:
    """What a value map needs from the data: its distance unit and dimension."""

    s: float | None = None
    d: int | None = None


@dataclass(frozen=True)
class Param:
    name: str
    kind: str
    lo: float | None = None
    hi: float | None = None
    values: tuple | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}; one of {KINDS}")
        if self.kind == "choice":
            if not self.values:
                raise ValueError(f"{self.name}: a choice needs at least one value")
            return
        if self.lo is None or self.hi is None or not self.lo < self.hi:
            raise ValueError(f"{self.name}: need lo < hi, got {self.lo!r}, {self.hi!r}")
        if self.kind in ("log", "log_int", "scaled") and self.lo <= 0:
            raise ValueError(f"{self.name}: a log range must stay above 0, got lo={self.lo}")

    # -- unit interval -> value ------------------------------------------------
    def value(self, u: float, ctx: SpaceContext):
        u = min(max(float(u), 0.0), 1.0)
        if self.kind == "lin":
            return float(self.lo + (self.hi - self.lo) * u)
        if self.kind == "log":
            return float(self.lo * (self.hi / self.lo) ** u)
        if self.kind == "log_int":
            return int(round(self.lo * (self.hi / self.lo) ** u))
        if self.kind == "choice":
            return self.values[min(int(u * len(self.values)), len(self.values) - 1)]
        if self.kind == "scaled":
            if ctx.s is None:
                raise ValueError(f"{self.name}: a scaled parameter needs the distance unit s")
            return float(self.lo * (self.hi / self.lo) ** u * ctx.s)
        if self.kind == "frac_of_d":
            if ctx.d is None:
                raise ValueError(f"{self.name}: frac_of_d needs the feature count d")
            return max(1, int(round((self.lo + (self.hi - self.lo) * u) * ctx.d)))
        raise AssertionError(self.kind)  # pragma: no cover

    # -- value -> unit interval (clipped) ---------------------------------------
    def unit_of(self, value, ctx: SpaceContext) -> float:
        if self.kind == "choice":
            if value not in self.values:
                raise ValueError(f"{self.name}: {value!r} is not one of {self.values}")
            i = self.values.index(value)
            return (i + 0.5) / len(self.values)
        if self.kind == "lin":
            u = (float(value) - self.lo) / (self.hi - self.lo)
        elif self.kind in ("log", "log_int"):
            u = math.log(float(value) / self.lo) / math.log(self.hi / self.lo)
        elif self.kind == "scaled":
            if ctx.s is None:
                raise ValueError(f"{self.name}: a scaled parameter needs the distance unit s")
            u = math.log(float(value) / ctx.s / self.lo) / math.log(self.hi / self.lo)
        else:  # frac_of_d
            if ctx.d is None:
                raise ValueError(f"{self.name}: frac_of_d needs the feature count d")
            u = (float(value) / ctx.d - self.lo) / (self.hi - self.lo)
        return min(max(u, 0.0), 1.0)


@dataclass(frozen=True)
class ParamSpace:
    """The tunable parameters of one algorithm, plus the ones pinned for it."""

    algo: str
    params: tuple[Param, ...]
    fixed: dict = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    @property
    def dim(self) -> int:
        return len(self.params)

    def resolve(self, u: dict, ctx: SpaceContext) -> dict:
        """Unit point -> constructor kwargs, fixed values included."""
        out = {p.name: p.value(u[p.name], ctx) for p in self.params}
        out.update(self.fixed)
        return out

    def unit_point(self, params: dict, ctx: SpaceContext) -> dict:
        """Constructor kwargs -> unit point over the tunable dimensions only."""
        return {p.name: p.unit_of(params[p.name], ctx) for p in self.params}


@dataclass
class SamplerState:
    """Per-member exploration width: a sigma per numeric parameter, a probability
    vector per categorical one."""

    sigma: dict
    probs: dict

    @classmethod
    def initial(cls, space: ParamSpace) -> "SamplerState":
        sigma = {p.name: INITIAL_SIGMA for p in space.params if p.kind in NUMERIC_KINDS}
        probs = {
            p.name: np.full(len(p.values), 1.0 / len(p.values))
            for p in space.params if p.kind == "choice"
        }
        return cls(sigma=sigma, probs=probs)

    def reset(self, space: ParamSpace) -> "SamplerState":
        return self.initial(space)

    def offspring(self, parent: dict, rng: np.random.Generator, lam: float):
        """Draw a child unit point around ``parent``; returns ``(child, child_state)``.

        Numeric parameters: truncated normal on [0, 1], mean at the parent,
        standard deviation the parent's sigma; the child's sigma is the parent's
        times ``2 ** -lam`` (Eq. 2). Categorical parameters: one draw from the
        probability vector; the drawn entry is multiplied by ``2 - 2 ** -lam``
        and the vector renormalised (Eq. 3). The parent's state is not touched.
        """
        shrink = 2.0 ** -lam
        child: dict = {}
        sigma: dict = {}
        probs: dict = {}
        for name, sd in self.sigma.items():
            mean = min(max(float(parent[name]), 0.0), 1.0)
            if sd <= 0:
                child[name] = mean
            else:
                a, b = (0.0 - mean) / sd, (1.0 - mean) / sd
                child[name] = float(truncnorm.rvs(a, b, loc=mean, scale=sd, random_state=rng))
            sigma[name] = sd * shrink
        for name, vec in self.probs.items():
            vec = np.asarray(vec, dtype=float)
            drawn = int(rng.choice(len(vec), p=vec / vec.sum()))
            child[name] = (drawn + 0.5) / len(vec)
            grown = vec.copy()
            grown[drawn] *= 2.0 - shrink
            probs[name] = grown / grown.sum()
        return child, SamplerState(sigma=sigma, probs=probs)
