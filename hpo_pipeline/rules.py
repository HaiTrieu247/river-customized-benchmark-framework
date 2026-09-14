"""Stage 6.1 -- the four selection rules.

A rule maps an 8-point label-free response curve to a choice of resolution:

    g_hat = R( { (g, stats(a, g, theta, d)) : g = 1..8 } ; phi )

``stats`` is label-free by construction, so a rule is something a real system
could actually run -- which is what separates this from an oracle.

``R_const`` needs no function here: it *is* standard LODO, the zero-order rule
that ignores the curve and carries a fixed level over from the optimisation set.
That makes P2R a generalisation of P2 rather than a replacement for it, and it
is why ``R_const`` belongs in the same table as the other three: same matrix,
same folds, same budget.

``R_parsimony`` is the MATR-CV rule, ``min{r : l_r >= max l - Delta}``. The one
difference is the important one: ``Delta`` is not set by hand here, it is
``phi``, fitted by LODO on the optimisation set and tested on the held-out fold.
That is precisely the gap Fan et al. leave open.

Written down before running, so the result cannot be read as post-hoc: on
``matr_trace`` the criterion is a negated WCSS and rises almost monotonically
with the cluster count, so ``R_max`` will always pick ``g = 8`` and lose badly.
That is not a bug, it is the very failure ``R_parsimony`` exists to prevent. On
``silhouette``, which is not monotone in k, ``R_max`` should not degenerate --
and that contrast is itself the result.
"""

from __future__ import annotations

import numpy as np

from ._common import SMALLER_IS_BETTER

#: Candidate values of the free parameter of R_parsimony, fitted by LODO.
PHIS = (0.01, 0.02, 0.05, 0.10, 0.20)


def orient(values, criterion: str):
    """Put every criterion on a "bigger is better" footing.

    River's streaming silhouette is a cohesion/separation ratio and its SSQ is a
    distance, so both are negated; ``matr_trace`` is already a negated WCSS.
    """
    v = np.asarray(values, dtype=float)
    return -v if criterion.split("_win")[0] in SMALLER_IS_BETTER else v


def r_max(curve, phi=None) -> int:
    """Plain argmax. The strawman that shows why a rule is needed at all."""
    return int(np.argmax(curve))


def r_parsimony(curve, phi) -> int:
    """The COARSEST level still within ``phi * range`` of the peak.

    "Coarsest" is "smallest index" only because of the Stage 3 direction
    convention. If that convention is broken this rule silently inverts.
    """
    curve = np.asarray(curve, dtype=float)
    span = float(curve.max() - curve.min())
    if span <= 0:
        return 0
    within = np.flatnonzero(curve >= curve.max() - phi * span)
    return int(within.min())


def r_knee(curve, phi=None) -> int:
    """Point furthest from the chord joining the curve's two ends.

    The classic elbow, and the parameter-free control.
    """
    curve = np.asarray(curve, dtype=float)
    n = len(curve)
    span = float(curve.max() - curve.min())
    if span <= 0:
        return 0
    x = np.linspace(0.0, 1.0, n)
    y = (curve - curve.min()) / span
    # The denominator of the point-line distance is constant along the curve, so
    # maximising the numerator is enough.
    return int(np.argmax(np.abs((y[-1] - y[0]) * x - y + y[0])))


#: name -> (function, the phi values to fit over)
RULES = {
    "R_max": (r_max, (None,)),
    "R_parsimony": (r_parsimony, PHIS),
    "R_knee": (r_knee, (None,)),
}
