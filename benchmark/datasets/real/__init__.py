"""Real-world dataset loaders.

Which streams are here, and why each one earns its place:

* ``kddcup99`` — the field's default benchmark; kept for comparability.
* ``covtype`` — the second most-used stream-clustering dataset; large and
  genuinely high-dimensional.
* ``insects`` — the only one whose drift type and timing are documented rather
  than assumed (Souza et al. 2020), which makes it the honest choice for a
  drift scenario.
* ``shuttle``, ``keystroke`` — small, fast, and awkward in useful ways: severe
  class imbalance and a high class count respectively.
* ``electricity`` — included for continuity with the literature only; see the
  warning in its docstring.
* ``sms_spam`` — **text**, for the clusterers that consume token counts rather
  than coordinate vectors (TextClust).
* ``kddcup98`` — regression-flavoured RFM data, kept from the original repo.
* ``rialto``, ``sensor`` and the imbalanced incremental INSECTS variants
  (``drift_streams``) — real streams with natural drift that River does not
  ship, for checking that findings hold on drift nobody designed.

Note that KDDCup99 and CoverType both arrive in long same-class blocks. That is
not a reason to drop them, but it is a reason to read
``RunResult.window_label_diagnostics`` before reading any per-window score.
"""

from __future__ import annotations

from .covtype import COVTYPE_QUANTITATIVE, load_covtype
from .drift_streams import INSECTS_EXTRA_VARIANTS, load_insects_extra, load_rialto, load_sensor
from .kdd import load_kddcup98, load_kddcup99
from .river_streams import (
    INSECTS_VARIANTS,
    from_river_dataset,
    load_electricity,
    load_insects,
    load_keystroke,
    load_shuttle,
    load_sms_spam,
)

__all__ = [
    "load_kddcup99",
    "load_kddcup98",
    "load_covtype",
    "COVTYPE_QUANTITATIVE",
    "from_river_dataset",
    "load_insects",
    "INSECTS_VARIANTS",
    "load_shuttle",
    "load_keystroke",
    "load_sms_spam",
    "load_electricity",
    "load_rialto",
    "load_sensor",
    "load_insects_extra",
    "INSECTS_EXTRA_VARIANTS",
]
