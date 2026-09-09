"""Synthetic datasets DS1 and DS2 of RepStream (Lühr & Lazarescu, DKE 68, 2009).

Section 5.1 uses two hand-crafted 2-d sets to show that RepStream handles clusters of
arbitrary shape, size and density while DBSCAN, tied to one global density, cannot.

What the paper actually specifies
---------------------------------
Very little. Sec. 5.1 gives only:

* ``|DS1| = 9153`` and ``|DS2| = 5458`` points;
* a randomised point ordering, fixed once and reused for every run;
* Manhattan distance;
* the pictures in Fig. 13.

There are no coordinates, no per-shape point counts and no density values. Everything
below is therefore **read off Fig. 13 and calibrated**, not quoted. The shape inventory
and layout come from Fig. 13; the ground-truth colouring of Fig. 14a and Fig. 17a was
used to confirm how many clusters each panel is meant to contain (14 for DS1, 13 for
DS2) and which blobs are one cluster rather than two.

Calibrating the scale
---------------------
The one quantitative handle the paper leaves is the DBSCAN sweep: with ``minPts = 5``,
DS1 is said to cluster its dense shapes at ``eps = 15`` while its sparse shapes fragment,
and to merge the two top-left triangles at ``eps = 16``; DS2 leaves points unclustered at
``eps = 6``, is optimal at ``eps = 6.5`` and merges two nearby clusters at ``eps = 7``.
Those numbers pin the coordinate scale of the two sets, because they are absolute radii.

``SCALE_DS1`` and ``SCALE_DS2`` were chosen by sweeping the scale factor and keeping the
value whose DBSCAN behaviour matches those sentences; ``benchmark/experiments/papers/
repstream.py`` re-runs that check as Fig. 16 and Fig. 18 and prints what it found, so the
calibration is visible in the output rather than asserted here.

What reproduces and what does not
---------------------------------
The shape inventory, the relative densities and the qualitative DBSCAN behaviour
reproduce. The exact point coordinates cannot: they were never published. Any statement
of the form "RepStream finds N clusters on DS1" is therefore a statement about *this*
reconstruction, and Fig. 14/15/17 should be read as shape comparisons against the
paper's pictures, not as numeric agreement.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from ..base import IterableDataset

#: Frame both sets are laid out in before scaling: x in [0, 1000], y in [0, 680].
FRAME_WIDTH = 1000.0
FRAME_HEIGHT = 680.0

#: Calibrated so DBSCAN(minPts=5) behaves as Sec. 5.1 describes at eps = 15 / 16.
SCALE_DS1 = 1.95
#: Frame-unit shift that brings the two top-left triangles of Fig. 13a close enough for
#: DBSCAN to merge them at eps = 16 but not at eps = 15, which is what Sec. 5.1 reports.
DS1_TRIANGLE_GAP = 32.5
#: Calibrated so DBSCAN(minPts=5) behaves as Sec. 5.1 describes at eps = 6 / 6.5 / 7.
SCALE_DS2 = 0.55
#: Frame-unit drop that pulls the thin line of Fig. 13b clear of the top bar, so the two
#: stay separate across the whole eps sweep as Sec. 5.1 implies.
DS2_LINE_DROP = 10.0
#: Frame-unit drop that brings the upper donut close enough to the triangle apex for
#: DBSCAN to merge the pair at eps = 7 but not at eps = 6.5.
DS2_DONUT_DROP = 10.0


# --- shape samplers --------------------------------------------------------
def _triangle(rng, n, a, b, c):
    """Uniform over the filled triangle ``a b c``."""
    u = rng.random(n)
    v = rng.random(n)
    root = np.sqrt(u)
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    return a + root[:, None] * (1 - v)[:, None] * (b - a) + root[:, None] * v[:, None] * (c - a)


def _disc(rng, n, center, rx, ry):
    """Uniform over a filled ellipse."""
    theta = rng.random(n) * 2 * math.pi
    radius = np.sqrt(rng.random(n))
    return np.column_stack(
        [center[0] + radius * rx * np.cos(theta), center[1] + radius * ry * np.sin(theta)]
    )


def _annulus(rng, n, center, r_inner, r_outer, ry_scale=1.0):
    """Uniform over a ring; ``ry_scale`` squashes it into an ellipse."""
    theta = rng.random(n) * 2 * math.pi
    radius = np.sqrt(rng.random(n) * (r_outer**2 - r_inner**2) + r_inner**2)
    return np.column_stack(
        [
            center[0] + radius * np.cos(theta),
            center[1] + radius * ry_scale * np.sin(theta),
        ]
    )


def _box(rng, n, x0, y0, x1, y1):
    """Uniform over an axis-aligned rectangle."""
    return np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])


def _rotated_box(rng, n, center, width, height, angle_deg):
    local = np.column_stack(
        [rng.uniform(-width / 2, width / 2, n), rng.uniform(-height / 2, height / 2, n)]
    )
    angle = math.radians(angle_deg)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return local @ rot.T + np.asarray(center, float)


def _polyline(rng, n, vertices, thickness, closed=False):
    """Uniform along a polyline, jittered normal to it by ``thickness``."""
    pts = np.asarray(vertices, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = pts[1:] - pts[:-1]
    lengths = np.hypot(seg[:, 0], seg[:, 1])
    weights = lengths / lengths.sum()
    which = rng.choice(len(seg), size=n, p=weights)
    t = rng.random(n)
    base = pts[:-1][which] + seg[which] * t[:, None]
    normal = np.column_stack([-seg[which][:, 1], seg[which][:, 0]])
    normal /= np.hypot(normal[:, 0], normal[:, 1])[:, None]
    offset = rng.uniform(-thickness / 2, thickness / 2, n)
    return base + normal * offset[:, None]


def _sine(rng, n, x0, x1, y_center, amplitude, cycles, thickness):
    x = rng.uniform(x0, x1, n)
    phase = 2 * math.pi * cycles * (x - x0) / (x1 - x0)
    y = y_center + amplitude * np.sin(phase)
    return np.column_stack([x, y + rng.uniform(-thickness / 2, thickness / 2, n)])


# --- DS1 -------------------------------------------------------------------
def _ds1_shapes(rng, counts):
    """The 14 clusters of Fig. 13a, in the order they are listed in ``DS1_WEIGHTS``."""
    n = counts
    return [
        _triangle(rng, n[0], (163, 629), (93, 490), (211, 490)),
        _triangle(rng, n[1], (245 - DS1_TRIANGLE_GAP, 493), (219 - DS1_TRIANGLE_GAP, 621), (336 - DS1_TRIANGLE_GAP, 610)),
        _triangle(rng, n[2], (518, 632), (457, 486), (598, 486)),
        _triangle(rng, n[3], (679, 494), (614, 632), (763, 632)),
        _triangle(rng, n[4], (836, 629), (747, 486), (920, 486)),
        _annulus(rng, n[5], (285, 261), 175, 205),
        _triangle(rng, n[6], (242, 281), (187, 347), (296, 347)),
        _disc(rng, n[7], (360, 316), 44, 40),
        _disc(rng, n[8], (235, 210), 46, 40),
        _triangle(rng, n[9], (358, 258), (310, 166), (417, 166)),
        _disc(rng, n[10], (619, 346), 68, 62),
        _disc(rng, n[11], (826, 330), 85, 80),
        _rotated_box(rng, n[12], (719, 258), 89, 44, -20.0),
        _sine(rng, n[13], 590, 885, 138, 52, 1.5, 26),
    ]


#: Relative point mass of each DS1 cluster; the ring takes the rounding remainder.
DS1_WEIGHTS = [700, 700, 190, 240, 800, 3000, 480, 120, 130, 520, 340, 900, 330, 380]
DS1_SIZE = 9153

#: Human-readable names, matching the reading order of Fig. 13a.
DS1_NAMES = [
    "triangle_up_dense_left",
    "triangle_down_dense_left",
    "triangle_up_sparse",
    "triangle_down_sparse",
    "triangle_up_dense_right",
    "ring",
    "triangle_down_in_ring",
    "blob_sparse_in_ring_upper",
    "blob_sparse_in_ring_lower",
    "triangle_up_in_ring",
    "blob_sparse_right",
    "blob_dense_right",
    "parallelogram",
    "wave",
]


# --- DS2 -------------------------------------------------------------------
def _ds2_shapes(rng, counts):
    """The 13 clusters of Fig. 13b, in the order they are listed in ``DS2_WEIGHTS``."""
    n = counts
    return [
        _box(rng, n[0], 117, 590, 858, 630),
        _box(rng, n[1], 75, 574 - DS2_LINE_DROP, 229, 584 - DS2_LINE_DROP),
        _box(rng, n[2], 121, 79, 173, 542),
        _box(rng, n[3], 871, 139, 931, 640),
        _box(rng, n[4], 196, 79, 806, 120),
        _disc(rng, n[5], (386, 498), 88, 56),
        _disc(rng, n[6], (703, 502), 88, 52),
        _annulus(rng, n[7], (546, 428 - DS2_DONUT_DROP), 23, 54, ry_scale=0.70),
        _annulus(rng, n[8], (383, 365), 23, 54, ry_scale=0.70),
        _annulus(rng, n[9], (703, 365), 23, 54, ry_scale=0.70),
        _polyline(rng, n[10], [(546, 365), (321, 157), (796, 157)], 26, closed=True),
        _disc(rng, n[11], (838, 80), 20, 22),
        _disc(rng, n[12], (892, 100), 18, 14),
    ]


#: Relative point mass of each DS2 cluster; the triangle takes the rounding remainder.
DS2_WEIGHTS = [700, 150, 600, 620, 600, 320, 300, 190, 190, 180, 1450, 30, 60]
DS2_SIZE = 5458

DS2_NAMES = [
    "bar_top",
    "line_thin",
    "bar_left",
    "bar_right",
    "bar_bottom",
    "blob_sparse_left",
    "blob_sparse_right",
    "donut_top",
    "donut_left",
    "donut_right",
    "triangle_outline",
    "blob_tiny_sparse",
    "blob_tiny_dense",
]


# --- assembly --------------------------------------------------------------
def _allocate(weights, total, remainder_index):
    """Split ``total`` points across ``weights``, giving the remainder to one cluster."""
    scaled = [max(1, round(total * w / sum(weights))) for w in weights]
    scaled[remainder_index] += total - sum(scaled)
    return scaled


def _build(shapes_fn, weights, total, remainder_index, scale, seed):
    rng = np.random.default_rng(seed)
    counts = _allocate(weights, total, remainder_index)
    blocks = shapes_fn(rng, counts)
    points = np.vstack(blocks) * scale
    labels = np.concatenate([np.full(len(b), i) for i, b in enumerate(blocks)])
    order = rng.permutation(len(points))
    return points[order], labels[order]


def _dataset(points, labels, name, names):
    rows = [
        ({"x0": float(p[0]), "x1": float(p[1])}, int(c)) for p, c in zip(points, labels)
    ]

    def factory() -> Iterator[tuple[dict[str, float], int]]:
        return iter(rows)

    dataset = IterableDataset(
        factory, name=name, n_features=2, has_labels=True, length=len(rows)
    )
    dataset.cluster_names = names
    dataset.points = points
    dataset.labels = labels
    return dataset


def make_repstream_ds1(seed: int = 42, scale: float = SCALE_DS1) -> IterableDataset:
    """DS1 of Fig. 13a: 9153 points in 14 clusters of mixed shape and density."""
    points, labels = _build(_ds1_shapes, DS1_WEIGHTS, DS1_SIZE, 5, scale, seed)
    return _dataset(points, labels, "repstream_ds1", DS1_NAMES)


def make_repstream_ds2(seed: int = 42, scale: float = SCALE_DS2) -> IterableDataset:
    """DS2 of Fig. 13b: 5458 points in 13 clusters of mixed shape and density."""
    points, labels = _build(_ds2_shapes, DS2_WEIGHTS, DS2_SIZE, 10, scale, seed)
    return _dataset(points, labels, "repstream_ds2", DS2_NAMES)
