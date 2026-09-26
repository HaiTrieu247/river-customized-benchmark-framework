"""Online hyperparameter adaptation for stream clusterers.

The machinery behind a confStream-style loop (Carnein et al., 2020): a window
buffer that can be replayed into fresh models, a batch criterion scored on that
buffer, a search space that samples offspring around a parent, and the ensemble
that ties them together. Nothing here knows which datasets or algorithms a
protocol runs; that is the caller's business.
"""
