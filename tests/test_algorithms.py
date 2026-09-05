"""Every clusterer in river.cluster must be benchmarkable, not just the distance-based ones."""

from __future__ import annotations

import pytest

from benchmark.algorithms.registry import (
    ALGORITHMS,
    NUMERIC,
    TOKENS,
    algorithms_for,
    discover_clusterers,
    get_algorithm,
    get_spec,
    make_preprocessor,
)
from benchmark.datasets.synthetic import StreamConfig, SyntheticStream
from benchmark.evaluation.runner import run_once


def _stream(n=3000, k=4, d=6, seed=1):
    return SyntheticStream(
        StreamConfig(n_samples=n, n_clusters=k, n_features=d, seed=seed)
    )


def test_registry_covers_every_river_clusterer():
    """A clusterer River adds and the registry misses is silently un-benchmarkable."""
    assert set(discover_clusterers()) == set(ALGORITHMS)


def test_capability_filters_split_the_registry_correctly():
    assert set(algorithms_for()) == {
        "CluStream", "DenStream", "DBSTREAM", "SOStream", "STREAMKMeans", "KMeans",
    }
    assert algorithms_for(partitions_samples=False) == ["ODAC"]
    assert algorithms_for(input_kind=TOKENS, partitions_samples=None) == ["TextClust"]
    assert get_spec("ODAC").partitions_samples is False
    assert get_spec("TextClust").input_kind == TOKENS
    assert get_spec("DBSTREAM").input_kind == NUMERIC


@pytest.mark.parametrize(
    "name", ["CluStream", "DenStream", "DBSTREAM", "SOStream", "STREAMKMeans", "KMeans"]
)
def test_sample_partitioning_algorithms_produce_scored_runs(name):
    res = run_once(get_algorithm(name, scaler="standard"), _stream(),
                   n_checkpoints=3, window_size=500)
    assert res.partitions_samples is True
    assert res.quality["ari"] is not None
    assert res.efficiency.throughput_sps > 0


def test_odac_clusters_variables_so_it_is_scored_on_structure_not_labels():
    """ODAC's predict_one raises NotImplementedError by design.

    Treating it like the others would either crash the run or, worse, record a
    meaningless quality score. It must come back with no external metrics and a
    structure summary instead.
    """
    res = run_once(get_algorithm("ODAC"), _stream(), n_checkpoints=3, window_size=500)

    assert res.partitions_samples is False
    assert res.quality == {}                 # empty by design, not by failure
    assert res.quality_stream == {}
    assert res.efficiency.throughput_sps > 0  # still measured on resources
    assert set(res.structure) == {"n_clusters", "n_active_clusters", "height"}
    assert res.structure["n_clusters"] >= 1
    # and the structure is tracked along the stream, not only at the end
    assert any(c.structure for c in res.checkpoints)
    assert res.to_dict()["partitions_samples"] is False


def test_odac_survives_a_registry_that_wrongly_claims_it_partitions():
    """River is the authority on the protocol, not our metadata."""
    adapter = get_algorithm("ODAC")
    adapter.partitions_samples = True  # pretend the registry got it wrong
    res = run_once(adapter, _stream(n=1000), n_checkpoints=2, window_size=500)
    assert res.partitions_samples is False  # corrected from the NotImplementedError
    assert res.quality == {}


def test_textclust_runs_on_token_counts_with_a_bag_of_words_preprocessor():
    """TextClust takes term frequencies, not coordinates — a different input kind."""
    from benchmark.datasets.base import IterableDataset

    docs = [
        ("machine learning is fun", "ml"),
        ("deep learning neural network", "ml"),
        ("cats and dogs are pets", "pets"),
        ("my dog loves cats", "pets"),
        ("neural network training", "ml"),
        ("pets need food and love", "pets"),
    ] * 60
    stream = IterableDataset(
        lambda: (({"body": text}, label) for text, label in docs),
        name="tiny_text", n_features=1, has_labels=True,
    )

    adapter = get_algorithm(
        "TextClust", params_unused=None,
    ) if False else get_algorithm(
        "TextClust", scaler="bag_of_words", scaler_params={"on": "body"},
        radius=0.4, tgap=50,
    )
    res = run_once(adapter, stream, n_checkpoints=3, window_size=120)

    assert res.partitions_samples is True
    assert res.scaler == "bag_of_words"
    assert res.scaler_params == {"on": "body"}
    assert res.quality["ari"] is not None
    assert res.efficiency.mean_micro_clusters and res.efficiency.mean_micro_clusters > 0


def test_textclust_gets_the_stream_position_it_needs_for_real_time_fading():
    """`learn_one(x, t=...)` is required when real-time fading is on.

    Without it River raises `TypeError: unsupported operand type(s) for -`, far
    from the cause. The adapter decides by signature and supplies the position.
    """
    adapter = get_algorithm(
        "TextClust", scaler="bag_of_words", scaler_params={"on": "body"},
        real_time_fading=True, tgap=5,
    )
    assert adapter._learn_takes_t is True
    for i in range(30):
        adapter.process_one({"body": f"token{i % 4} shared word"})
    assert adapter._n_seen == 30


def test_bag_of_words_is_available_as_a_preprocessor():
    """The preprocessing slot holds more than scalers."""
    bow = make_preprocessor("bag_of_words", on="body")
    bow.learn_one({"body": "hello hello world"})
    assert bow.transform_one({"body": "hello hello world"}) == {"hello": 2, "world": 1}


def test_unknown_preprocessor_and_algorithm_names_fail_loudly():
    with pytest.raises(KeyError, match="unknown preprocessor"):
        make_preprocessor("standrad")
    with pytest.raises(KeyError, match="unknown algorithm"):
        get_spec("DBStream")  # wrong casing


def test_variable_clusters_are_not_reported_as_sample_clusters():
    """ODAC's n_clusters counts variable groups, a different unit entirely.

    Printing it in the shared "Macro K" column next to everyone else's
    sample-cluster count invites a comparison that is not valid.
    """
    from benchmark.reports import efficiency_table, structure_table

    odac = run_once(get_algorithm("ODAC"), _stream(), n_checkpoints=3, window_size=500)
    dbstream = run_once(get_algorithm("DBSTREAM", scaler="standard"), _stream(),
                        n_checkpoints=3, window_size=500)

    assert odac.efficiency.final_macro_clusters is None
    assert dbstream.efficiency.final_macro_clusters is not None
    # the real counts are still reported, under a heading that names the unit
    assert "Variable clusters" in structure_table([odac])
    assert str(odac.structure["n_clusters"]) in structure_table([odac])
    # ODAC stays in the efficiency table: runtime and memory are comparable
    assert "ODAC" in efficiency_table([odac, dbstream])


def _ari_in_subprocess(hashseed: str) -> str:
    import os
    import subprocess
    import sys

    script = (
        "from benchmark.datasets.synthetic import StreamConfig, SyntheticStream;"
        "from benchmark.experiments import BenchmarkConfig, Setting, run_setting;"
        "ds = SyntheticStream(StreamConfig("
        "n_samples=3000, n_features=10, n_clusters=5, separation=1.0, seed=1));"
        "r = run_setting(Setting('STREAMKMeans', params={'n_clusters': 5, 'chunk_size': 500},"
        " scaler='standard'), ds, BenchmarkConfig(n_checkpoints=3, window_size=1000), seed=1);"
        "print(round(r.quality['ari'], 10))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env={**os.environ, "PYTHONHASHSEED": hashseed},
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_a_pinned_hash_seed_makes_a_run_reproducible():
    """The reproducibility guarantee that actually holds — pin PYTHONHASHSEED.

    River's KMeans is not reproducible from `seed` alone (RIVER_BUGS.md BUG-2),
    so this asserts the weaker property the framework can stand behind, and the
    environment record carries the hash seed a result was produced under.
    """
    assert len({_ari_in_subprocess("0") for _ in range(3)}) == 1


def test_hash_seed_sensitivity_is_recorded_rather_than_hidden():
    """BUG-2 is a river bug we do not patch, so the framework must surface it.

    If this test ever starts failing because the scores agree, river has fixed
    the underlying issue and the RIVER_BUGS.md entry should be closed.
    """
    from benchmark.reports import capture_environment

    scores = {_ari_in_subprocess(h) for h in ("0", "1", "7")}
    assert len(scores) > 1, (
        "river KMeans now looks hash-order independent; re-check RIVER_BUGS.md BUG-2"
    )
    # the environment record is what makes such a result interpretable
    assert "python_hash_seed" in capture_environment()
