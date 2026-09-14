"""HPO pipeline for streaming clustering.

Self-contained: the stages, the frozen configs, the cached permutations and
every output live under this one directory. Run the stages in order, from the
repo root::

    python -m hpo_pipeline.pool          # Stage 0  configs/pool.json
    python -m hpo_pipeline.stream        # Stage 1  cache/*.npy
    python -m hpo_pipeline.probe         # Stage 2  configs/probe.json
    python -m hpo_pipeline.space         # Stage 3  configs/search_space.json
    python -m hpo_pipeline.manifest      #          out/manifest.json
    python -m hpo_pipeline.build_matrix  # Stage 4  out/matrix.jsonl
    python -m hpo_pipeline.health        # Stage 5  out/health.md        <- GATE
    python -m hpo_pipeline.views         # Stage 6  out/views/*.csv
    python -m hpo_pipeline.prefix        # Stage 7a out/prefix.jsonl
    python -m hpo_pipeline.matr_cv       # Stage 7b out/matrcv.jsonl
    python -m hpo_pipeline.analyze       # Stage 8  out/report/

Layout::

    hpo_pipeline/
    |-- <the stage modules>
    |-- configs/   frozen once under a fixed seed, and versioned: the candidate
    |              set is part of the protocol, not a by-product of it
    |-- cache/     materialised permutations, large and regenerable
    `-- out/       matrix, health report, views, analysis

Only Stages 4 and 7 cost compute. Stage 6 is free: it is a set of readings over
one matrix, which is the whole reason the matrix exists.
"""
