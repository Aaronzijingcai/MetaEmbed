exp_oracle runs three independently trained g1/g2/g3 models and aggregates an
oracle-style report.

Usage:
  cd /path/to/MetaEmbed
  bash colqwen_multigranularity/experiments/exp_oracle/eval_oracle_3sets.sh \
    /path/to/g1_model \
    /path/to/g2_model \
    /path/to/g3_model

Default model paths:
  output/colqwen2.5-g1-full
  output/colqwen2.5-g2-full
  output/colqwen2.5-g3-full

Output:
  colqwen_multigranularity/runs/exp_oracle/raw/g1/*.json
  colqwen_multigranularity/runs/exp_oracle/raw/g2/*.json
  colqwen_multigranularity/runs/exp_oracle/raw/g3/*.json
  colqwen_multigranularity/runs/exp_oracle/per_query/g1/*/*.per_query.json
  colqwen_multigranularity/runs/exp_oracle/per_query/g2/*/*.per_query.json
  colqwen_multigranularity/runs/exp_oracle/per_query/g3/*/*.per_query.json
  colqwen_multigranularity/runs/exp_oracle/oracle_report.json

Current oracle definition:
  Strict per-query oracle. For each query, choose the best of g1/g2/g3 by the
  benchmark main metric: ndcg_at_5 for ViDoRe v1/v2 and recall_at_1 for MMEB.

Eval settings:
  The script loads the processor from each g1/g2/g3 model directory, uses 1024
  visual tokens, and includes multilingual ViDoRe v2 subsets. These settings
  match the historical single-granularity eval outputs under output/colqwen2.5-g*-full.
