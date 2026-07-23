# Ablation Experiments

Every ablation family owns its configuration and output tree. Formal artifacts
are written only to:

```text
<family>/runs/<variant>/<run-id>/
  run_manifest.json
  logs/train.log
  checkpoint-*/
  wandb/
```

`P0` contains experiments required to support the paper's central claims.
`P1` contains optional analyses that may be omitted when compute or time is
limited. A variant has one of three states:

- `ready`: implemented and admitted by the launcher;
- `eval_only`: consumes existing predictions or checkpoints and cannot train;
- `pending_*`: preserved in the plan but blocked until its protocol or code is
  independently verified.

The launcher refuses to train non-ready variants and refuses to reuse an
existing run directory. It records the resolved environment plus config and
backend hashes in `run_manifest.json` before starting training.

Validate the full catalog:

```bash
./validate_all.sh
```

Inspect a variant without launching:

```bash
python3 suite.py show \
  --config P0/01_rhc_components/experiment.json \
  --variant no_cross_granularity_gain
```

Dry-run or launch one isolated variant:

```bash
./run_train.sh P0/01_rhc_components/experiment.json \
  no_cross_granularity_gain --dry-run --run-id preflight

./run_train.sh P0/01_rhc_components/experiment.json \
  no_cross_granularity_gain --run-id <unique-run-id>
```

Each family also exposes the same operation through its local `run.sh`, e.g.
`P0/05_token_budget/run.sh budget_64 --dry-run --run-id preflight`.

## Result Records

Each family has a `RESULTS.md` comparison table, and each concrete variant has
its own `variants/<variant>/RESULTS.md`. The variant record contains the full
resolved configuration, interaction rule, training and evaluation protocols,
artifact paths, paper-level summary metrics, all 36 MMEB task results, all 10
ViDoRe V1 subset results, and all 7 ViDoRe V2 subset results.

Refresh generated configuration blocks after editing an experiment config:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B result_records.py
```

Only text inside the generated markers is refreshed. Result tables,
observations, and manually entered artifact paths remain unchanged. Running
`materialize.py` also refreshes the applicable family and variant records.

The shared defaults in `defaults/rhc_formal.json` mirror the final main-model
training recipe. A family file should override only the variable under study.
`LEGACY_SOURCE_MAP.md` records where each implementation currently lives and
which adapters remain intentionally blocked.
