# Run tracker

Tiny YAML-based experiment tracker. Every train-script invocation writes
one manifest with hparams + per-serial metrics + paths to weights/TB dirs.
Optional Aim UI on top if you want it; works fine without.

## Where the code lives

| File | What it does |
|---|---|
| `src/bst_refactor/run_tracker.py` | `track_run(config, run_id, log_path=...)` and `track_serial(run_dir, serial_no, weights_path, tb_dir, metrics)`. Writes `manifest.yaml` and (optionally) mirrors into `.aim/`. |
| `src/bst_refactor/run_overview.py` | Aggregator. `python run_overview.py` prints a table across all runs under `experiments/` (mean/stdev/max per metric). |
| `src/bst_refactor/aim_backfill.py` | One-shot script that walks every manifest + its test log, and mirrors each serial into Aim with description/tags/name. Idempotent, so re-run it any time Aim was missing when training finished, or after editing manifests. |
| `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py` | Integrated: two calls to the tracker, test methods now return metric dicts, TB directory is threaded through to `train_network`. |

## How it's wired into bst_train.py

```python
from run_tracker import track_run, track_serial

run_dir, run_id = track_run(config=hyp, run_id=f'run_{timestamp}')
weight_dir = run_dir / 'weights'

for serial_no in range(1, 6):
    tb_dir = run_dir / 'tb' / f'serial_{serial_no}'
    task.seek_network_weights(model_info=..., serial_no=serial_no, tb_dir=tb_dir)
    test_metrics = task.test(...)
    topk_metrics = task.test_topk_acc(k=2)
    track_serial(run_dir, serial_no,
                 weights_path=task.weight_path,
                 tb_dir=tb_dir,
                 metrics={**test_metrics, **topk_metrics})
```

What each call configures:

- **`config=hyp`**: the hparam payload on `track_run`. Accepts any dataclass, namedtuple, dict, or object with `vars()`; lands verbatim under `config:` in `manifest.yaml`.
- **`run_id`**: names the `experiments/<run_id>/` subfolder. `run_{timestamp}` is the convention for regular runs; pass any string for a named/legacy run (e.g. `bst_cg_ap_base_17_04_2026`), or pass `None` to let `track_run` auto-generate a `run_YYYYMMDD_HHMMSS` id.
- **`weights_path` / `tb_dir` / `metrics`**: the per-serial payload on `track_serial`; lands in the manifest's `serials:` list. No layout is enforced, but by convention weights live at `run_dir/weights/` and TB events at `run_dir/tb/serial_N/`. `track_serial` is keyed by `serial_no`, so re-running a test updates the entry in place.
- **`log_path=<path>`** (optional, on `track_run`): stored on the manifest so `aim_backfill.py` can slice per-serial blocks out of the test log later. Not needed during the live run; only matters if you want the backfill to enrich Aim descriptions.
- **Aim mirror**: auto-activates if `aim` is pip-installed and `track_serial` has metrics. Silently skips otherwise; nothing in the training loop breaks either way.

That's the whole integration. Any other train script (Scott's, a future
3D CNN extension) can do the same two calls.

## Directory layout

```
src/bst_refactor/stroke_classification/main_on_shuttleset/
  experiments/
    run_20260418_174244/
      manifest.yaml                             (tracked in git)
      weights/bst_CG_AP_..._merged_25.pt        (gitignored)
      tb/serial_1/, serial_2/, ...              (gitignored)
  test_logs/
    test_20260418_174244.log                    (unchanged, pairs with run_id)
```

Launch TensorBoard with `tensorboard --logdir experiments/<run_id>/tb` to
see all serials of a run grouped together.

## Manifest format

```yaml
run_id: run_20260418_174244
started_at: 2026-04-18T17:42:44
git_sha: e2c2b74...
git_dirty: true
host: engelbart.une.edu.au
log_path: test_logs/test_20260418_174244.log   # optional; enables aim_backfill
config:
  n_epochs: 80
  lr: 0.0005
  use_aux_schedule: false
  ...
serials:
  - serial_no: 1
    weights_path: experiments/run_.../weights/bst_CG_AP_..._1.pt
    tb_dir: experiments/run_.../tb/serial_1
    metrics:
      macro_f1: 0.834
      min_f1: 0.591
      accuracy: 0.846
      top2_accuracy: 0.963
      num_strokes: 3486
    recorded_at: 2026-04-18T17:48:12
best_serials: [1, 4]                           # optional; serial_nos tagged 'best' in Aim
notes: ...                                     # optional; shown as Aim 'run_notes' param
tags: [arch1_baseline]                         # optional; extra Aim tags
```

`track_serial` is idempotent by `serial_no` so re-running a test updates
the entry in place rather than appending a duplicate.

## Aggregator usage

```bash
cd src/bst_refactor/stroke_classification/main_on_shuttleset
python ../../run_overview.py                              # default experiments/
python ../../run_overview.py -c n_epochs,use_aux_schedule -m macro_f1,min_f1
```

Prints one row per run with mean/stdev/max across serials.

## Aim UI (optional)

If `aim` is pip-installed in the venv, each call to `track_serial` also
opens a lightweight `aim.Run(run_hash='<run_id>_s<N>')` and logs hparams
+ metrics. Browse with:

```bash
pip install aim
aim up                                      # local UI at http://localhost:43800
```

If aim is not installed, the tracker silently skips the mirror step.
Nothing in the training loop breaks either way.

### Backfill (also: recover from "forgot to install aim")

`aim_backfill.py` regenerates every Aim run from scratch by reading
`experiments/*/manifest.yaml` + the test log each manifest points to.
It enriches each serial's Aim run with:

- the serial's full test-log block as the run description,
- auto-derived tags (`legacy`, `no_aux_anneal` / `anneal_gentle` /
  `anneal_aggressive` / `cg_ap_off_from_start`, and `best` when the
  serial appears in `best_serials`),
- a human-readable name `<run_id>_s<N>`.

```bash
pip install aim
cd src/bst_refactor/stroke_classification/main_on_shuttleset
python ../../aim_backfill.py
aim up
```

Idempotent. Aim keys each run by the hash `<run_id>_s<N>`, so re-running
the backfill overwrites the existing entries rather than duplicating.
Run it after any training batch where aim wasn't installed, or after
editing tags / notes in a manifest.

## Other loggers

The tracker records *paths* for any logger (TB, W&B offline, CSV, plain
text). It does not try to parse arbitrary event formats, so the
cross-run aggregator (`run_overview.py`) only reads metrics from
`manifest.yaml` (which the train script populates from whatever source
it wants). If the team wants cross-run metric scraping to just work,
standardize on passing final metrics into `track_serial(metrics=...)`
regardless of which logger produced them.

## Dependencies

- `pyyaml>=6.0,<7` (required) — add to
  `src/bst_refactor/stroke_classification/requirements.txt` if not
  already there.
- `aim` (optional) — only needed if you want the UI.
