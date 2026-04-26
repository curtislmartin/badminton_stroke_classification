# Pre-Phase-2 Tidy Execution Plan

**Date:** 2026-04-26
**Source:** `pre_phase_2_review_2026-04-26.md` plus user feedback in this session.

This is the operational plan for executing the agreed tidy actions before the X3D-S layer lands. Commit-by-commit, each step gated by automated local safety checks. The big remote-enabled batch (engelbart V100, real data) is run by you at the end; nothing in this plan needs HPC access.

---

## Branch destination

**Decision:** sit on the `pre-phase-2-tidy` branch after local tests pass. After remote tests pass, the user will decide. No push, no PR, no merge to main without explicit go-ahead.

## X3D-S wire-in invariant

The X3D-S wrist-crop layer fuses into the **BST model internals**. The following points are read-only across this entire branch. They are where the new module wires in, not external orchestration surface.

- `bst.py:106-110` `CrossTransformerLayer` signature and attention plumbing.
- `bst.py` `BST_CG_AP` forward graph: token sequencing, positional embeddings, `d_model=100`/`d_head=128`/`n_head=6` defaults, per-stream embedding heads (pose, shuttle, position).
- `bst.py:28` building-block imports (`TCN`, `MLP`, `MLP_Head`, `FeedForward`, `TransformerEncoder`). Step 5 deletes the four standalone `TemPose_*` classes; the building blocks themselves stay verbatim in `tempose.py`.
- `bst.py:433-437` the five `BST_*` partials.
- No new abstraction layer between `bst.py` and the train loop. `bst_common.py` lifts `MODELS` / `Task` / dataloader helpers only; direct `MODELS[...](**kw)` instantiation stays.

Internal refactor of `bst_train.py`, `bst_infer.py`, `prepare_train_on_shuttleset.py`, `shuttleset_dataset.py`, etc. is fine. The BST model graph itself is not refactored on this branch; this branch only addresses sprawl and dead code in the surrounding scaffolding.

---

## Branch strategy

- New branch `pre-phase-2-tidy` cut from `main` at the current HEAD (`d4fd644`).
- One commit per logical step, smallest blast radius first.
- Branch is **local-only until you say otherwise**. No `git push`. No PR.
- If any step fails its safety check, stop and ping. We do not "fix forward" within auto-mode; we revert the offending commit and reassess together.

---

## Per-step protocol

```
1. Confirm pytest is green on the current commit.
2. Apply the step (file edits, deletes, moves).
3. Run the step's safety checks (pytest + targeted greps + import smokes).
4. If green, commit with a single-sentence message in repo style.
5. If red, do not commit; report the failure and stop.
```

---

## Local safety checks (what I run at each step)

These all run without HPC, without `BST_DATA_DIR` set, and without GPU.

1. **`pytest`** — runs `tests/test_environment.py`, `tests/test_dataset.py`, `tests/test_api.py`; `tests/test_integration.py` auto-skips when `BST_DATA_DIR` is unset. Baseline must stay green throughout.
2. **Targeted import smoke** — for steps that touch importable modules, a one-liner Python invocation that imports the affected entry points and instantiates the high-level objects (`Task`, `MODELS`, dataset classes). Catches surface-level breakage that pytest doesn't reach.
3. **Grep verification before deletes** — for every name being deleted, `grep -rn` across `src/`, `scripts/`, `tests/`, `notebooks/`. Zero hits outside the file itself before deletion proceeds.
4. **`ruff check`** on touched files only. Scope is set in `pyproject.toml`; existing exclusions for `notebooks/`, `**/deprecated/`, `TrackNetV3/` honoured.

If a step adds new tests (sticky_anchor), those tests join the local checks from that commit onward.

---

## Remote-enabled checks (you run at the end, on engelbart)

These belong on the V100 with the npy data accessible. I will leave a one-page handoff appended to this doc when the local branch is ready.

1. **Byte-identity gate.** Run `failsafe_bst_mmpose_zeroing_check_equivalence` on the 50-clip hit-zone sample. Confirms the BST/MMPose path is bit-identical to the committed extract. Required for any commit that touches `apply_heuristic.py`, `heuristics/`, `prepare_train_on_shuttleset.py` MMPose code, or imports thereof.
2. **`pytest tests/test_integration.py`** with `BST_DATA_DIR` pointed at the active ablation-tagged collated dir. Confirms the dataset->loader->BST_0 forward path still works end-to-end.
3. **2-epoch smoke train.** `python bst_train.py` with `n_epochs=2`, `early_stop_n_epochs=999`, single seed. Compare loss/accuracy curves against a fresh pre-tidy baseline run done on `main` before branching. Curves should match within run-to-run noise.

If all three pass on engelbart, you ping me and we resolve the open question above.

---

## Pre-execution baseline capture

Before cutting the branch I will:
1. Confirm `git status` clean.
2. Run `pytest` on `main` and record the full output.
3. Note the current HEAD SHA.
4. Capture a `git log --stat` snapshot of the files about to be touched (so anything unexpected on the branch is easy to spot in review).

You will additionally want to do the **2-epoch baseline run on `main` before I cut the branch** so the post-tidy smoke train has something to compare against. I'll flag this in the handoff section once I'm ready.

---

## Step-by-step plan

Order is smallest blast radius first. Steps 1-7 are bundled into one local-only branch. Steps 8+ are reserved for after-X3D-S in the review doc and are not part of this execution.

### Step 1 — Doc drift fixes

**Touches:** Markdown only.

- `.claude/project_overview.md:94-96`: rewrite the heuristics description (no `HeuristicFilter` class hierarchy exists; `apply(raw, ctx, **kw) -> HeuristicOutput` registered in `heuristics/__init__.py:REGISTRY`).
- `src/bst_refactor/data_pipeline_to_model_train.md:392`: rewrite `Hyp` defaults table to reflect live `bst_train.py:140-157` (`n_epochs=80`, `taxonomy='une_merge_v1_nosides'`, `aux_fade_end_epoch=15`, etc.).
- `src/bst_refactor/data_pipeline_to_model_train.md:121, 535` and `src/bst_refactor/pipeline/README.md:165, 247`: add `une_merge_v1_nosides` to taxonomy lists.
- `src/bst_refactor/data_pipeline_to_model_train.md:260-263`: remove `Dataset_npy_collated_one_side` and `_single_pose` from "primary classes" listing (or mark as orphaned-pending-deletion).
- `src/bst_refactor/run_tracker.md:64-90`: add `extra: data_provenance: {clips_csv_path, clips_csv_sha256, effective_ablation_id, npy_collated_dir}` to the manifest format example.
- `scratch/architecture_notes/arch_1_directions.md:101`: refresh `bst_train.py` line refs (cosine scheduler now at `:395-400`).
- `README.md:100-104`: list the actual contents of `tests/` (`test_api`, `test_data_access`, `test_dataset`, `test_environment`, `test_integration`).

**Safety checks:** `pytest`. (Doc-only, but runs as a sanity gate on the working tree.)

**Commit message draft:** "Doc drift sweep: refresh Hyp defaults, taxonomy lists, heuristics description, manifest example, line refs."

---

### Step 2 — Create historical doc skeletons + script archive directory

**Touches:** New empty/skeleton files, new directories.

- `scratch/architecture_notes/historical_bst.md` — already drafted as a skeleton in this same session (see below). Will be filled in steps 3 and 4 as content is excised.
- `scratch/architecture_notes/pipeline_context_notes.md` — light skeleton; filled if any pipeline-area context notes are excised in later steps.
- `scratch/project_history/` — new directory for the relocated `deprecated/` and `outdated_*` content. Directory created with a `README.md` explaining what lives there.
- `scripts/archive/` — new directory with a `README.md` explaining that everything inside is one-shot tooling kept for reference, not part of the active build.

**Safety checks:** `pytest`. New directories don't affect imports.

**Commit message draft:** "Add historical doc skeletons and archive directories for the tidy pass."

---

### Step 3 — Move historical/deprecated content out of `src/`

**Touches:** Moves only; no source-code edits.

- `src/bst_refactor/deprecated/` → `scratch/project_history/bst_refactor_deprecated/`
- `src/bst_refactor/ShuttleSet/deprecated/` → `scratch/project_history/shuttleset_deprecated/`
- `src/bst_refactor/stroke_classification/main_on_shuttleset/tmp/` → `scratch/project_history/main_on_shuttleset_tmp/`

Each move uses `git mv`. The five `outdated_*.md` and `historical_*.md` files inside `src/bst_refactor/deprecated/` come along for the ride.

`scratch/project_history/README.md` records the original locations and the date of the move so a future report can reconstruct the layout.

**Safety checks:**
- `pytest`.
- `grep -rn "from .*deprecated" src/ scripts/ tests/` returns zero hits in active code (the only allowed hits are inside the moved directories themselves, which are now under `scratch/`).
- `grep -rn "main_on_shuttleset/tmp\|main_on_shuttleset\.tmp" src/ scripts/ tests/` returns zero hits.

**Commit message draft:** "Relocate src/-tree historical and deprecated trees into scratch/project_history/."

---

### Step 4 — Capture excised content into `historical_bst.md` (pre-deletion fill)

**Touches:** `scratch/architecture_notes/historical_bst.md`, `scratch/architecture_notes/arch_1_directions.md`. No source-code deletes yet.

Before deleting anything from `tempose.py`, `shuttleset_dataset.py`, or `bst_train.py`, capture:

- The four `TemPose_*` standalone classes' purpose (one paragraph each, what they did, why they were preserved through phase 0, and the verbatim source preserved as a code block in case future reproduction needs them).
- The original BST `Hyp` namedtuple defaults (`n_epochs=1600`, etc.) verbatim from the commented-out block at `bst_train.py:85-101`.
- The "LR-SCHEDULE RETUNE 2026-04-17" and "AUX-SCHEDULE 2026-04-18" rationale blocks verbatim. Trim to one-line pointer in `bst_train.py`; full block lives in `historical_bst.md`.
- The `Dataset_npy`, `Dataset_npy_collated_one_side`, `Dataset_npy_collated_single_pose` class headers and what they were for (one paragraph each), plus the verbatim source as a code block.
- The `compare_pred_gt_on_specific_type` debug method (verbatim).
- The `normalize_joints` `center_align=False` upstream default plus the apologia paragraph.

`arch_1_directions.md` gains a "current LR + aux schedule" subsection (one paragraph distillation of the active config) with a cross-link to `historical_bst.md` for the dated history.

**Safety checks:** `pytest`. Doc-only.

**Commit message draft:** "Capture excised BST history into historical_bst.md ahead of dead-code deletion."

---

### Step 5 — Delete dead BST code and orphan datasets

**Touches:** `tempose.py`, `shuttleset_dataset.py`, `bst_train.py`, `bst_infer.py`, `data_pipeline_to_model_train.md`.

- `src/bst_refactor/model/tempose.py`: delete `TemPose_V` (156-258), `TemPose_PF` (261-396), `TemPose_SF` (399-526), `TemPose_TF` (529-667). Keep `TCN`, `MLP`, `MLP_Head`, `FeedForward`, `TransformerEncoder`, the helper functions, and the `__main__` smoke check. Drop the unused `from torchinfo import summary  # noqa: F401`. Roughly 710 LOC down to ~200.
- `src/bst_refactor/stroke_classification/preparing_data/shuttleset_dataset.py`: delete `Dataset_npy` (142-246), `Dataset_npy_collated_one_side` (351-420), `Dataset_npy_collated_single_pose` (423-497), `prepare_npy_loaders` (500-531), `prepare_npy_collated_one_side_loaders` (568-599), `prepare_npy_collated_single_pose_loaders` (602-633). Update module docstring/header to reflect the trimmed surface.
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py`: delete `Task.compare_pred_gt_on_specific_type` (706-733) and the `Dataset_npy` import. Delete the commented-out `Hyp` block (85-101) and the commented-out scheduler (389-394).
- `src/bst_refactor/data_pipeline_to_model_train.md:260-263`: drop the `_one_side` / `_single_pose` references entirely now that they're gone.
- `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_infer.py`: confirm the import surface matches the trimmed `shuttleset_dataset.py` and `bst.py`.

**Safety checks:**
- `grep -rn "TemPose_V\|TemPose_PF\|TemPose_SF\|TemPose_TF" src/ scripts/ tests/ notebooks/` returns zero hits outside `historical_bst.md`.
- `grep -rn "Dataset_npy_collated_one_side\|Dataset_npy_collated_single_pose\|prepare_npy_loaders\|prepare_npy_collated_one_side_loaders\|prepare_npy_collated_single_pose_loaders" src/ scripts/ tests/ notebooks/` returns zero hits outside `historical_bst.md`.
- `grep -rn "Dataset_npy[^_]" src/ scripts/ tests/ notebooks/` (negative lookahead for `_collated`): zero hits outside `historical_bst.md`.
- `grep -rn "compare_pred_gt_on_specific_type" src/ scripts/ tests/ notebooks/`: zero hits outside `historical_bst.md`.
- Import smoke: `python -c "from src.bst_refactor.stroke_classification.preparing_data.shuttleset_dataset import Dataset_npy_collated, get_bone_pairs, POSE_BONE_MULTIPLIER; from src.bst_refactor.model.bst import BST_CG_AP; from src.bst_refactor.model.tempose import TCN, MLP, MLP_Head, FeedForward, TransformerEncoder; print('ok')"`.
- `pytest`. `tests/test_dataset.py` and `tests/test_integration.py` (auto-skip locally) both depend on `Dataset_npy_collated` only; they should remain green.

**Commit message draft:** "Drop dead BST code: TemPose variants, orphan datasets, compare-pred debug helper. ~850 LOC, no callers."

---

### Step 5b — `prepare_train_on_shuttleset.py` tidy (scope: light)

**Touches:** `prepare_train_on_shuttleset.py` only.

This step is folded into the branch at the user's direction. Scope is deliberately limited to (a) and (b) below; the full module split into `mmpose_extract.py` + `homography.py` + `collate.py` is reserved for after-X3D-S.

- (a) **Lift the per-clip iteration loop**. `prepare_2d_dataset_npy_from_raw_video` (lines 533-...) and `prepare_3d_dataset_npy_from_raw_video` (lines ...-657) share roughly 80% of their body: same iteration, same resume check, same gc-on-success path, diverging only in which detector is invoked and the joint dimensionality. Lift into one `_prepare_dataset_from_raw_video(detect_fn, joint_dim, ...)` helper; the two public functions become thin wrappers that pass `detect_players_2d` or `detect_players_3d`. The X3D-S wrist-crop layer will reuse this iteration shape.
- (b) **Collapse the `:1066-1116` mirror block**. This block mirrors `bst_train.py:744-783` (the model_info / npy_collated_dir construction). Done in lockstep with step 8 so the produced collated dir name stays byte-identical pre/post on both sides. Either factor a small shared helper or pass the collated-dir builder one way. Use whichever produces the smaller, more obvious diff.

**Out of scope:** the full module split; the homography helpers (`get_H` etc); the shuttle subprocess wrappers; any docstring rewrite of upstream-anchored functions. Those stay for after-X3D-S.

**Safety checks:**
- `pytest`.
- One-liner that constructs the npy collated dir name on a representative `Hyp` and confirms the string matches a known-good literal from a previous run's manifest.
- Import smoke: `python -c "from src.bst_refactor.stroke_classification.preparing_data.prepare_train_on_shuttleset import prepare_2d_dataset_npy_from_raw_video, prepare_3d_dataset_npy_from_raw_video, collate_npy, pad_and_augment_one_npy_video; print('ok')"`.
- Diff review: walk the two `prepare_*_dataset_npy_from_raw_video` functions and confirm the lifted helper produces the same per-clip filesystem outputs (same `_pos`/`_joints`/`_failed` filenames, same shapes, same dtypes, same iteration order).

**Commit message draft:** "Lift prepare_2d/prepare_3d shared iteration into _prepare_dataset_from_raw_video; collapse the model_info mirror block."

---

### Step 5c — `bst_common.py` extraction

**Touches:** new file `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_common.py`. Edits to `bst_train.py` and `bst_infer.py`.

This step is folded into the branch at the user's direction. The motivation, per `bst_infer.py:7-12`'s pre-existing TODO and the review doc: a third entry point (X3D-S training script) is about to land, and the shared scaffolding would otherwise be triple-copied.

- New module `bst_common.py` next to `bst_train.py` (same dir, minimum import-path churn).
- Lift into `bst_common.py`:
  - `MODELS` dict (constructed from the five `BST_*` partials).
  - The `Task` class (or its base; if Task has a small amount of train-specific state, keep that subclassed in `bst_train.py` and lift only the genuinely shared parts).
  - `Tee` stdout helper.
  - The `__main__` plumbing for run-id derivation, clips-CSV hashing, and the `track_run` call signature. The X3D-S training script will need the same.
  - Dataloader helper(s) currently inline in `bst_train.py`.
- `bst_train.py` becomes a thin entry point: import from `bst_common.py`, define the `Hyp` namedtuple (active config + the bst-train-specific knobs), wire up the run loop, call `track_run`. No copy of the lifted logic remains.
- `bst_infer.py` switches its imports to `bst_common.py` for `MODELS` and `Task`; the inference-specific path stays in `bst_infer.py`. The pre-existing TODO at `bst_infer.py:7-12` is removed (work done).

**Wire-in invariant (re-confirming):** the BST model graph itself is not touched. `MODELS[...](**kw)` stays the construction call. `Task.train()`, `Task.eval()`, etc. keep their public signatures. The X3D-S layer will be a sixth entry in `MODELS` later, not a wrapper around `bst_common`.

**Safety checks:**
- `pytest`.
- Import smoke: `python -c "from src.bst_refactor.stroke_classification.main_on_shuttleset.bst_common import MODELS, Task, Tee; from src.bst_refactor.stroke_classification.main_on_shuttleset.bst_train import Hyp; print(list(MODELS.keys())); print(Hyp._fields)"` — confirms `MODELS` keys are unchanged and `Hyp` fields are unchanged.
- `python -c "from src.bst_refactor.stroke_classification.main_on_shuttleset.bst_infer import *; print('ok')"` — bst_infer still imports cleanly.
- Diff review: `bst_train.py` line count drops materially; the dropped lines are entirely accounted for by the `bst_common.py` additions.

**Commit message draft:** "Extract bst_common.py: MODELS, Task, Tee, dataloader helpers, run-id and clips-CSV plumbing. Closes the bst_infer dedup TODO."

---

### Step 6 — Sticky_anchor unit tests (write tests **before** any sticky_anchor logic touches)

**Touches:** New file `tests/test_sticky_anchor.py`. No source-code edits.

Test list (sites + invariants only — see "Sticky_anchor Tier 1 test list" below for full detail). All tests use synthetic `RawClip` arrays with hand-built bbox/score data and a synthetic court rectangle; no MMPose, no real video frames, no GPU.

Roughly 6-8 tests. Should run in well under a second.

**Safety checks:**
- `pytest tests/test_sticky_anchor.py` green.
- `pytest` overall green.

**Commit message draft:** "Pin sticky_anchor invariants: Voronoi pick, Bottom-first ordering, sitting tiebreaker, EMA reset semantics."

---

### Step 7 — Sticky_anchor hyperparameter triplication collapse

**Touches:** `apply_heuristic.py`, `heuristics/sticky_anchor.py`, possibly `heuristics/base.py`.

- Introduce `@dataclass(frozen=True) StickyAnchorParams` (location: probably `heuristics/sticky_anchor.py` next to the existing `RawClip`/`ClipContext`/`HeuristicOutput`-style types in `base.py`).
- `apply_heuristic.py:265-287` — derive argparse `add_argument` calls from the dataclass fields plus their defaults, single source of truth.
- `sticky_anchor.py:262-269` — accept the dataclass instance instead of `**hyperparams`, drop the `hyperparams.get(..., default)` triplet.
- The `apply()` signature on `sticky_anchor.py` stays compatible with the registry contract (`apply(raw, ctx, **kw)`); construction of `StickyAnchorParams` happens at the boundary.

**Safety checks:**
- `pytest tests/test_sticky_anchor.py` green (the tests pinned the invariants; they catch any pick/EMA regression introduced by the refactor).
- `pytest` overall green.
- `grep` on every removed default literal to confirm it now lives in exactly one place.
- Import smoke: instantiate `StickyAnchorParams()` with defaults and confirm field values match the prior triplicated defaults exactly.

**Commit message draft:** "Collapse sticky_anchor hyperparameter triplication into StickyAnchorParams dataclass."

---

### Step 8 — `bst_train.py:744-783` model_info / npy_collated_dir block simplification

**Touches:** `bst_train.py` only.

- Drop the dead `additional_model_info` sentinel (top of `__main__`, never set anywhere).
- Collapse the three nested string-builders for `additional_model_info` / `train_partial` / `model_info` into a small helper or tagged tuple.
- The `match` on `seq_len` only handles 30 and 100; preserve that exactly (no behaviour change).

**Safety checks:**
- `pytest`.
- Manual diff review: the produced `npy_collated_dir` string for the active `Hyp` config must be identical pre/post. I will exercise this with a one-liner Python snippet that constructs the dict on a representative `Hyp` and prints the `npy_collated_dir` against the expected literal from a previous run's manifest.
- `prepare_train_on_shuttleset.py:1066-1116` is **not touched** in this step — it mirrors the same block but is deferred to after-X3D-S in the review doc. The mirror stays in sync because we did not change the produced string.

**Commit message draft:** "Simplify model_info / collated dir construction in bst_train __main__."

---

### Step 9 — Script archive sweep

**Touches:** Moves only. No source-code edits.

- `scripts/example_mlflow_run.py`: stays in place but gains a top-of-file header comment `"""TODO (delete before delivery if Scott has not picked up MLflow integration). Stub from 2026-04-08; project settled on Aim + run_tracker."""`. README experiment-tracking paragraph gains a one-line note in the same vein.
- `scripts/test_clip_index.py` → `scripts/archive/test_clip_index.py`.
- `scripts/flatten_copy.sh` → `scripts/archive/flatten_copy.sh`.
- `scripts/verify_flatten.py` → `scripts/archive/verify_flatten.py`.
- `scripts/symlink_merge_phase1.py` → `scripts/archive/symlink_merge_phase1.py`. While moving, fix the docstring at line 14 ("Run from the repo root or from `src/bst_refactor/stroke_classification/`" → "Run from the repo root.") since the relative defaults make the dual-location claim false.
- `scripts/verify_v1_collate.py` → `scripts/archive/verify_v1_collate.py`.
- `scripts/archive/README.md` (created in step 2) describes each one-line: what it did, when, why it's archived rather than deleted.
- `README.md`: update the scripts section to list only the active scripts (`rename_videos.py`, `validate_videos.py`, `setup_data.sh`, `example_mlflow_run.py`).

**Safety checks:**
- `pytest`.
- `grep -rn "scripts/flatten_copy\|scripts/verify_flatten\|scripts/symlink_merge_phase1\|scripts/verify_v1_collate\|scripts/test_clip_index" src/ tests/ notebooks/ README.md scratch/` — any hit must be a doc reference that can be updated to the archive path or removed; no live code path may import or shell out to these.

**Commit message draft:** "Archive completed-phase scripts; flag mlflow stub for delivery review."

---

### Step 10 — `bst_train.py` configuration block tidy (final pass)

**Touches:** `bst_train.py` and `arch_1_directions.md`.

This step depends on step 4 (historical_bst.md fill) and step 5 (dead-code excision). Done last in this branch because it's the most stylistic and any merge conflict here is cheapest to resolve.

- Move LR/aux-schedule rationale paragraphs (`bst_train.py:65-157`) into `arch_1_directions.md` (current state) and `historical_bst.md` (verbatim history). Keep only a one-line cross-link in `bst_train.py` plus the live `Hyp` config.
- Delete migration-anchor comments at `bst_train.py:1-2, 53-57`.
- Delete the task-anchored comment at `bst_train.py:151` (`# Aggressive CG/AP annealing — matches preferred config from run_20260418_151139.`).

**Safety checks:**
- `pytest`.
- Import smoke: `python -c "from src.bst_refactor.stroke_classification.main_on_shuttleset.bst_train import Hyp, MODELS, Task; print(Hyp._fields)"` — confirms the namedtuple fields are unchanged.

**Commit message draft:** "Move bst_train tuning rationale into arch_1_directions and historical_bst; trim configuration block."

---

### Steps 11+ (deferred, not executed in this branch)

These are explicitly **not** in scope per the review doc and your direction:

- Path/IO abstraction (focus area 5).
- `prepare_train_on_shuttleset.py` full module split into `mmpose_extract.py` + `homography.py` + `collate.py`. The light tidy (a) + (b) is in this branch as step 5b; the structural split is reserved for after-X3D-S.
- Validation script triplet shared core (focus area 6, site 5).
- Bulk style pass: AU/UK rename for `normalize_*` and "labeled"/"vectorized"; em-dash sweep; "fade" prose to "anneal" / "downtune".
- `aim_backfill._derive_tags` parameterisation.
- `pipeline/build_dataset.py:79-139` `dry_run()` consolidation.
- `pipeline/clip_index.py` docstring trim.

---

## Sticky_anchor Tier 1 test list

Sites and invariants only. The actual test code is written in step 6.

All tests live in `tests/test_sticky_anchor.py` and use synthetic data: small `RawClip` arrays (e.g. 4 frames, 3 candidate detections per frame) with hand-built `bbox`, `score`, and `joints` arrays, and a synthetic `ClipContext` carrying a fixed court rectangle and halfcourt centre. No MMPose, no real video, no GPU.

### Test 1 — Voronoi partition picks the right side

**Site:** `sticky_anchor.py:128-253` (`_pick_one_frame`), specifically the Voronoi partition at line 213.

**Invariant:** Given two candidates straddling the halfcourt line, the candidate above the line is assigned to the Top slot and the one below to the Bottom slot. Swapping the candidates' input order does not change the assignment.

**Synthetic setup:** Two candidates, one at `y = halfcourt_y - 50`, one at `y = halfcourt_y + 50`. Equal scores. `ClipContext.halfcourt_centre` set so both are inside the court rectangle.

### Test 2 — Bottom-first slot ordering with cross-slot exclusion

**Site:** `sticky_anchor.py:218-219`.

**Invariant:** When two candidates both fall on the Bottom side of the Voronoi partition, the Bottom slot fills first and the Top slot is left empty (or filled by a Top-side candidate from elsewhere); the Bottom-picked candidate is excluded from Top consideration in the same frame.

**Synthetic setup:** Two candidates both at `y > halfcourt_y`. Verify Bottom is filled, Top is empty (or fallback-filled per the rule).

### Test 3 — Sitting tiebreaker with fallback

**Site:** `sticky_anchor.py:232-237`.

**Invariant:** When two candidates are tied on the primary score, the "non-sitting" tiebreaker prefers the candidate whose pose passes the sitting filter; if all candidates fail the sitting filter, the fallback rule kicks in (whatever it is — capture the current behaviour exactly).

**Synthetic setup:** Two candidates with identical positions and scores; one with joints arranged to pass the sitting filter, one to fail. Verify the non-sitting one is picked. Then a second case where both fail; verify the fallback rule.

### Test 4 — Rally-presence rejection

**Site:** `sticky_anchor.py:242-248`.

**Invariant:** A candidate that fails the rally-presence check is not picked even when it would otherwise win on score.

**Synthetic setup:** Two candidates; the higher-score one is positioned so it fails the rally-presence check, the lower-score one passes. Verify the lower-score one is picked.

### Test 5 — EMA reset on full-frame failure

**Site:** `sticky_anchor.py:294`.

**Invariant:** When a frame produces zero successful picks, the EMA state for both slots resets to `halfcourt_centre`. The next frame's prediction is anchored at the centre, not at the previous frame's stale value.

**Synthetic setup:** Two-frame sequence. Frame 1 produces valid picks (EMA advances to non-centre values). Frame 2 has all candidates positioned outside the court (full-frame failure). After frame 2, query the internal EMA state; both slots are at `halfcourt_centre`.

### Test 6 — Mixed-pick reset only the unpicked slot

**Site:** `sticky_anchor.py:302-306`.

**Invariant:** When a frame picks a Bottom but no Top, the Top EMA resets to `halfcourt_centre` while the Bottom EMA advances normally. Symmetric for the inverse case.

**Synthetic setup:** Frame 1 establishes both slots non-centre. Frame 2 has only one Bottom-side candidate (no Top candidate found). After frame 2, Bottom EMA has advanced; Top EMA is at `halfcourt_centre`.

### Test 7 — `update_gate_eps` in-court guard before EMA update

**Site:** `sticky_anchor.py:315`.

**Invariant:** A picked candidate whose position is just outside the court rectangle (within `update_gate_eps`) does not update the EMA. The EMA stays at its previous value.

**Synthetic setup:** Pick a candidate just outside the court boundary. Confirm EMA does not move.

### Tests not written

- `current.py` byte-identity: covered by `failsafe_bst_mmpose_zeroing_check_equivalence.py` on real data; a synthetic stand-in would have to re-stub `check_pos_in_court` and `normalize_joints` and would drift from upstream.

---

## Test harness layout

Single new file: `tests/test_sticky_anchor.py`. Uses `pytest`. No fixtures shared with other test files.

The synthetic `RawClip` builder lives at the top of the file as a small helper:

```python
def _make_raw_clip(n_frames, candidates_per_frame, court, ...):
    """Build a RawClip with hand-specified bbox/score/joints arrays."""
    ...
```

Each test calls `_make_raw_clip(...)` then `sticky_anchor.apply(raw, ctx, ...)` and asserts on the returned `HeuristicOutput`.

If `sticky_anchor.apply()` does not currently expose internal EMA state for tests 5-7 to inspect, I will add a minimal accessor (e.g. return the EMA history alongside the output, or expose a `_state` attribute on a debug path). Documented in the commit message.

---

## What this plan does not do

- **No edits to the BST model graph itself** (`bst.py` `BST_CG_AP` forward, `CrossTransformerLayer`, the building blocks, the five `BST_*` partials). This branch addresses sprawl and dead code in the surrounding scaffolding only; the model is read-only.
- **No abstraction work on paths.** Reserved for after-X3D-S.
- **No full module split of `prepare_train_on_shuttleset.py`.** The light tidy is in step 5b; the structural split is reserved.
- **No bulk rename.** AU/UK and "fade" prose passes are reserved.
- **No new tests beyond sticky_anchor.** No coverage push for `current.py`, datasets, or training loops in this branch.

If anything in this list surprises you, ping me before I cut the branch.
