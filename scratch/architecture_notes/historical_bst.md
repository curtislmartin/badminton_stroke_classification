# Historical BST Reference

**Purpose.** This document preserves BST-origin and pre-Phase-2 content that has been excised from the active source tree. It is the canonical reference for the end-of-project report and diary, and for any reproduction or revert work.

Each section captures: where the content originally lived (file:line at the time of excision), why it was preserved through earlier phases, why it was removed, and the verbatim source as a fenced code block.

This file is **read-only history**. Do not edit excised content to reflect later changes; current state lives in the active source tree and in `arch_1_directions.md`.

---

## Status

Skeleton drafted 2026-04-26. Sections are filled by the pre-phase-2 tidy execution plan as content is excised. Until then, every section below is a placeholder.

---

## 1. TemPose variant classes (deleted from `model/tempose.py`)

**Original location:** `src/bst_refactor/model/tempose.py:156-667` (four classes, ~510 LOC).

**Why preserved through phase 0/1:** byte-identity reproduction with the upstream BST repo. TemPose is the BST paper's predecessor and its source was kept verbatim alongside `bst.py` so any backed-out comparison run could fall back to the original code.

**Why removed pre-phase-2:** none of the four standalone classes (`TemPose_V`, `TemPose_PF`, `TemPose_SF`, `TemPose_TF`) are imported anywhere outside `tempose.py`'s own `__main__` smoke check. `bst.py:28` only consumes the building-block utilities (`TCN`, `MLP`, `MLP_Head`, `FeedForward`, `TransformerEncoder`), which stay. TemPose is not a project baseline; the apples-to-apples preservation goal is served by the `BST_*` partials in `bst.py:433-437`.

### 1.1 `TemPose_V`

*To be filled by Step 4 of the tidy execution plan.*

```python
# verbatim source from src/bst_refactor/model/tempose.py:156-258 at SHA <to-be-filled>
```

### 1.2 `TemPose_PF`

*To be filled.*

### 1.3 `TemPose_SF`

*To be filled.*

### 1.4 `TemPose_TF`

*To be filled.*

---

## 2. Original BST `Hyp` namedtuple defaults

**Original location:** `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py:85-101` (commented-out block).

**Why preserved:** the BST paper's published numbers are produced with these defaults; reproducing those numbers requires this exact configuration.

**Why removed:** by 2026-04-26 the active config has diverged enough that the commented-out block is misleading reference noise. The "no backwards-compat shims for unshipped code" principle applies; the values live here for the historical record.

*Verbatim block to be filled by Step 4.*

```python
# Hyp(...) verbatim from bst_train.py:85-101 at SHA <to-be-filled>
```

**Reproducing the BST paper's published numbers** today: branch off from a pre-tidy commit (most recent is `d4fd644`), use the `Hyp` defaults captured in this section, and run on the BST taxonomy (`merged_25`) with the BST split column. Phase 1 demonstrated this reproduces the paper's headline numbers; the most recent phase-1 reproduction run is logged in the experiment manifest.

---

## 3. LR-schedule and aux-schedule retune rationale (excised from `bst_train.py`)

**Original location:** `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py:65-157`.

**Why preserved through phase 1:** the dated rationale paragraphs ("LR-SCHEDULE RETUNE 2026-04-17", "AUX-SCHEDULE 2026-04-18") record decisions that materially affected the active config. Useful for ablation interpretation and for picking up the work after a long context gap.

**Why moved out:** the rationale belongs in a writeup-style document, not in the configuration block of the live training script. `arch_1_directions.md` carries the current-state distillation; the verbatim dated history lives here for the report.

### 3.1 LR-schedule retune (2026-04-17)

*To be filled.*

```text
# verbatim block from bst_train.py:65-... at SHA <to-be-filled>
```

### 3.2 Aux-schedule retune (2026-04-18)

*To be filled.*

```text
# verbatim block from bst_train.py:... at SHA <to-be-filled>
```

### 3.3 Cross-link

Current state: `scratch/architecture_notes/arch_1_directions.md` (under "current LR + aux schedule").

---

## 4. Orphan dataset classes (deleted from `shuttleset_dataset.py`)

**Original location:** `src/bst_refactor/stroke_classification/preparing_data/shuttleset_dataset.py`.

**Why preserved through phase 1:** these were experimental dataset variants for ablation studies that never made it onto the active path. Kept in case the ablation revived.

**Why removed:** no callers in active code; the variants assume `unknown_first=True` and break under `une_merge_v1_nosides` (the current default taxonomy).

### 4.1 `Dataset_npy` (lines 142-246)

**Purpose:** pre-flatten loader that read directly from the nested directory layout. Superseded by `Dataset_npy_collated`, which reads the master CSV's split and label columns.

*Verbatim source to be filled.*

### 4.2 `Dataset_npy_collated_one_side` (lines 351-420)

**Purpose:** ablation variant that exposed only one player's pose stream per sample. Used in the "single-side ablation" study (date and run id to be looked up).

*Verbatim source to be filled.*

### 4.3 `Dataset_npy_collated_single_pose` (lines 423-497)

**Purpose:** ablation variant that exposed a single concatenated pose stream rather than two-player pose. Companion to `_one_side`.

*Verbatim source to be filled.*

### 4.4 Loader helpers

`prepare_npy_loaders` (lines 500-531), `prepare_npy_collated_one_side_loaders` (lines 568-599), `prepare_npy_collated_single_pose_loaders` (lines 602-633).

*Verbatim sources to be filled.*

---

## 5. `compare_pred_gt_on_specific_type` debug helper

**Original location:** `src/bst_refactor/stroke_classification/main_on_shuttleset/bst_train.py:706-733`.

**Purpose:** Debug helper that loaded `Dataset_npy` and compared per-sample predictions against ground truth for a chosen stroke type.

**Why preserved:** scratch debugging during phase 1 ablation passes.

**Why removed:** unreachable from the active run path; the only consumer of `Dataset_npy`. Removing it is what unblocks deleting `Dataset_npy` itself.

*Verbatim source to be filled.*

---

## 6. `normalize_joints` upstream default (changed in `prepare_train_on_shuttleset.py`)

**Original location:** `src/bst_refactor/stroke_classification/preparing_data/prepare_train_on_shuttleset.py:170-175`.

**Original signature:** `normalize_joints(..., center_align=False)` (matching upstream BST verbatim).

**Production behaviour:** every active caller passes `joints_center_align=True`. The signature default has been misleading since the refactor.

**Change:** flip the default to `True`. The apologia paragraph is removed from the source. This file records that the upstream default was `False` so reproductions of the original BST paper using upstream defaults remain reproducible.

*Apologia paragraph (verbatim from `prepare_train_on_shuttleset.py:169-174`) to be filled.*

---

## 7. Migration anchors and task-anchored comments removed from `bst_train.py`

**Original locations:**
- `bst_train.py:1-2`: `# Consolidated BST training script for ShuttleSet` / `# Replaces: bst_main.py, bst_main_summary_writer.py, bst_backbone_main.py`.
- `bst_train.py:53-57`: refactor cross-ref to `scratch/architecture_notes/completed_general_refactors/dir_flatten_refactor.md`.
- `bst_train.py:151`: `# Aggressive CG/AP annealing — matches preferred config from run_20260418_151139.`

**Why preserved:** lineage record during the multi-step phase-1 consolidation.

**Why removed:** the migrations are done; the run id reference rots as new ablation runs supersede it.

The completed_general_refactors directory at `scratch/architecture_notes/completed_general_refactors/` still holds the long-form refactor notes; this file just records that the in-source pointers to it have been removed.

---

## 8. Project-history relocations (no source-code change)

**2026-04-26:** moved out of `src/` into `scratch/project_history/`:

- `src/bst_refactor/deprecated/` → `scratch/project_history/bst_refactor_deprecated/`
- `src/bst_refactor/ShuttleSet/deprecated/` → `scratch/project_history/shuttleset_deprecated/`
- `src/bst_refactor/stroke_classification/main_on_shuttleset/tmp/` → `scratch/project_history/main_on_shuttleset_tmp/`

The relocated trees include:
- The original BST author's pre-phase-1 scripts (`gen_my_dataset.py`, `get_each_class_total.py`, etc.).
- The pre-flatten snapshot of `pipeline/` and `stroke_classification/` (`before_flattening_asset_dirs/`).
- Phase-0 historical documentation (`outdated_bst_repo_reusability_assessment.md`, `outdated_bst_models_refactor.md`, `outdated_pipeline_build.md`, `historical_README_bst_original.md`, `historical_predecessor_analysis_summary.md`).
- The `tmp/` smoke tests (`test_dataloader.py`, `test_fwd.py`, `test_train_step.py`).

Original locations are recorded in `scratch/project_history/README.md`.

---

## Cross-references

- `scratch/architecture_notes/arch_1_directions.md` — current Architecture 1 state and recent decision history.
- `scratch/architecture_notes/pipeline_context_notes.md` — pipeline-area excisions (separate from BST-core).
- `scratch/architecture_notes/pre_phase_2_review_2026-04-26.md` — review that drove the tidy pass.
- `scratch/architecture_notes/pre_phase_2_tidy_plan.md` — execution plan for the tidy pass.
- `.claude/project_overview.md` — project handover document.

---

## Maintenance

Append new sections at the end as further excisions happen. Do not edit existing sections to reflect later changes; spawn new sections cross-linked to the old ones if behaviour is later restored or revised.
