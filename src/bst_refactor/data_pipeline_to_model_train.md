# Data Pipeline to Model Training: Module Reference

End-to-end walkthrough of the modules needed to go from raw ShuttleSet data to a trained BST model, with notes on where a custom (non-BST) architecture would diverge.

---

## Quick Start: End-to-End Execution

The project uses three separate Python environments because the OpenMMLab stack (MMPose) requires numpy < 2.0 while the training stack uses numpy 2.x. All three target **Python 3.11**.

| Environment | Requirements file | Purpose |
|---|---|---|
| **Pipeline** | `pipeline/requirements.txt` | Download videos, generate clips, verify output |
| **MMPose** | `stroke_classification/preparing_data/requirements.txt` | Pose estimation (steps 1-2 of data preparation) |
| **BST training** | `stroke_classification/requirements.txt` | Collation, training, inference. Also shared by TrackNetV3. |

### Environment setup

```bash
# 1. Pipeline venv
python3.11 -m venv venv-pipeline
source venv-pipeline/bin/activate
pip install -r pipeline/requirements.txt

# 2. MMPose venv (requires C++ compiler + CUDA toolkit for mmcv source build)
python3.11 -m venv venv-mmpose
source venv-mmpose/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
mim install mmcv==2.1.0
pip install -r stroke_classification/preparing_data/requirements.txt

# 3. BST training venv
python3.11 -m venv venv-bst
source venv-bst/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r stroke_classification/requirements.txt
```

### Execution order

```bash
# ── Stage 1: Build dataset (pipeline venv) ──────────────────────────
source venv-pipeline/bin/activate

python -m pipeline.build_dataset --dry-run                # preview
python -m pipeline.build_dataset --skip-shuttle            # download + clips + verify
# Optional: shuttle extraction (uses BST venv for TrackNetV3)
python -m pipeline.build_dataset --skip-download \
    --tracknet-dir TrackNetV3 \
    --tracknet-python /path/to/venv-bst/bin/python

# ── Stage 2: Pose estimation (MMPose venv) ──────────────────────────
source venv-mmpose/bin/activate
cd stroke_classification

python -m preparing_data.prepare_train_on_shuttleset \
    --skip-trajectory --skip-collate                       # pose only

# ── Stage 3: Collation + training (BST venv) ────────────────────────
source venv-bst/bin/activate
cd stroke_classification

python -m preparing_data.prepare_train_on_shuttleset \
    --skip-trajectory --skip-pose                          # collate only

cd main_on_shuttleset
python bst_train.py                                        # train (5 serial trials)
python bst_infer.py                                        # inference
```

Each stage's output feeds the next. Stages are independently re-runnable — use `--skip-*` flags to avoid repeating completed work.

---

## Part 1: BST on ShuttleSet

### Stage 1 -- Build the Dataset (`pipeline/`)

The pipeline downloads match videos, cuts them into labeled stroke clips, optionally extracts shuttle trajectories, and verifies the result. All configuration is centralized in `config.py`; the orchestrator `build_dataset.py` runs the steps in sequence.

#### Modules

| Module | Role | Key functions / concepts |
|--------|------|--------------------------|
| `config.py` | Single source of truth for paths, stroke types, splits, flaw records, and merge rules. Every other pipeline module imports from here. | `Taxonomy` (frozen dataclass with `class_list()`, `n_classes`, `merge_map`, `standalone_set`), `TAXONOMIES` (dict of named taxonomies: `'une_merge_v1'`, `'merged_25'`, `'raw_35'`), `DEFAULT_TAXONOMY` (name of the default taxonomy, currently `'une_merge_v1'`), `UNPREFIXED_TYPES` (frozenset of raw types that never get Top_/Bottom_ prefixed folders during clip generation), `SPLITS` (train/val/test video ID lists, auto-stripped of excluded videos), `UNE_MERGE_V1_MAP` (default 19 -> 14 class reduction), `MERGE_MAP` (legacy 19 -> 12 class reduction), `EN_TO_ZH` / `ZH_TO_EN` (English-Chinese name mapping for CSV I/O only), `parse_flaw_records()` (reads `flaw_shot_records.csv` to populate `EXCLUDED_VIDEOS` and `REMOVED_SHOTS`). |
| `build_dataset.py` | One-command orchestrator. Runs steps 1-6 in order with CLI flags to skip individual steps. | `run_pipeline()` (main entry point), `dry_run()` (preview without side effects), `_validate_inputs()` (fail-fast checks before long work). |
| `download_videos.py` | Downloads 40 ShuttleSet match videos from YouTube via yt-dlp. Also builds a resolution CSV by scanning each video with OpenCV. | `download_all_videos(max_workers)`, `build_resolution_csv()`. Output: `ShuttleSet/raw_video/{id} {match_name}.mp4` and `ShuttleSet/my_raw_video_resolution.csv`. |
| `clip_generator.py` | Extracts individual stroke clips from full match videos. Reads ShuttleSet CSV annotations (Chinese column names), maps A/B players to Top/Bottom, filters excluded videos and removed shots, and organizes clips into `{split}/{Player}_{stroke_type}/` folders. | `generate_all_clips()`, `apply_class_merge()` (moves clips from rare subtype folders into their parent type folders per the active taxonomy's merge map). Three clip window modes: `middle_in_a_sec`, `between_2_hits`, `between_2_hits_with_max_limits` (default, clamps to 1.5s each side). |
| `player_mapping.py` | Maps the A/B player labels in ShuttleSet annotations to Top/Bottom court positions. Handles set-3 court switches. | `get_top_bottom_mapping(video_id, set_num)`. |
| `verify.py` | Post-generation sanity checks: all splits present, no clips from excluded videos, no removed shots, merged subtype folders empty, no orphan files. | `verify_splits_present()`, `verify_no_excluded()`, `verify_no_removed_shots()`, `verify_class_merge()`, `verify_shuttle_sync()`, `print_dataset_summary()`. |
| `shuttle_extractor.py` | Runs TrackNetV3 on each clip to detect shuttle positions, then converts CSVs to normalized `(t, 3)` numpy arrays `[x_norm, y_norm, visibility]`. TrackNetV3 shares the BST training venv and is called as a subprocess via `--tracknet-python`. | `extract_all_shuttles(tracknet_dir, tracknet_python, max_workers)`, `shuttle_csvs_to_npy()`. Output: `ShuttleSet/shuttle_npy/{split}/{Player}_{stroke_type}/{clip}.npy`. |
| `court_utils.py` | Optional. Homography-based camera-to-court coordinate projection. Not required for the core pipeline. | `project_to_court()`, `normalize_court_position()`. |

#### Pipeline output structure

```
ShuttleSet/
  raw_video/                         # Full match videos
  my_raw_video_resolution.csv        # Width/height per video
  clips/                             # Labeled stroke clips
    train/{Top,Bottom}_{type}/*.mp4
    val/{Top,Bottom}_{type}/*.mp4
    test/{Top,Bottom}_{type}/*.mp4
  shuttle_npy/                       # Shuttle trajectories (optional)
    train/{Top,Bottom}_{type}/*.npy
    val/{Top,Bottom}_{type}/*.npy
    test/{Top,Bottom}_{type}/*.npy
```

#### Key concepts

- **Class merging**: The default taxonomy (`une_merge_v1`) folds 4 rare subtypes into parent types, reducing 19 raw types to 14 merged types (29 classes with Top/Bottom prefixes + `unknown`). The legacy `merged_25` taxonomy folds 6 subtypes down to 12 types (25 classes). The `raw_35` taxonomy applies no merging (35 classes).
- **Flaw records**: `flaw_shot_records.csv` is the single source of truth for data exclusions. Whole-video exclusions and individual shot removals are parsed at import time.
- **Clip windows**: Control how much temporal context surrounds each stroke. `between_2_hits_with_max_limits` (default) uses the interval between adjacent shots, clamped to 1.5s per side.
- **Video resolution**: The pipeline downloads the best available mp4 (video-only, no audio). Downstream models resize frames internally — TrackNetV3 to 512x288 (`TrackNetV3/utils/general.py`), MMPose to ~256x192 depending on model config — so resolutions above 720p provide no practical benefit while increasing file size and processing time.

---

### Stage 2 -- Prepare Training Data (`stroke_classification/preparing_data/`)

The pipeline produces **video clips** and **shuttle .npy files**. BST does not operate on raw video -- it needs pre-extracted skeletal pose, court position, and shuttle trajectory arrays. This stage bridges the gap.

#### Module

| Module | Role | Key functions / concepts |
|--------|------|--------------------------|
| `prepare_train_on_shuttleset.py` | Runs MMPose on each clip to extract 2D (or 3D) player keypoints, combines them with shuttle trajectories, normalizes everything, and collates per-sample arrays into batch-ready `.npy` files. | **Step 1**: `prepare_trajectory()` -- run TrackNetV3 on clips (if shuttle extraction wasn't done in the pipeline stage). **Step 2**: `prepare_2d_dataset_npy_from_raw_video()` -- run MMPose pose estimation, extract court positions via homography, normalize joints by bounding box, save per-clip `_joints.npy`, `_pos.npy`, `_shuttle.npy`. **Step 3**: `collate_npy(taxonomy=...)` -- pad all samples to uniform `seq_len`, compute bone vectors and interpolated joints, stack into single arrays per split. The `taxonomy` parameter (a `Taxonomy` instance from `pipeline.config`) determines the class list for label assignment. MMPose resizes input frames internally (typically 256x192 for RTMPose COCO-17), so video resolution does not affect pose estimation quality beyond ~720p. |

#### CLI usage

Run from `stroke_classification/`:

```bash
# Preview what would be done:
python -m preparing_data.prepare_train_on_shuttleset --dry-run

# Run only collation with the default taxonomy (une_merge_v1):
python -m preparing_data.prepare_train_on_shuttleset --skip-trajectory --skip-pose

# Full run with TrackNetV3:
python -m preparing_data.prepare_train_on_shuttleset --tracknet-dir /path/to/TrackNetV3
```

Key flags: `--seq-len` (30 or 100), `--taxonomy` (`une_merge_v1`, `merged_25`, or `raw_35`), `--use-3d-pose`, `--skip-trajectory`, `--skip-pose`, `--skip-collate`, `--clips-dir`, `--tracknet-dir`, `--dry-run`.

#### Data transformations in detail

1. **Pose detection** (`detect_players_2d`): MMPose extracts 17 COCO keypoints per frame. Players are identified by court projection of their feet -- only the two players whose feet project inside the court boundaries are kept, ordered Top-first by y-coordinate.

2. **Joint normalization** (`normalize_joints`): Keypoints are normalized relative to the player's bounding box diagonal. Optionally center-aligned.

3. **Shuttle normalization** (`normalize_shuttlecock`): Shuttle xy divided by video resolution to get [0,1] range. Frames where pose detection failed are zeroed out.

4. **Padding and augmentation** (`pad_and_augment_one_npy_video`): Each sample is padded (or strided) to a fixed `seq_len` (30 or 100 frames). Four pose representations are pre-computed:
   - `J_only`: raw joints `(t, 2, 17, 2)`
   - `JnB_interp`: joints + bone midpoints `(t, 2, 36, 2)`
   - `JnB_bone`: joints + bone vectors `(t, 2, 36, 2)`
   - `Jn2B`: interpolated joints + bone vectors `(t, 2, 55, 2)`

5. **Collation** (`collate_npy`): All samples in a split are stacked into single arrays and saved:
   - `{pose_style}.npy`, `pos.npy`, `shuttle.npy`, `videos_len.npy`, `labels.npy`

#### Collated output structure

```
preparing_data/ShuttleSet_data_{taxonomy.name}/dataset_npy_collated/
  train/
    J_only.npy, JnB_interp.npy, JnB_bone.npy, Jn2B.npy
    pos.npy, shuttle.npy, videos_len.npy, labels.npy
  val/
    ...
  test/
    ...
```

For example, `ShuttleSet_data_une_merge_v1/`, `ShuttleSet_data_merged_25/`, or `ShuttleSet_data_raw_35/`.

---

### Stage 3 -- Dataset Loading (`stroke_classification/preparing_data/shuttleset_dataset.py`)

Bridges collated `.npy` files to PyTorch `DataLoader`s. Imports `Taxonomy` from `pipeline.config` for class list construction.

#### Key classes and functions

| Name | Role |
|------|------|
| `Dataset_npy_collated` | Primary Dataset class for BST. Loads pre-collated arrays from disk. Supports `train_partial` to use a fraction of training data. Returns `(human_pose, pos, shuttle), video_len, label` per sample. |
| `Dataset_npy_collated_one_side` | Single-side variant: filters to Top or Bottom labels only (halves the dataset). **Requires `unknown_first=True`** — asserts at init. Uses the position of `'unknown'` in `class_list()` to find the Top/Bottom label boundary. |
| `Dataset_npy_collated_single_pose` | Extracts only the acting player's pose per sample (Top or Bottom). **Requires `unknown_first=True`** — same label-boundary assumption as `Dataset_npy_collated_one_side`. |
| `Dataset_npy` | Alternative that loads per-clip `.npy` files on-the-fly (slower, but doesn't require pre-collation). Accepts a `taxonomy` parameter for label indexing. Applies `RandomTranslation` during training. |
| `prepare_npy_collated_loaders()` | Convenience function: creates train/val/test `DataLoader`s from a collated directory. |
| `make_seq_len_same()` | Pads or strides a sample to match `seq_len`. Shared between `Dataset_npy` and `collate_npy`. |
| `create_bones()` / `interpolate_joints()` | Bone vector and midpoint computation from joint arrays. |
| `POSE_BONE_MULTIPLIER` | Dict mapping pose style names to bone-set multipliers: `{'J_only': 0, 'JnB_bone': 1, 'JnB_interp': 1, 'Jn2B': 2}`. Used by train/infer scripts to compute `in_dim`. |
| `pad_class_labels()` | Pads class label strings to uniform width for aligned F1 display. |
| `RandomTranslation` / `RandomTranslation_batch` | Data augmentation: small random xy shifts applied to joint coordinates during training. |

#### Tensor shapes at model input

```
human_pose:  (batch, seq_len, 2, n_pose_features, 2)  ->  flattened to (batch, seq_len, 2, in_dim)
pos:         (batch, seq_len, 2, 2)
shuttle:     (batch, seq_len, 2)
video_len:   (batch,)
labels:      (batch,)
```

---

### Stage 4 -- Model (`stroke_classification/model/`)

#### Modules

| Module | Role |
|--------|------|
| `tempose.py` | Building blocks reused by BST: `TCN` (dilated 1D temporal convolutions), `MLP`, `MLP_Head` (LayerNorm + MLP), `FeedForward` (MLP + Dropout), `MultiHeadAttention`, `TransformerLayer`, `TransformerEncoder`. Also contains standalone TemPose variants (`TemPose_V`, `TemPose_PF`, `TemPose_SF`, `TemPose_TF`). |
| `bst.py` | The BST model. Imports `TCN`, `FeedForward`, `MLP`, `MLP_Head`, `TransformerEncoder` from `tempose.py`. Adds `MultiHeadCrossAttention` and `CrossTransformerLayer` for player-shuttle interaction. Also defines pre-configured variant partials (`BST_0`, `BST_PPF`, `BST_CG`, `BST_AP`, `BST_CG_AP`) — these are the single source of truth for variant flag combinations, imported by the train/infer scripts. |

#### BST architecture (forward pass)

1. **PPF (Pose Position Fusion)** -- optional: projects court positions to `in_dim` via MLP, multiplies with skeleton features (multiplicative fusion with residual).
2. **TCN feature extraction**: separate TCNs for pose `(b*n, in_dim, t) -> (b*n, d_model, t)` and shuttle `(b, 2, t) -> (b, d_model, t)`.
3. **Temporal Transformer**: each of the 3 streams (player1, player2, shuttle) gets a learnable CLS token prepended, positional embeddings added, then processed by shared self-attention layers independently. Padding mask prevents attention to zero-padded frames.
4. **Cross Transformer**: each player's frame-level representation attends to the shuttle's representation via cross-attention (player queries, shuttle provides keys+values).
5. **Interactional Transformer**: combines player-shuttle interactions across players with another CLS token and self-attention.
6. **CG (Clean Gate)** -- optional: subtracts shared player noise from shuttle CLS via learned MLP.
7. **AP (Aim Player)** -- optional: weights player contributions by cosine similarity to shuttle CLS.
8. **MLP Head**: concatenated CLS tokens -> LayerNorm -> MLP -> class logits.

#### BST variants

```python
BST_0     = BST(use_ppf=False, use_cg=False, use_ap=False)  # Bare transformer
BST_PPF   = BST(use_ppf=True,  use_cg=False, use_ap=False)  # + Pose Position Fusion
BST_CG    = BST(use_ppf=True,  use_cg=True,  use_ap=False)  # + Clean Gate
BST_AP    = BST(use_ppf=True,  use_cg=False, use_ap=True)   # + Aim Player
BST_CG_AP = BST(use_ppf=True,  use_cg=True,  use_ap=True)   # Full model
```

#### Key hyperparameters (defaults from `bst_train.py`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `d_model` | 100 | Hidden dimension throughout |
| `d_head` | 128 | Dimension per attention head |
| `n_head` | 6 | Number of attention heads |
| `depth_tem` | 2 | Temporal transformer layers |
| `depth_inter` | 1 | Interactional transformer layers |
| `drop_p` | 0.3 | Dropout rate |
| `tcn_kernel_size` | 5 | TCN convolution kernel |

---

### Stage 5 -- Training (`stroke_classification/main_on_shuttleset/bst_train.py`)

#### Key components

| Name | Role |
|------|------|
| `Hyp` (namedtuple) | Experiment hyperparameters: `n_epochs=1600`, `batch_size=128`, `lr=5e-4`, `warm_up_step=400`, `early_stop_n_epochs=300`, `taxonomy=DEFAULT_TAXONOMY` (key into `TAXONOMIES`, currently `'une_merge_v1'`), `seq_len=30`, `pose_style='JnB_bone'`, `train_partial=0.25`. Edit these to configure experiments. |
| `train_one_epoch()` | Standard PyTorch training loop: forward pass, cross-entropy loss (with label smoothing 0.1), backward, optimizer step, scheduler step. Applies `RandomTranslation_batch` to joints (not bones). |
| `validate()` | Evaluates on val set. Accumulates per-class TP/FP/FN across batches, computes macro F1 and min-class F1. |
| `test()` | Runs inference on test set, returns `(predictions, ground_truth)` tensors. |
| `train_network()` | Full training loop with AdamW optimizer, cosine LR schedule with warmup, early stopping on macro F1, TensorBoard logging, and best-checkpoint saving. |
| `MODELS` (dict) | Maps variant names (`'BST_0'`, `'BST'`, etc.) to pre-configured partials imported from `bst.py`. Used by `get_network_architecture()` to instantiate the model without local flag dicts. |
| `Task` (class) | Orchestrates the full workflow: `prepare_dataloaders()` -> `get_network_architecture()` -> `seek_network_weights()` (loads existing or trains) -> `test()`. |

#### Training flow

```
Task()
  .prepare_dataloaders(root_dir, pose_style, train_partial)
  .get_network_architecture(model_name='BST_CG_AP', in_channels=2)
  .seek_network_weights(model_info, serial_no)   # trains if no checkpoint found
  .test(show_details, show_confusion_matrix)
  .test_topk_acc(k=2)
```

The `__main__` block runs 5 serial trials (serial_no 1-5) to measure variance.

---

### Stage 6 -- Inference (`stroke_classification/main_on_shuttleset/bst_infer.py`)

Lightweight script for loading a trained checkpoint and predicting stroke types. Suitable as a Gradio backend.

| Name | Role |
|------|------|
| `infer()` | Runs the model in eval mode on a DataLoader, returns predicted class indices. |
| `Task` (class) | `prepare_loader()` -> `get_network_architecture()` -> `load_weight()` -> `infer()`. |

---

### Stage 7 -- Results (`stroke_classification/result_utils.py`)

| Name | Role |
|------|------|
| `show_f1_results()` | Displays per-class and macro/min F1 scores as a pandas DataFrame. |
| `plot_confusion_matrix()` | Generates side-by-side precision and recall confusion matrices using matplotlib. |

---

### Full dependency chain (BST on ShuttleSet)

```
pipeline/config.py                     # Taxonomy, stroke types, splits, paths, merge map
    |
    v
pipeline/build_dataset.py             # Orchestrates Steps 1-6 (--taxonomy flag)
  -> download_videos.py               # Step 1: yt-dlp download
  -> clip_generator.py                # Steps 3-4: clip extraction + class merge
     -> player_mapping.py             # A/B -> Top/Bottom
  -> verify.py                        # Step 5: sanity checks
  -> shuttle_extractor.py             # Step 6: TrackNetV3 shuttle detection
    |
    v  (produces ShuttleSet/clips/ and ShuttleSet/shuttle_npy/)
    |
preparing_data/prepare_train_on_shuttleset.py  (--taxonomy flag)
  -> MMPose (2D/3D pose estimation)   # Extract joints, positions
  -> collate_npy(taxonomy=...)         # Pad, augment, stack into batch arrays
    |
    v  (produces preparing_data/ShuttleSet_data_{taxonomy.name}/dataset_npy_collated/)
    |
preparing_data/shuttleset_dataset.py  # PyTorch Dataset + DataLoader wrappers
  -> pipeline.config                  # Imports Taxonomy, TAXONOMIES
    |
    v
model/tempose.py                      # TCN, MLP, TransformerEncoder, etc.
model/bst.py                          # BST model (imports tempose building blocks)
    |
    v
main_on_shuttleset/bst_train.py       # Training loop (taxonomy in Hyp namedtuple)
main_on_shuttleset/bst_infer.py       # Inference from checkpoint
    |
    v
result_utils.py                       # F1 scores, confusion matrices
```

---

## Part 2: Adapting for a Custom (Non-BST) Model

### What stays the same

- **The entire `pipeline/` directory.** The pipeline produces labeled video clips and shuttle trajectories. It is model-agnostic -- it doesn't know or care what architecture consumes its output.
- **`pipeline/config.py`** remains the single source of truth for stroke types, class labels, splits, and merge rules. Your custom dataset loader should import from here to stay in sync.
- **`result_utils.py`** works with any model that produces `(predictions, ground_truth)` tensors. `show_f1_results()` and `plot_confusion_matrix()` are architecture-agnostic.

### What changes or may be replaced

#### 1. Data preparation (`prepare_train_on_shuttleset.py`)

This is the most likely point of divergence.

- **If your model operates on raw video** (e.g. a video transformer, 3D CNN, or SlowFast): you can skip pose estimation entirely. Load clips directly from `ShuttleSet/clips/` using a standard video DataLoader. The folder structure already encodes labels via directory names (`{Player}_{stroke_type}`).

- **If your model uses different input features**: you may need different preprocessing. For example, optical flow, different skeleton formats (not COCO-17), or different normalization schemes. Write your own preparation script, but reuse `pipeline.config` for label definitions.

- **If your model uses pose but at different granularity**: the existing `collate_npy()` supports 4 pose styles (J_only, JnB_interp, JnB_bone, Jn2B). If these suffice, you can reuse the collated arrays directly. If not (e.g., you need raw unnormalized keypoints, or a different skeleton topology), modify the preparation step.

#### 2. Dataset class (`shuttleset_dataset.py`)

BST's dataset classes return a specific tuple format: `(human_pose, pos, shuttle), video_len, label`.

- **If your model expects different inputs**: write a new Dataset class. Key decisions:
  - Does your model need all 3 input streams (pose, position, shuttle)? BST uses all three. TemPose variants use subsets.
  - Does your model handle variable-length sequences internally (e.g. via packed sequences or attention masks), or does it need pre-padded fixed-length input? BST uses fixed-length padding + a `video_len` mask.
  - Does your model operate on pre-collated batched arrays, or per-clip files? The `Dataset_npy_collated` class loads everything into RAM at init; `Dataset_npy` loads per-clip lazily.

- **Label list construction**: All class labels are now English. Use `taxonomy.class_list()` from any `Taxonomy` instance in `pipeline.config.TAXONOMIES` to get the label list. The default is `TAXONOMIES[DEFAULT_TAXONOMY]` (`une_merge_v1`, 29 classes). To add a custom taxonomy, define it in `pipeline/config.py` (see the `Taxonomy` dataclass and existing instances for the pattern).

#### 3. Model architecture (`model/`)

Replace `bst.py` (and optionally `tempose.py`) with your own architecture.

- **Reusable building blocks from `tempose.py`**: `TCN`, `MLP`, `MLP_Head`, `FeedForward`, `TransformerEncoder` are generic components. If your custom model is transformer-based, you can import these directly rather than reimplementing.

- **BST-specific components you'd replace**: `MultiHeadCrossAttention`, `CrossTransformerLayer`, and the BST `forward()` logic (PPF, CG, AP). These encode BST's specific inductive biases about player-shuttle interaction.

- **Input contract**: BST's `forward()` expects `(JnB, shuttle, pos, video_len)`. Your model defines its own signature. The dataset class and training loop must agree on this contract.

#### 4. Training script (`bst_train.py`)

The training loop is tightly coupled to BST's input format and hyperparameters.

- **Reusable patterns**: The overall structure (train/validate/test functions, early stopping, cosine LR schedule, TensorBoard logging, `Task` orchestration pattern) can be adapted.

- **What to change**:
  - The `Hyp` namedtuple values (learning rate, batch size, epochs, etc.)
  - The model construction in `get_network_architecture()` (replace BST with your model)
  - The data unpacking in `train_one_epoch()` and `validate()` (the `for (human_pose, pos, shuttle), video_len, labels in loader` destructuring must match your Dataset's return format)
  - The bone-aware augmentation logic (lines 88-95 of `train_one_epoch`) -- this is BST-specific

#### 5. Inference script (`bst_infer.py`)

Same pattern as training: replace the model construction and data unpacking to match your architecture.

### Summary of divergence points

| Stage | BST-specific? | Custom model action |
|-------|--------------|---------------------|
| Pipeline (`pipeline/`) | No | Reuse as-is |
| Pose extraction (`prepare_train_on_shuttleset.py`) | Partially | Replace if your model uses different features (raw video, optical flow, etc.) |
| Dataset class (`shuttleset_dataset.py`) | Yes | Write new Dataset matching your model's input contract |
| Model (`bst.py` + `tempose.py`) | Yes | Replace with your architecture; optionally reuse tempose building blocks |
| Training loop (`bst_train.py`) | Yes | Adapt the loop structure; change data unpacking, model init, augmentation |
| Results (`result_utils.py`) | No | Reuse as-is |
