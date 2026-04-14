# MMPose pose extraction: changes and pipeline context

This directory bridges the pipeline's clip output and BST's expected input format. The pose extraction code in `prepare_train_on_shuttleset.py` runs MMPose on ~33k short clips to produce per-clip skeleton keypoints, court positions, and shuttle trajectories.

---

## Summary of changes vs original BST

The MMPose integration is almost entirely unchanged from the [original BST repo](https://github.com/YuanGao-James/Badminton_Stroke_Timing_Classification). Every function in the pose processing chain is **byte-identical** to the original. Two targeted fixes were added, plus non-functional improvements (docstrings, imports, comments).

### Changes that affect runtime behavior

#### 1. Frame count alignment (bugfix)

**Problem**: MMPose and TrackNetV3 use different video backends that can disagree by 1-2 frames on the same `.mp4` file. The original code crashes or produces mismatched `.npy` files when this happens:

```python
# Original -- crashes with IndexError when len(failed_ls) != len(shuttle_result)
shuttle_result[failed_ls, :] = 0
```

Even when no frames failed, the saved `_joints.npy` (MMPose frame count) and `_shuttle.npy` (TrackNetV3 frame count) could have different temporal dimensions, causing `np.stack()` to crash during collation in `make_seq_len_same()`.

**Fix**: Tail-truncate all arrays to the shorter length before saving:

```python
min_t = min(len(failed_ls), len(shuttle_result))
if min_t < len(failed_ls) or min_t < len(shuttle_result):
    players_positions = players_positions[:min_t]
    joints = joints[:min_t]
    shuttle_result = shuttle_result[:min_t]
    failed_ls = failed_ls[:min_t]
```

**Why tail-truncation, not centering**: Both decoders start at frame 0 and agree on early frames -- the disagreement is about whether the last 1-2 frames are valid (partial frames, B-frame dependencies). Tail-truncation preserves the 1:1 correspondence between `joints[i]` and `shuttle[i]` for all kept frames. Centering would shift one array relative to the other, misaligning every frame by 1.

**Impact**: Loses at most 1-2 frames from the end of a 75-105 frame clip. No effect on clips where frame counts match (the vast majority).

Applied in both `prepare_2d_dataset_npy_from_raw_video()` and `prepare_3d_dataset_npy_from_raw_video()`.

#### 2. GPU memory cleanup between clips

Added `gc.collect()` + `torch.cuda.empty_cache()` after each clip in both the 2D and 3D pose loops.

**Why**: MMPose runs RTMDet + RTMPose on the GPU. Over ~33k clips, unreferenced GPU tensors can fragment CUDA's memory pool. Periodic cleanup prevents gradual accumulation.

**Risk**: Zero. `torch.cuda.empty_cache()` only frees genuinely unreferenced memory -- the MMPoseInferencer's model weights remain on GPU. CuDNN workspace caching is separate and unaffected. Performance cost is ~30-60 seconds total over 33k clips.

**Likelihood of being needed**: Low for this workload. MMPose's allocation pattern is highly uniform (same models, same-ish input, 2-4 people per frame), and PyTorch's caching allocator handles uniform patterns well. This is insurance, not a fix for an observed problem.

### Non-functional changes

- Added module-level docstring (lines 1-11)
- Added docstring to `prepare_2d_dataset_npy_from_raw_video()`
- Import path changes for `pipeline.config` integration
- `sys.path` setup for running as `python -m preparing_data.prepare_train_on_shuttleset`
- Expanded comment on the 3D inferencer per-clip reload bug workaround (lines 320-326)
- Refactored `detect_players_2d()` from nested if-else to early-return with `continue` (functionally identical)

---

## Functions unchanged from original BST

Every helper function in the pose processing chain is byte-identical:

| Function | Purpose |
|----------|---------|
| `get_H()` | Extract homography matrix from DataFrame |
| `get_corner_camera()` | Extract court corner coordinates |
| `scale_pos_by_resolution()` | Scale coordinates to 1280x720 reference |
| `convert_homogeneous()` | Convert to homogeneous coordinates |
| `project()` | Apply homography projection |
| `get_court_info()` | Build court info dict (homography + boundaries) |
| `to_court_coordinate()` | Camera-to-court coordinate transform |
| `normalize_position()` | Normalize by court boundaries to [0, 1] |
| `normalize_joints()` | Normalize keypoints by bbox diagonal or video height |
| `normalize_shuttlecock()` | Normalize by video resolution to [0, 1] |
| `check_pos_in_court()` | Determine which detected people are on court |
| `get_shuttle_result()` | Read TrackNetV3 CSV and normalize |
| `make_seq_len_same()` | Pad/stride clips to uniform seq_len |
| `create_bones()` | Compute bone vectors from joint pairs |
| `interpolate_joints()` | Compute bone midpoints |
| `pad_and_augment_one_npy_video()` | Full per-clip augmentation pipeline |
| `collate_npy()` | Stack per-clip .npy files into batch arrays |

---

## MMPose model and configuration

| Aspect | Value |
|--------|-------|
| Inferencer | `MMPoseInferencer('human')` |
| Resolved models | RTMDet-nano person detector (~30 MB) + RTMPose-L pose estimator (~250 MB) |
| MMPose version | 1.3.2 (pinned in `requirements.txt`) |
| Keypoint format | COCO 17-joint |
| Input | Raw `.mp4` file path (no preprocessing) |
| Frame processing | Per-frame via generator (always, see note below) |
| 2D inferencer lifecycle | Loaded once, reused across all clips |
| 3D inferencer lifecycle | Reloaded per clip (MMPose bug workaround -- see lines 320-326) |

**Frame-level batching is not possible**: MMPoseInferencer accepts a `batch_size` parameter, but MMPose 1.3.2's top-down pipeline **ignores it**. `BaseMMPoseInferencer.preprocess()` overrides MMEngine's `BaseInferencer.preprocess()` (which does implement batching via `_get_chunk_data`) with a method that processes one frame at a time regardless of the `batch_size` value. The source explicitly comments `# only supports inference with batch size 1`. The only batching that occurs is automatic: multiple detected people's crops within a single frame are collated into one RTMPose forward pass. Bypassing the inferencer API for manual frame batching would require ~100+ lines of new code for unclear benefit — RTMDet-nano is fast (~5-10ms/frame) and RTMPose crops are already batched per-frame.

---

## Pose extraction pipeline

### Call chain

```
prepare_train_on_shuttleset.py    main() dispatches 3 steps
    |
    Step 2:  prepare_2d_dataset_npy_from_raw_video()
    |            |
    |            +-- MMPoseInferencer('human')    loaded once
    |            |
    |            +-- detect_players_2d()           per clip
    |            |       |
    |            |       +-- inferencer(video_path)    per-frame generator
    |            |       +-- check_pos_in_court()      filter to 2 on-court players
    |            |       +-- normalize_joints()        normalize by bbox
    |            |
    |            +-- get_shuttle_result()           read TrackNetV3 CSV
    |            +-- tail-truncate to align frames
    |            +-- save _joints.npy, _pos.npy, _shuttle.npy
    |            +-- gc.collect() + torch.cuda.empty_cache()
    |
    Step 3:  collate_npy()
                 |
                 +-- load per-clip .npy files (ThreadPoolExecutor)
                 +-- pad_and_augment_one_npy_video() per clip (ProcessPoolExecutor)
                 +-- np.stack() all clips into batch arrays
                 +-- save J_only.npy, JnB_interp.npy, JnB_bone.npy, Jn2B.npy, ...
```

### Per-clip processing detail

For each clip, `detect_players_2d()`:
1. Iterates MMPose's generator frame by frame
2. Extracts keypoints `(m, 17, 2)` and bboxes `(m, 4)` per frame
3. Requires >= 2 detected people; exactly 2 with feet projecting inside court
4. Orders players top-before-bottom by court y-coordinate
5. Normalizes joints relative to bounding box (or video height)
6. Returns `failed_ls`, `players_positions (t, 2, 2)`, `players_joints (t, 2, 17, 2)`

### Output format per clip

| File | Shape | Contents |
|------|-------|----------|
| `*_joints.npy` | `(F, 2, 17, 2)` or `(F, 2, 17, 3)` | Normalized joint keypoints (2D or 3D) |
| `*_pos.npy` | `(F, 2, 2)` | Court-projected player positions |
| `*_shuttle.npy` | `(F, 2)` | Normalized shuttle coordinates |

### Resume logic

`prepare_2d_dataset_npy_from_raw_video()` checks for an existing `_shuttle.npy` before processing each clip (line 537). Safe to re-run after crashes.

---

## Accuracy guarantees

No accuracy-affecting changes were made. Specifically:

- **MMPose model and call**: identical (`MMPoseInferencer('human')`, `show=False`, `batch_size=1`)
- **Input to MMPose**: raw `.mp4` path, no preprocessing, no resizing, no cropping
- **Keypoint extraction**: same `result['predictions'][0]` access pattern
- **Player filtering**: same >=2 people check, same court projection, same 2-player-on-court requirement
- **Player ordering**: same top-before-bottom by court y-coordinate
- **Joint normalization**: same `normalize_joints()` with identical args
- **Court projection**: same homography chain (`scale_pos_by_resolution` -> `convert_homogeneous` -> `project` -> `normalize_position`)
- **Shuttle normalization**: same `normalize_shuttlecock()`
- **Frame alignment fix**: only trims 1-2 tail frames that one decoder doesn't see; kept frames are unmodified
- **Memory cleanup**: only frees unreferenced GPU memory; does not affect model weights or inference results
