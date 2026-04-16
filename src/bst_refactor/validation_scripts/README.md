# Validation Scripts

Post-extraction analysis tools for the pose/shuttle dataset. Run these **after** MMPose extraction and collation to assess data quality before training.

## Scripts

### `validate_zeroed_frames.py`

Analyses two independent detection failure modes across the dataset:

1. **MMPose failures** (from `*_failed.npy`): MMPose failed to detect exactly 2 players on court — joints, court positions, and shuttle coordinates are all zeroed on these frames. The BST transformer does **not** mask them in attention, so they act as noise.

2. **Shuttle detection failures** (from shuttle NPYs, optional): TrackNetV3 reported visibility=0 (shuttle not detected). Independent of MMPose — the visibility column is dropped during collation, so these failures are invisible to the model as silent (0, 0) shuttle coordinates.

**Minimal usage** (MMPose failure stats only):

```bash
python validate_zeroed_frames.py \
    --data-root /scratch/comp320a/ShuttleSet_data_merged_25
```

**Full usage** (adds flaw cross-reference, hit-frame proximity, and shuttle analysis):

```bash
python validate_zeroed_frames.py \
    --data-root /scratch/comp320a/ShuttleSet_data_merged_25 \
    --set-dir ShuttleSet/set \
    --hit-window 10 \
    --shuttle-npy-dir ShuttleSet/shuttle_npy
```

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data-root` | Yes | - | Path to `ShuttleSet_data_{taxonomy}` directory. The per-clip npy directory is auto-discovered inside it. |
| `--taxonomy` | No | `merged_25` | Taxonomy name. Used in output filenames and display headers. |
| `--threshold` | No | `0.5` | Fail-rate cutoff for the flagged-clips list. |
| `--set-dir` | No | - | Path to `ShuttleSet/set/` directory. Enables flaw annotation cross-reference and hit-frame proximity analysis. |
| `--hit-window` | No | `10` | Frames either side of the hit frame to check for failures. Requires `--set-dir`. |
| `--shuttle-npy-dir` | No | - | Path to `ShuttleSet/shuttle_npy/` directory. Enables shuttle detection failure analysis using TrackNet visibility column. |

**Output** (all saved to `zeroed_frames_analysis_outputs/`):

| File | Contents |
|---|---|
| `analysis_{taxonomy}_{date}_{time}.txt` | Full text report (mirrors terminal output) |
| `fail_rate_histogram_{taxonomy}_{date}_{time}.png` | Per-clip fail rate distribution (log y-axis) |
| `temporal_pattern_{taxonomy}_{date}_{time}.png` | Mean fail rate by normalised clip position |
| `hit_frame_profile_{taxonomy}_{date}_{time}.png` | Fail rate by frame offset from hit, with shuttle overlay if available *(requires `--set-dir`)* |
| `hit_zone_heatmap_{taxonomy}_{date}_{time}.png` | Heatmap of % clips exceeding threshold in hit zone, by class × split *(requires `--set-dir`)* |
| `surviving_clips_{taxonomy}_{date}_{time}.png` | Per-class clip counts remaining after hit-zone quality filter, by split *(requires `--set-dir`)* |
| `hit_oob_clips_{taxonomy}_{date}_{time}.txt` | Clips where hit-frame index exceeded clip length, skipped from hit-frame profile *(requires `--set-dir`; only written when OOB clips exist)* |

Timestamps use Sydney time (AEST/AEDT). The `unknown/` garbage class is excluded from figures, tiered clip counts, flaw cross-reference, shuttle overlap, and hit-frame proximity sections. It is included in overall, per-split, and per-stroke stats (visible as a row).

**Report sections:**

1. **Overall MMPose stats** — total failed frames / total frames across all clips
2. **Per-split breakdown** — train/val/test MMPose fail rates
3. **Per-stroke-type** — MMPose fail rates by stroke class, sorted highest first
4. **Tiered clip counts** — clips at 100%, >90%, >75%, >50% failure, with names for the worst offenders
5. **Flaw cross-reference** *(requires `--set-dir`)* — compares fail rates for shots marked `flaw=1.0` in the original ShuttleSet annotations vs. non-flaw shots
6. **Shuttle detection failures** *(requires `--shuttle-npy-dir`)* — overall and per-split shuttle non-detection rates, plus a 2×2 overlap table showing how MMPose and shuttle failures correlate (both fail, only one, or neither)
7. **Hit-frame proximity** *(requires `--set-dir`)* — compares MMPose fail rates near the hit vs. away, tiered hit-zone clip counts, per-stroke breakdown. When shuttle data is available, also reports shuttle miss rates near the hit, combined data-quality metric (frames where both MMPose and shuttle succeeded), and per-stroke shuttle hit-zone breakdown

### `hit_frame_lookup.py`

Reusable library module (not a CLI script). Maps clip stems to the 0-based frame index of the hit within the clip by re-deriving clip boundaries from the ShuttleSet set CSVs.

```python
from hit_frame_lookup import build_hit_frame_lookup
lookup = build_hit_frame_lookup(Path("ShuttleSet/set"), Path("ShuttleSet/video_metadata.csv"))
# lookup["35_1_10_17"] == 23  means the hit is at frame index 23
```

Uses the same `between_2_hits_with_max_limits` windowing logic as the clip generator, without needing video files. FPS is read from `video_metadata.csv` (the same source of truth as the clip generator) rather than estimated from annotations.

## Dependencies

- Python 3.10+ (uses `X | None` union syntax)
- `numpy`, `matplotlib`, `pandas` — all available in the mmpose venv
- `zoneinfo` — stdlib (Python 3.9+)
- No imports from the project codebase (intentionally decoupled)
