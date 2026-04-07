# pipeline/

Shared data pipeline for the ShuttleSet badminton stroke classification project. Produces labeled video clips and shuttle trajectory files consumed by both team architectures.

## Quick Start

```bash
# Preview what the pipeline will do (no files created)
python -m pipeline.build_dataset --skip-shuttle --dry-run

# Run steps 1-5 (download, resolution CSV, clips, merge, verify)
python -m pipeline.build_dataset --skip-shuttle

# Run everything including shuttle extraction
python -m pipeline.build_dataset --tracknet-dir /path/to/TrackNetV3
```

## Prerequisites

| Dependency | Install | Used by |
|---|---|---|
| yt-dlp | `pip install yt-dlp` | Step 1: video download |
| OpenCV | `pip install opencv-python` | Step 2: resolution scanning |
| MoviePy | `pip install moviepy` | Step 3: clip generation |
| pandas, numpy | `pip install pandas numpy` | All steps |
| TrackNetV3 | Clone from GitHub | Step 6: shuttle extraction (optional) |

## Pipeline Steps

### Step 1: Download Videos

Downloads 40 ShuttleSet match videos from YouTube using yt-dlp. Checks that yt-dlp is installed before spawning workers. Skips videos that already exist on disk.

```bash
python -m pipeline.download_videos --workers 4
```

Output: `ShuttleSet/raw_video/{id} {match_name}.mp4`

### Step 2: Build Resolution CSV

Scans downloaded videos with OpenCV and writes `my_raw_video_resolution.csv`. Replaces the need to manually create this file.

Output: `ShuttleSet/my_raw_video_resolution.csv`

### Step 3: Generate Clips

For each video in each split, extracts individual stroke clips using temporal boundaries from adjacent shots. Filters out excluded videos and individually removed shots automatically.

```bash
python -m pipeline.clip_generator --clip-window between_2_hits_with_max_limits
```

Three clip window options:
- `middle_in_a_sec` -- fixed 1-second window centered on the shot frame
- `between_2_hits` -- from previous shot's frame to next shot's frame
- `between_2_hits_with_max_limits` -- same as above, clamped to 1.5 sec each side (default)

Output: `ShuttleSet/clips/{train,val,test}/{Player}_{stroke_type}/{vid}_{set}_{rally}_{ball_round}.mp4`

### Step 4: Class Merge

Merges 6 rare stroke subtypes into their parent types:

| Subtype | Merged into |
|---|---|
| wrist_smash | smash |
| defensive_return_lob | lob |
| driven_flight | unknown |
| back_court_drive | drive |
| passive_drop | drop |
| defensive_return_drive | drive |

This reduces 19 raw types to 12 merged types (25 classes with Top/Bottom prefixes + unknown).

### Step 5: Verify

Checks that:
- All splits (train/val/test) exist and contain clips
- No clips from excluded videos (IDs from `flaw_shot_records.csv`)
- No individually removed shots present
- Merged subtype folders are empty
- No orphan files with unexpected naming patterns

### Step 6: Shuttle Extraction (Optional)

Runs TrackNetV3 on each clip to extract shuttle trajectories, then normalizes to `(t, 3)` numpy arrays: `[x_norm, y_norm, visibility]`.

```bash
python -m pipeline.shuttle_extractor --tracknet-dir /path/to/TrackNetV3 --workers 2
```

Output: `ShuttleSet/shuttle_npy/{train,val,test}/{Player}_{stroke_type}/{vid}_{set}_{rally}_{ball_round}.npy`

Each `.npy` file has shape `(t, 3)`. To get xy-only coordinates: `shuttle[:, :2]`. To get the visibility mask: `shuttle[:, 2]`.

## CLI Flags

```
python -m pipeline.build_dataset [OPTIONS]

--tracknet-dir PATH    Path to cloned TrackNetV3 repo (required unless --skip-shuttle)
--workers N            Parallel workers (default 2, safe for shared GPU nodes)
--skip-download        Skip YouTube download (videos must already exist)
--skip-shuttle         Skip TrackNetV3 shuttle extraction
--no-merge             Keep all 19 stroke types (skip class merging)
--dry-run              Preview what the pipeline would do without executing
--force                Continue past verification failures
```

## Output Structure

```
ShuttleSet/
  raw_video/                                    # Step 1
    {id} {match_name}.mp4
  my_raw_video_resolution.csv                   # Step 2
  clips/                                        # Steps 3-4
    train/{Top,Bottom}_{stroke_type}/*.mp4
    val/{Top,Bottom}_{stroke_type}/*.mp4
    test/{Top,Bottom}_{stroke_type}/*.mp4
  shuttle_npy/                                  # Step 6
    train/{Top,Bottom}_{stroke_type}/*.npy
    val/{Top,Bottom}_{stroke_type}/*.npy
    test/{Top,Bottom}_{stroke_type}/*.npy
```

Clip filenames: `{video_id}_{set}_{rally}_{ball_round}.mp4`

## Configuration

All configuration lives in `pipeline/config.py`. Key constants:

| Constant | Description |
|---|---|
| `SPLITS` | Train/val/test video ID lists (excluded videos auto-stripped) |
| `EXCLUDED_VIDEOS` | Parsed from `flaw_shot_records.csv` at import time |
| `REMOVED_SHOTS` | Individual bad shots, also from `flaw_shot_records.csv` |
| `MERGE_MAP` | Which subtypes merge into which parents |
| `CLIP_WINDOW` | Default temporal clipping strategy |
| `EN_TO_ZH` / `ZH_TO_EN` | English-Chinese stroke name translation (used at CSV I/O boundary only) |

### Changing Splits

Edit `_SPLITS_RAW` in `config.py`. Use full ranges -- excluded videos are stripped automatically:

```python
_SPLITS_RAW = {
    'train': list(range(1, 35)),        # Videos 1-34 minus exclusions
    'val':   list(range(35, 39)) + [41],
    'test':  [39, 40, 42, 43, 44],
}
```

### Adding Exclusions

Update `ShuttleSet/flaw_shot_records.csv`. The pipeline reads it at import time -- no code changes needed.

## Module Reference

| Module | Purpose |
|---|---|
| `config.py` | All constants, paths, stroke types, splits, flaw records |
| `player_mapping.py` | A/B to Top/Bottom mapping with set 3 court-switch handling |
| `download_videos.py` | yt-dlp downloader + resolution CSV builder |
| `clip_generator.py` | Clip extraction, flaw filtering, class merging |
| `shuttle_extractor.py` | TrackNetV3 wrapper + CSV-to-NPY normalization |
| `court_utils.py` | Optional homography-based court projection utilities |
| `verify.py` | Post-generation sanity checks |
| `build_dataset.py` | One-command orchestrator |

## Running Individual Steps

Each module can be run standalone:

```bash
python -m pipeline.download_videos --workers 4
python -m pipeline.clip_generator --clip-window between_2_hits
python -m pipeline.shuttle_extractor --tracknet-dir /path/to/TrackNetV3
python -m pipeline.verify --clips-dir ShuttleSet/clips
```

## For Downstream Consumers

Both architectures read from the same `clips/` and `shuttle_npy/` directories. The pipeline doesn't care what you do with the output.

```python
# Loading clips (example)
from pathlib import Path
clips = sorted(Path('ShuttleSet/clips/train').rglob('*.mp4'))

# Loading shuttle trajectories
import numpy as np
shuttle = np.load('ShuttleSet/shuttle_npy/train/Top_smash/1_1_3_2.npy')
xy = shuttle[:, :2]           # (t, 2) normalized coordinates
visibility = shuttle[:, 2]    # (t,) detection confidence

# Getting class labels
from pipeline.config import get_stroke_types
labels_25 = get_stroke_types(side='Both', merged=True)   # 25 classes
labels_35 = get_stroke_types(side='Both', merged=False)   # 35 classes
```
