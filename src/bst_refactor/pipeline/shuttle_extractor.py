"""TrackNetV3 shuttle trajectory extraction and normalization.

Runs TrackNetV3 inference on clip .mp4 files to produce per-clip shuttle
trajectory arrays. Both architectures share this step.

TrackNetV3 is included in the repo (trimmed to inference only) and shares the
BST training venv. Pretrained weights must be downloaded separately — see
TrackNetV3/README.md.

Usage:
    python -m pipeline.shuttle_extractor --tracknet-dir TrackNetV3 [--clips-dir DIR] \
        [--tracknet-python /path/to/bst-venv/bin/python]
"""
import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import (
    CLIPS_OUTPUT_DIR, SHUTTLE_OUTPUT_DIR, RESOLUTION_CSV_PATH,
)

_DEFAULT_TRACKNET_SUBPATH = Path('ckpts') / 'TrackNet_best.pt'
_DEFAULT_INPAINTNET_SUBPATH = Path('ckpts') / 'InpaintNet_best.pt'


def _default_csv_dir(clips_dir: Path) -> Path:
    """Default location for TrackNetV3 CSV outputs: clips_dir/../shuttle_csv."""
    return clips_dir.parent / 'shuttle_csv'


# ---------------------------------------------------------------------------
# Normalization (from prepare_train_on_shuttleset.py:150-159)
# ---------------------------------------------------------------------------
def normalize_shuttlecock(arr: np.ndarray, v_width: float, v_height: float) -> np.ndarray:
    """Normalize shuttle coordinates by video resolution.

    Normalizes x and y to [0, 1]. If a visibility column is present
    (3rd column), it is passed through unchanged.

    :param arr: (t, 2) or (t, 3) array. Columns: x, y, [visibility].
    :param v_width: Video width in pixels.
    :param v_height: Video height in pixels.
    :return: Array with same shape, xy columns normalized.
    """
    result = arr.astype(float)
    result[:, 0] /= v_width
    result[:, 1] /= v_height
    return result


# ---------------------------------------------------------------------------
# TrackNetV3 subprocess invocation
# ---------------------------------------------------------------------------
def extract_shuttle_trajectory(
    clip_path: Path,
    tracknet_dir: Path,
    output_csv_dir: Path,
    model_path: Path | None = None,
    inpaintnet_path: Path | None = None,
    tracknet_python: Path | None = None,
    cur_i: int = 0,
    total: int = 0,
) -> bool:
    """Run TrackNetV3 on a single clip.

    Adapted from detect_shuttlecock_by_TrackNetV3_with_attention() in
    prepare_train_on_shuttleset.py, with parameterised paths.

    :param clip_path: Path to the .mp4 clip file.
    :param tracknet_dir: Path to the cloned TrackNetV3 repository.
    :param output_csv_dir: Directory to write the output CSV.
    :param model_path: Path to TrackNet weights. Defaults to tracknet_dir/ckpts/TrackNet_best.pt.
    :param inpaintnet_path: Path to InpaintNet weights, or None to skip inpainting.
    :param tracknet_python: Python executable in BST venv (shared with TrackNetV3).
        Defaults to sys.executable (assumes shared environment).
    :param cur_i: Current clip index (for progress logging).
    :param total: Total number of clips (for progress logging).
    :return: True on success, False on failure.
    """
    # Skip if result already exists (safety net for direct callers;
    # extract_all_shuttles also pre-filters for performance).
    result_path = output_csv_dir / (clip_path.stem + '_ball.csv')
    if result_path.exists():
        return True

    if model_path is None:
        model_path = tracknet_dir / _DEFAULT_TRACKNET_SUBPATH
    # inpaintnet_path=None means "no inpainting" (caller decides).
    # extract_all_shuttles resolves the default and checks existence.

    # Use BST venv's Python if provided (TrackNetV3 shares BST venv)
    python_exe = str(tracknet_python) if tracknet_python else sys.executable

    process_args = [
        python_exe, str(tracknet_dir / 'predict.py'),
        '--video_file', str(clip_path),
        '--tracknet_file', str(model_path),
        '--save_dir', str(output_csv_dir),
    ]
    # InpaintNet fills in occluded shuttle positions. Without it you're
    # running TrackNet alone, not the full TrackNetV3 pipeline.
    if inpaintnet_path and str(inpaintnet_path):
        process_args.extend(['--inpaintnet_file', str(inpaintnet_path)])

    try:
        r = subprocess.run(process_args, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f'  ERROR ({cur_i}/{total}) {clip_path.name}: {r.stderr.strip()[:200]}')
            return False
        if cur_i and total:
            print(f'  ({cur_i}/{total}) {clip_path.name} done')
        return True
    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT ({cur_i}/{total}) {clip_path.name}')
        return False
    except Exception as e:
        print(f'  ERROR ({cur_i}/{total}) {clip_path.name}: {e}')
        return False


def extract_all_shuttles(
    clips_dir: Path = CLIPS_OUTPUT_DIR,
    tracknet_dir: Path = Path('.'),
    output_csv_dir: Path | None = None,
    model_path: Path | None = None,
    inpaintnet_path: Path | None = None,
    tracknet_python: Path | None = None,
    max_workers: int = 2,
) -> None:
    """Run TrackNetV3 on all clips in parallel.

    Note: max_workers should be modest since TrackNetV3 is GPU-bound.
    Default is 2 to avoid CUDA OOM on shared nodes.

    :param clips_dir: Root clips directory to scan for .mp4 files.
    :param tracknet_dir: Path to the cloned TrackNetV3 repository.
    :param output_csv_dir: Directory for TrackNetV3 CSV outputs.
        Defaults to clips_dir/../shuttle_csv.
    :param model_path: Path to TrackNet weights. Defaults to tracknet_dir/ckpts/TrackNet_best.pt.
    :param inpaintnet_path: Path to InpaintNet weights. Defaults to tracknet_dir/ckpts/InpaintNet_best.pt.
    :param tracknet_python: Python executable in BST venv (shared with TrackNetV3).
        Defaults to sys.executable (assumes shared environment).
    :param max_workers: Number of parallel worker processes (default 2).
    """
    # Preflight: verify TrackNetV3 is set up correctly
    if not tracknet_dir.is_dir():
        raise FileNotFoundError(f'TrackNetV3 directory not found: {tracknet_dir}')
    if not (tracknet_dir / 'predict.py').exists():
        raise FileNotFoundError(f'predict.py not found in: {tracknet_dir}')

    resolved_model = model_path or (tracknet_dir / _DEFAULT_TRACKNET_SUBPATH)
    if not resolved_model.exists():
        raise FileNotFoundError(f'TrackNet weights not found: {resolved_model}')

    resolved_inpaint = inpaintnet_path or (tracknet_dir / _DEFAULT_INPAINTNET_SUBPATH)
    if not resolved_inpaint.exists():
        print(f'  WARNING: InpaintNet weights not found: {resolved_inpaint}')
        print(f'  Running TrackNet only (no inpainting of occluded frames)')
        resolved_inpaint = None

    if output_csv_dir is None:
        output_csv_dir = _default_csv_dir(clips_dir)
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    all_clips = sorted(clips_dir.rglob('*.mp4'))
    # Filter to clips that don't already have results
    pending = [c for c in all_clips
               if not (output_csv_dir / (c.stem + '_ball.csv')).exists()]

    print(f'TrackNetV3 extraction: {len(pending)} pending of {len(all_clips)} total clips')

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, clip_path in enumerate(pending, 1):
            futures.append(executor.submit(
                extract_shuttle_trajectory,
                clip_path, tracknet_dir, output_csv_dir, model_path,
                resolved_inpaint, tracknet_python, i, len(pending),
            ))
        successes = sum(f.result() for f in futures)

    print(f'Extraction complete: {successes}/{len(pending)} succeeded')


# ---------------------------------------------------------------------------
# CSV -> NPY conversion
# ---------------------------------------------------------------------------
def shuttle_csvs_to_npy(
    clips_dir: Path = CLIPS_OUTPUT_DIR,
    csv_dir: Path | None = None,
    npy_output_dir: Path = SHUTTLE_OUTPUT_DIR,
    resolution_csv_path: Path = RESOLUTION_CSV_PATH,
) -> None:
    """Convert TrackNetV3 CSV outputs to normalized .npy files.

    Mirrors the clip directory structure so each clip has a corresponding
    shuttle .npy file:
      clips/train/Top_smash/1_1_3_2.mp4  ->  shuttle_npy/train/Top_smash/1_1_3_2.npy

    :param clips_dir: Root clips directory (used to discover all clips).
    :param csv_dir: Directory containing TrackNetV3 CSV outputs.
        Defaults to clips_dir/../shuttle_csv.
    :param npy_output_dir: Output directory for normalized .npy files.
    :param resolution_csv_path: Path to video resolution CSV (for normalization).
    """
    if csv_dir is None:
        csv_dir = _default_csv_dir(clips_dir)

    res_df = pd.read_csv(resolution_csv_path).set_index('id')
    converted = 0
    missing = 0

    for clip_path in sorted(clips_dir.rglob('*.mp4')):
        # Determine output path (mirror directory structure)
        rel = clip_path.relative_to(clips_dir)
        npy_path = npy_output_dir / rel.with_suffix('.npy')

        if npy_path.exists():
            continue

        # Find corresponding TrackNetV3 CSV
        csv_path = csv_dir / (clip_path.stem + '_ball.csv')
        if not csv_path.exists():
            missing += 1
            continue

        # Get video resolution for normalization
        vid_id = int(clip_path.stem.split('_')[0])
        if vid_id not in res_df.index:
            print(f'  WARNING: No resolution data for video {vid_id}')
            continue

        v_width = res_df.loc[vid_id, 'width']
        v_height = res_df.loc[vid_id, 'height']

        # Read TrackNetV3 CSV and normalize
        df = pd.read_csv(str(csv_path))
        expected_cols = {'Frame', 'X', 'Y', 'Visibility'}
        if not expected_cols.issubset(df.columns):
            print(f'  WARNING: Unexpected CSV format in {csv_path.name}, '
                  f'expected columns {expected_cols}, got {set(df.columns)}')
            continue

        df = df.drop_duplicates('Frame').set_index('Frame')
        # Keep Visibility column -- save as (t, 3): [x, y, visibility].
        # Consumers that only need xy can slice [:, :2].
        shuttle_camera = df[['X', 'Y', 'Visibility']].to_numpy().astype(float)
        shuttle_norm = normalize_shuttlecock(shuttle_camera, v_width, v_height)

        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), shuttle_norm)
        converted += 1

    print(f'Shuttle NPY conversion: {converted} files written, {missing} missing CSVs')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Extract shuttle trajectories from ShuttleSet clips using TrackNetV3.',
    )
    parser.add_argument('--tracknet-dir', type=Path, required=True,
                        help='Path to cloned TrackNetV3 repository')
    parser.add_argument('--clips-dir', type=Path, default=CLIPS_OUTPUT_DIR,
                        help='Directory containing generated clips')
    parser.add_argument('--csv-dir', type=Path, default=None,
                        help='Directory for TrackNetV3 CSV outputs (default: clips_dir/../shuttle_csv)')
    parser.add_argument('--npy-dir', type=Path, default=SHUTTLE_OUTPUT_DIR,
                        help='Output directory for normalized .npy files')
    parser.add_argument('--resolution-csv', type=Path, default=RESOLUTION_CSV_PATH,
                        help='Path to video resolution CSV')
    parser.add_argument('--model-path', type=Path, default=None,
                        help='Path to TrackNet weights (default: tracknet-dir/ckpts/TrackNet_best.pt)')
    parser.add_argument('--inpaintnet-path', type=Path, default=None,
                        help='Path to InpaintNet weights (default: tracknet-dir/ckpts/InpaintNet_best.pt)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Parallel workers for TrackNetV3 (default 2, GPU-bound)')
    parser.add_argument('--tracknet-python', type=Path, default=None,
                        help='Python executable in BST venv (shared with TrackNetV3)')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip TrackNetV3 extraction, only convert existing CSVs to NPY')
    args = parser.parse_args()

    if not args.skip_extraction:
        print('=== Extracting shuttle trajectories ===')
        extract_all_shuttles(
            clips_dir=args.clips_dir,
            tracknet_dir=args.tracknet_dir,
            output_csv_dir=args.csv_dir,
            model_path=args.model_path,
            inpaintnet_path=args.inpaintnet_path,
            tracknet_python=args.tracknet_python,
            max_workers=args.workers,
        )

    print('\n=== Converting shuttle CSVs to NPY ===')
    shuttle_csvs_to_npy(
        clips_dir=args.clips_dir,
        csv_dir=args.csv_dir,
        npy_output_dir=args.npy_dir,
        resolution_csv_path=args.resolution_csv,
    )


if __name__ == '__main__':
    main()
