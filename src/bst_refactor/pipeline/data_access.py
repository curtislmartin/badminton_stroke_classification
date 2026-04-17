"""Access ShuttleSet clips, shuttle npy, and mmpose npy filtered by split and taxonomy class.

On-disk layout (all three trees mirror the same structure)
----------------------------------------------------------
clips_dir/
  {split}/                   # train | val | test
    {taxonomy_class}/        # e.g. Top_smash, Bottom_lob, unknown
      {match}_{set}_{rally}_{ball}.mp4

shuttle_npy_dir/             # same layout, .npy instead of .mp4
mmpose_npy_dir/              # same layout, {stem}_joints.npy + {stem}_pos.npy per clip
                             # only present after prepare_train_on_shuttleset.py Step 2

Taxonomy class names
--------------------
The default taxonomy is 'une_merge_v1' (25 classes):
  Top_<stroke> / Bottom_<stroke> for each of the 14 base types, plus 'unknown'.
List all classes actually present on disk:

    python -m pipeline.data_access --list-classes

Python API
----------
    from pipeline.data_access import get_clip_records, DataPaths

    # Defaults: clips and shuttle_npy from pipeline.config paths, no mmpose dir.
    paths = DataPaths()

    # Filter by split and/or class. Both are optional.
    records = get_clip_records(paths, split='val', taxonomy_class='Top_smash')

    for r in records:
        print(r.clip)           # Path to .mp4
        print(r.shuttle_npy)    # Path to shuttle .npy, or None if missing
        print(r.mmpose_joints)  # Path to _joints.npy, or None if not generated yet

    # When mmpose data exists, pass its root directory:
    paths = DataPaths(mmpose_npy_dir=Path('preparing_data/ShuttleSet_data_une_merge_v1/dataset_npy'))
    records = get_clip_records(paths, split='train')

CLI usage
---------
Run from the project root (or any directory with pipeline importable):

    # Count table for all splits and classes
    python -m pipeline.data_access --summary

    # Count table filtered to one split
    python -m pipeline.data_access --split val --summary

    # Count table for one class across all splits
    python -m pipeline.data_access --class Top_smash --summary

    # TSV of all file paths (clip, shuttle, mmpose) — redirect to file for later use
    python -m pipeline.data_access --split train > train_paths.tsv

    # List all class folder names found on disk
    python -m pipeline.data_access --list-classes

    # Override default data paths (e.g. different HPC scratch location)
    python -m pipeline.data_access --clips-dir /scratch/comp320a/ShuttleSet/clips \\
                                   --shuttle-npy-dir /scratch/comp320a/ShuttleSet/shuttle_npy \\
                                   --summary

    # Include mmpose paths once pose estimation has been run
    python -m pipeline.data_access \\
        --mmpose-npy-dir preparing_data/ShuttleSet_data_une_merge_v1/dataset_npy \\
        --summary
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pipeline.config import (
    CLIPS_OUTPUT_DIR,
    SHUTTLE_OUTPUT_DIR,
    TAXONOMIES,
    DEFAULT_TAXONOMY,
    Taxonomy,
)


SPLITS = ('train', 'val', 'test')


@dataclass
class DataPaths:
    """Root directories for each data type.

    All paths default to the config values but can be overridden when data
    lives in a non-standard location (e.g. a different HPC scratch directory).

    :param clips_dir: Root of the clips tree (contains train/val/test subdirs).
    :param shuttle_npy_dir: Root of the shuttle npy tree (mirrors clips_dir).
    :param mmpose_npy_dir: Root of the mmpose per-clip npy tree, or None if
        pose estimation has not been run. Expected layout:
        ``{mmpose_npy_dir}/{split}/{taxonomy_class}/{clip_stem}_joints.npy``
        (alongside ``_pos.npy``).
    """

    clips_dir: Path = CLIPS_OUTPUT_DIR
    shuttle_npy_dir: Path = SHUTTLE_OUTPUT_DIR
    mmpose_npy_dir: Path | None = None


@dataclass
class ClipRecord:
    """Paths for a single clip and its associated data files.

    :param split: Dataset split this clip belongs to ('train', 'val', or 'test').
    :param taxonomy_class: Class folder name, e.g. 'Top_smash' or 'unknown'.
    :param clip: Path to the .mp4 clip file.
    :param shuttle_npy: Path to the shuttle trajectory .npy, or None if missing.
    :param mmpose_joints: Path to the ``_joints.npy`` file, or None if not available.
    :param mmpose_pos: Path to the ``_pos.npy`` file, or None if not available.
    """

    split: str
    taxonomy_class: str
    clip: Path
    shuttle_npy: Path | None
    mmpose_joints: Path | None
    mmpose_pos: Path | None


def _class_dirs(split_dir: Path) -> Iterable[Path]:
    """Yield class subdirectories inside a split directory."""
    if not split_dir.is_dir():
        return
    for d in sorted(split_dir.iterdir()):
        if d.is_dir():
            yield d


def get_clip_records(
    paths: DataPaths,
    split: str | None = None,
    taxonomy_class: str | None = None,
    taxonomy: Taxonomy | None = None,
) -> list[ClipRecord]:
    """Return ClipRecords filtered by split and/or taxonomy class.

    :param paths: Root directories for each data type.
    :param split: One of 'train', 'val', 'test', or None for all splits.
    :param taxonomy_class: Exact class folder name (e.g. 'Top_smash'), or None
        for all classes. Use ``taxonomy.class_list()`` to enumerate valid names.
    :param taxonomy: If provided, validates that ``taxonomy_class`` (when given)
        is a recognised class in this taxonomy.
    :raises ValueError: If split or taxonomy_class are not valid.
    :return: List of ClipRecord, one per .mp4 file matching the filter.
    """
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    if taxonomy is not None and taxonomy_class is not None:
        valid_classes = set(taxonomy.class_list())
        if taxonomy_class not in valid_classes:
            raise ValueError(
                f"{taxonomy_class!r} is not a class in taxonomy {taxonomy.name!r}. "
                f"Valid classes: {sorted(valid_classes)}"
            )

    target_splits = [split] if split else list(SPLITS)
    records: list[ClipRecord] = []

    for sp in target_splits:
        split_clips_dir = paths.clips_dir / sp
        for cls_dir in _class_dirs(split_clips_dir):
            cls_name = cls_dir.name
            if taxonomy_class is not None and cls_name != taxonomy_class:
                continue

            for clip_path in sorted(cls_dir.glob('*.mp4')):
                stem = clip_path.stem

                shuttle = paths.shuttle_npy_dir / sp / cls_name / f'{stem}.npy'
                shuttle = shuttle if shuttle.exists() else None

                joints: Path | None = None
                pos: Path | None = None
                if paths.mmpose_npy_dir is not None:
                    j = paths.mmpose_npy_dir / sp / cls_name / f'{stem}_joints.npy'
                    p = paths.mmpose_npy_dir / sp / cls_name / f'{stem}_pos.npy'
                    joints = j if j.exists() else None
                    pos = p if p.exists() else None

                records.append(ClipRecord(
                    split=sp,
                    taxonomy_class=cls_name,
                    clip=clip_path,
                    shuttle_npy=shuttle,
                    mmpose_joints=joints,
                    mmpose_pos=pos,
                ))

    return records


def summarise(
    paths: DataPaths,
    split: str | None = None,
    taxonomy_class: str | None = None,
) -> None:
    """Print a per-split, per-class count table for the filtered selection.

    :param paths: Root directories for each data type.
    :param split: Split filter, or None for all.
    :param taxonomy_class: Class filter, or None for all.
    """
    records = get_clip_records(paths, split=split, taxonomy_class=taxonomy_class)

    from collections import defaultdict
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {'clips': 0, 'shuttle': 0, 'mmpose': 0})
    )
    for r in records:
        c = counts[r.split][r.taxonomy_class]
        c['clips'] += 1
        if r.shuttle_npy:
            c['shuttle'] += 1
        if r.mmpose_joints:
            c['mmpose'] += 1

    for sp in SPLITS:
        if sp not in counts:
            continue
        print(f'\n{sp}:')
        for cls_name, c in sorted(counts[sp].items()):
            mmpose_str = f"  mmpose={c['mmpose']}" if paths.mmpose_npy_dir else ''
            print(f"  {cls_name:<40}  clips={c['clips']}  shuttle={c['shuttle']}{mmpose_str}")

    total = len(records)
    shuttle_total = sum(1 for r in records if r.shuttle_npy)
    print(f'\nTotal: {total} clips, {shuttle_total} shuttle npys')
    if paths.mmpose_npy_dir:
        mmpose_total = sum(1 for r in records if r.mmpose_joints)
        print(f'       {mmpose_total} mmpose npy sets')


def _menu(prompt: str, options: list[str]) -> str:
    """Print a numbered menu and return the chosen option.

    :param prompt: Question to display above the options.
    :param options: List of option strings.
    :return: The selected option string.
    """
    print(f'\n{prompt}')
    for i, opt in enumerate(options, 1):
        print(f'  {i}) {opt}')
    while True:
        raw = input('> ').strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f'  Enter a number between 1 and {len(options)}.')


def interactive(paths: DataPaths) -> None:
    """Step-through TUI: pick split, class, and output type interactively.

    :param paths: Root directories for each data type.
    """
    # Step 1: split
    split_choice = _menu('Select split:', ['all'] + list(SPLITS))
    split = None if split_choice == 'all' else split_choice

    # Step 2: class — discover from disk so the list reflects what's actually there
    seen: set[str] = set()
    for sp in ([split] if split else list(SPLITS)):
        for d in _class_dirs(paths.clips_dir / sp):
            seen.add(d.name)
    class_options = ['all'] + sorted(seen)
    class_choice = _menu('Select class:', class_options)
    taxonomy_class = None if class_choice == 'all' else class_choice

    # Step 3: output
    output_choice = _menu('Show:', ['summary table', 'file paths'])

    print()
    if output_choice == 'summary table':
        summarise(paths, split=split, taxonomy_class=taxonomy_class)
    else:
        records = get_clip_records(paths, split=split, taxonomy_class=taxonomy_class)
        for r in records:
            shuttle_str = str(r.shuttle_npy) if r.shuttle_npy else 'MISSING'
            mmpose_str = str(r.mmpose_joints) if r.mmpose_joints else 'NO_MMPOSE'
            print(f'{r.split}\t{r.taxonomy_class}\t{r.clip}\t{shuttle_str}\t{mmpose_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--split', choices=list(SPLITS), default=None,
        help='Filter to one split (default: all splits)',
    )
    parser.add_argument(
        '--class', dest='taxonomy_class', default=None,
        help='Filter to one taxonomy class folder, e.g. Top_smash (default: all)',
    )
    parser.add_argument(
        '--taxonomy', choices=list(TAXONOMIES), default=DEFAULT_TAXONOMY,
        help=f'Taxonomy for class validation (default: {DEFAULT_TAXONOMY})',
    )
    parser.add_argument(
        '--clips-dir', type=Path, default=CLIPS_OUTPUT_DIR,
        help=f'Root clips directory (default: {CLIPS_OUTPUT_DIR})',
    )
    parser.add_argument(
        '--shuttle-npy-dir', type=Path, default=SHUTTLE_OUTPUT_DIR,
        help=f'Root shuttle npy directory (default: {SHUTTLE_OUTPUT_DIR})',
    )
    parser.add_argument(
        '--mmpose-npy-dir', type=Path, default=None,
        help='Root mmpose per-clip npy directory (default: not set)',
    )
    parser.add_argument(
        '--summary', action='store_true',
        help='Print per-split/class count table instead of individual paths',
    )
    parser.add_argument(
        '--list-classes', action='store_true',
        help='List all class names found on disk for the given split and exit',
    )
    args = parser.parse_args()

    taxonomy = TAXONOMIES[args.taxonomy]
    paths = DataPaths(
        clips_dir=args.clips_dir,
        shuttle_npy_dir=args.shuttle_npy_dir,
        mmpose_npy_dir=args.mmpose_npy_dir,
    )

    # No flags passed — launch interactive TUI
    no_flags = not any([args.split, args.taxonomy_class, args.summary, args.list_classes])
    if no_flags:
        interactive(paths)
    elif args.list_classes:
        target_splits = [args.split] if args.split else list(SPLITS)
        seen: set[str] = set()
        for sp in target_splits:
            for d in _class_dirs(paths.clips_dir / sp):
                seen.add(d.name)
        for name in sorted(seen):
            print(name)
    elif args.summary:
        summarise(paths, split=args.split, taxonomy_class=args.taxonomy_class)
    else:
        records = get_clip_records(
            paths,
            split=args.split,
            taxonomy_class=args.taxonomy_class,
            taxonomy=taxonomy,
        )
        for r in records:
            shuttle_str = str(r.shuttle_npy) if r.shuttle_npy else 'MISSING'
            mmpose_str = str(r.mmpose_joints) if r.mmpose_joints else 'NO_MMPOSE'
            print(f'{r.split}\t{r.taxonomy_class}\t{r.clip}\t{shuttle_str}\t{mmpose_str}')
