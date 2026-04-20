#!/usr/bin/env python3
"""Per-class MMPose fail-rate stats joined on clips_master.csv.

The existing validate_zeroed_frames.py parses class labels from nested folder
names (Top_smash/, Bottom_lob/, ...), which reflects the merged_25 extract and
can't distinguish the classes une_merge_v1 re-exposes (wrist_smash, passive_drop).

This script reads the flat per-clip *_failed.npy files, joins them to
clips_master.csv, applies the requested taxonomy (+ optional drop_unknown),
and prints per-class totals so you can see which une_merge_v1 class is carrying
the most zeroed frames.

Usage on engelbart (from repo root):
  python src/bst_refactor/validation_scripts/fail_rate_per_class.py \\
      --clips-csv notebooks/clips_master.csv \\
      --flat-dir /scratch/comp320a/ShuttleSet_data_merged_25/dataset_npy_between_2_hits_with_max_limits_flat \\
      --split-column split_bst_baseline \\
      --taxonomy une_merge_v1 \\
      --drop-unknown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BST_REFACTOR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BST_REFACTOR_ROOT))

from pipeline.config import TAXONOMIES, Taxonomy  # noqa: E402


def derive_labels(df: pd.DataFrame, taxonomy: Taxonomy) -> pd.Series:
    merge_map = taxonomy.merge_map or {}
    standalone_set = taxonomy.standalone_set
    merged = df['raw_type_en'].map(lambda s: merge_map.get(s, s))
    return merged.where(
        merged.isin(standalone_set),
        df['player_side'] + '_' + merged,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--clips-csv', type=Path, required=True)
    parser.add_argument('--flat-dir', type=Path, required=True,
                        help='Flat per-clip dir holding {clip_stem}_failed.npy')
    parser.add_argument('--split-column', default='split_bst_baseline')
    parser.add_argument('--taxonomy', default='une_merge_v1',
                        choices=list(TAXONOMIES.keys()))
    parser.add_argument('--drop-unknown', action='store_true')
    args = parser.parse_args()

    taxonomy = TAXONOMIES[args.taxonomy]
    df = pd.read_csv(args.clips_csv)
    if args.drop_unknown:
        df = df[df['raw_type_en'] != 'unknown'].copy()
    df['label'] = derive_labels(df, taxonomy)

    # Per-clip fail stats.
    totals, faileds, missing = [], [], 0
    for stem in df['clip_stem']:
        path = args.flat_dir / f'{stem}_failed.npy'
        if not path.exists():
            totals.append(0)
            faileds.append(0)
            missing += 1
            continue
        arr = np.load(path)
        totals.append(len(arr))
        faileds.append(int(arr.sum()))
    df['total_frames'] = totals
    df['failed_frames'] = faileds

    if missing:
        print(f'WARNING: {missing} clips had no *_failed.npy in {args.flat_dir}')

    # Aggregate by (split, label).
    agg = (
        df.groupby([args.split_column, 'label'])
          .agg(clips=('clip_stem', 'size'),
               total_frames=('total_frames', 'sum'),
               failed_frames=('failed_frames', 'sum'))
          .reset_index()
    )
    agg['fail_rate'] = agg['failed_frames'] / agg['total_frames']

    print(f'Taxonomy: {taxonomy.name}   Split: {args.split_column}   '
          f"drop_unknown={args.drop_unknown}")
    for split in ('train', 'val', 'test'):
        sub = agg[agg[args.split_column] == split].sort_values(
            'fail_rate', ascending=False,
        )
        if sub.empty:
            continue
        print()
        print(f'[{split}]  {sub["clips"].sum()} clips, '
              f'{sub["failed_frames"].sum():,} / {sub["total_frames"].sum():,} '
              f'frames failed '
              f'({sub["failed_frames"].sum() / sub["total_frames"].sum():.2%} overall)')
        print(f'  {"class":<30} {"clips":>6}   {"failed/total":>22}  {"rate":>7}')
        for _, r in sub.iterrows():
            ratio = f'{r.failed_frames:,} / {r.total_frames:,}'
            print(f'  {r.label:<30} {r.clips:>6}   {ratio:>22}  '
                  f'{r.fail_rate:>6.2%}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
