"""Single source of truth for ShuttleSet pipeline configuration.

Centralises splits, stroke type definitions (English with Chinese mappings for
CSV I/O), flaw records, merge rules, and default paths. Every other module in
the pipeline imports from here instead of hardcoding these values.
"""
import csv
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Default paths (anchored to project root, not cwd).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SET_INFO_DIR = PROJECT_ROOT / 'ShuttleSet' / 'set'
RAW_VIDEO_DIR = PROJECT_ROOT / 'ShuttleSet' / 'raw_video'
CLIPS_OUTPUT_DIR = PROJECT_ROOT / 'ShuttleSet' / 'clips'
SHUTTLE_OUTPUT_DIR = PROJECT_ROOT / 'ShuttleSet' / 'shuttle_npy'
SHUTTLE_CSV_DIR = PROJECT_ROOT / 'ShuttleSet' / 'shuttle_csv'
FLAW_RECORDS_PATH = PROJECT_ROOT / 'ShuttleSet' / 'flaw_shot_records.csv'
RESOLUTION_CSV_PATH = PROJECT_ROOT / 'ShuttleSet' / 'my_raw_video_resolution.csv'

# ---------------------------------------------------------------------------
# English <-> Chinese stroke name mappings
# Chinese names are used ONLY when reading/writing the upstream ShuttleSet CSV
# annotations. All pipeline code, folder names, and logs use English.
# ---------------------------------------------------------------------------
# 19 stroke types as they appear in the CSV annotations (Chinese)
# mapped to their official English translations.
EN_TO_ZH: dict[str, str] = {
    'net_shot':                '放小球',
    'return_net':              '擋小球',
    'smash':                   '殺球',
    'wrist_smash':             '點扣',
    'lob':                     '挑球',
    'defensive_return_lob':    '防守回挑',
    'clear':                   '長球',
    'drive':                   '平球',
    'driven_flight':           '小平球',
    'back_court_drive':        '後場抽平球',
    'drop':                    '切球',
    'passive_drop':            '過渡切球',
    'push':                    '推球',
    'rush':                    '撲球',
    'defensive_return_drive':  '防守回抽',
    'cross_court_net_shot':    '勾球',
    'short_service':           '發短球',
    'long_service':            '發長球',
    'unknown':                 '未知球種',
}

ZH_TO_EN: dict[str, str] = {v: k for k, v in EN_TO_ZH.items()}

# All 19 raw annotation types (English)
STROKE_TYPES_19 = list(EN_TO_ZH.keys())

# The 19 types as Chinese strings, for matching against CSV annotation data
STROKE_TYPES_19_ZH = list(EN_TO_ZH.values())

# ---------------------------------------------------------------------------
# Class merging: 19 -> 12 stroke types (rare subtypes folded into parents)
# ---------------------------------------------------------------------------
MERGE_MAP: dict[str, str] = {
    'wrist_smash':            'smash',
    'defensive_return_lob':   'lob',
    'driven_flight':          'unknown',
    'back_court_drive':       'drive',
    'passive_drop':           'drop',
    'defensive_return_drive': 'drive',
}

UNE_MERGE_V1_MAP: dict[str, str] = {
    'defensive_return_lob':   'lob',
    'driven_flight':          'drive',
    'back_court_drive':       'drive',
    'defensive_return_drive': 'drive',
}

# The 12 merged stroke types (English), in a stable order.
# These are the types that receive Top_/Bottom_ prefixes in the 25-class system.
STROKE_TYPES_12_MERGED = [
    'net_shot', 'return_net', 'smash', 'lob',
    'clear', 'drive', 'drop', 'push',
    'rush', 'cross_court_net_shot', 'short_service', 'long_service',
]

# The 14 UNE merged stroke types: keeps wrist_smash and passive_drop as
# distinct classes, folds driven_flight into drive instead of unknown.
STROKE_TYPES_14_UNE_MERGE_V1 = [
    'net_shot', 'return_net', 'smash', 'wrist_smash',
    'lob', 'clear', 'drive', 'drop',
    'passive_drop', 'push', 'rush', 'cross_court_net_shot',
    'short_service', 'long_service',
]

# The 17 raw stroke types (English) that receive Top_/Bottom_ prefixes in the
# 35-class system. This is all 19 minus 'unknown' and 'driven_flight' (which
# is always folded into 'unknown' even in the "raw" 35-class system).
STROKE_TYPES_17_RAW = [s for s in STROKE_TYPES_19 if s not in ('unknown', 'driven_flight')]

# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------
PLAYERS = ('Top', 'Bottom')

# ---------------------------------------------------------------------------
# Unprefixed types (clip-generation concern, NOT a taxonomy property)
# These raw ShuttleSet types never get Top_/Bottom_ prefixed folders because
# they lack meaningful player attribution.  Constant across all taxonomies.
# 'driven_flight' is a transient type that always gets merged into 'unknown'
# before training — it only exists as an unprefixed folder during pipeline clip
# generation.
# ---------------------------------------------------------------------------
UNPREFIXED_TYPES: frozenset[str] = frozenset({'unknown', 'driven_flight'})


# ---------------------------------------------------------------------------
# Taxonomy: single source of truth for class grouping schemes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Taxonomy:
    """A stroke-type grouping scheme for training and evaluation.

    :param name: Short identifier, e.g. 'merged_25', 'raw_35'.
    :param merge_map: Maps rare subtype names to parent names, or None if no
        merging is applied.  Only used by the pipeline merge/verify steps.
    :param base_types: Stroke types that receive Top_/Bottom_ player prefixes.
    :param standalone_types: Types that appear unprefixed in the final class
        list (e.g. ``('unknown',)``).
    :param unknown_first: If True, standalone types are placed *before* the
        prefixed types in ``class_list()``; otherwise they come last.
    """

    name: str
    merge_map: dict[str, str] | None
    base_types: tuple[str, ...]
    standalone_types: tuple[str, ...]
    unknown_first: bool

    @property
    def n_classes(self) -> int:
        """Total number of classes when side='Both'."""
        return len(self.base_types) * 2 + len(self.standalone_types)

    @property
    def standalone_set(self) -> frozenset[str]:
        return frozenset(self.standalone_types)

    def class_list(self, side: str = 'Both') -> list[str]:
        """Build the full class label list with Top_/Bottom_ prefixes (English).

        Used for training labels, evaluation display, and folder-to-index
        mapping.  NOT used for clip-generation folder creation (that uses
        the module-level ``UNPREFIXED_TYPES`` constant).

        :param side: ``'Both'``, ``'Top'``, or ``'Bottom'``.
        :return: Ordered list of class label strings.
        """
        base = list(self.base_types)
        standalone = list(self.standalone_types)
        match side:
            case 'Both':
                prefixed = (
                    [f'Top_{s}' for s in base]
                    + [f'Bottom_{s}' for s in base]
                )
            case 'Top':
                prefixed = [f'Top_{s}' for s in base]
            case 'Bottom':
                prefixed = [f'Bottom_{s}' for s in base]
            case _:
                raise ValueError(
                    f"side must be 'Both', 'Top', or 'Bottom', got {side!r}"
                )
        # unknown_first only applies to side='Both' (BST convention).
        # Single-side lists always place standalone types at the end.
        if side == 'Both' and self.unknown_first:
            return standalone + prefixed
        return prefixed + standalone


TAXONOMY_MERGED_25 = Taxonomy(
    name='merged_25',
    merge_map=MERGE_MAP,
    base_types=tuple(STROKE_TYPES_12_MERGED),
    standalone_types=('unknown',),
    unknown_first=True,
)

TAXONOMY_UNE_MERGE_V1 = Taxonomy(
    name='une_merge_v1',
    merge_map=UNE_MERGE_V1_MAP,
    base_types=tuple(STROKE_TYPES_14_UNE_MERGE_V1),
    standalone_types=('unknown',),
    unknown_first=True,
)

# Same merge_map and stroke set as une_merge_v1, but with the Top_/Bottom_
# side prefixes collapsed: every type lands in standalone_types so the
# collator emits an unprefixed label (the side branch in collate_npy is
# skipped whenever ``merged in standalone_set``). Tests whether a 14-class
# space with double the per-class N beats the split 28-class space, on the
# theory that Top_X and Bottom_X are spatial mirrors of the same shot.
TAXONOMY_UNE_MERGE_V1_NOSIDES = Taxonomy(
    name='une_merge_v1_nosides',
    merge_map=UNE_MERGE_V1_MAP,
    base_types=(),
    standalone_types=tuple(STROKE_TYPES_14_UNE_MERGE_V1) + ('unknown',),
    unknown_first=False,
)

TAXONOMY_RAW_35 = Taxonomy(
    name='raw_35',
    merge_map=None,
    base_types=tuple(STROKE_TYPES_17_RAW),
    standalone_types=('unknown',),
    unknown_first=False,
)

DEFAULT_TAXONOMY = 'une_merge_v1'

TAXONOMIES: dict[str, Taxonomy] = {
    'merged_25':            TAXONOMY_MERGED_25,
    'une_merge_v1':         TAXONOMY_UNE_MERGE_V1,
    'une_merge_v1_nosides': TAXONOMY_UNE_MERGE_V1_NOSIDES,
    'raw_35':               TAXONOMY_RAW_35,
}


# ---------------------------------------------------------------------------
# Clip window
# ---------------------------------------------------------------------------
CLIP_WINDOW = 'between_2_hits_with_max_limits'

# ---------------------------------------------------------------------------
# Homography reference resolution
# The homography matrices in homography.csv were computed at this resolution.
# Coordinates must be scaled to match before applying the homography.
# ---------------------------------------------------------------------------
HOMOGRAPHY_RESOLUTION = (1280, 720)


# ---------------------------------------------------------------------------
# Flaw record parsing -- CSV is the single source of truth for exclusions
# ---------------------------------------------------------------------------
def parse_flaw_records(
    csv_path: Path = FLAW_RECORDS_PATH,
) -> tuple[set[int], set[tuple[int, int, int, int]]]:
    """Parse flaw_shot_records.csv to extract excluded videos and removed shots.

    :param csv_path: Path to flaw_shot_records.csv.
    :return: Tuple of (excluded_video_ids, removed_shot_tuples).
    """
    excluded_videos: set[int] = set()
    removed_shots: set[tuple[int, int, int, int]] = set()

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['measure'] != 'removed':
                continue
            match_id = int(row['match'])
            if row['stroke_type'] == 'whole':
                excluded_videos.add(match_id)
            else:
                removed_shots.add((
                    match_id,
                    int(row['set']),
                    int(row['rally']),
                    int(row['ball_round']),
                ))

    return excluded_videos, removed_shots


def _load_flaw_records() -> tuple[set[int], set[tuple[int, int, int, int]]]:
    """Load flaw records lazily. Returns empty sets if CSV is missing.

    This lets modules import stroke types, merge maps, etc. without
    needing flaw_shot_records.csv to be present. The actual pipeline
    steps (clip generation, verification) will fail with clear errors
    if the data they need is empty.
    """
    try:
        return parse_flaw_records()
    except FileNotFoundError:
        import warnings
        warnings.warn(
            f'{FLAW_RECORDS_PATH} not found. '
            f'EXCLUDED_VIDEOS and REMOVED_SHOTS are empty. '
            f'This is fine for inspecting config, but the pipeline '
            f'will produce incorrect results without this file.',
            stacklevel=2,
        )
        return set(), set()


EXCLUDED_VIDEOS, REMOVED_SHOTS = _load_flaw_records()

# ---------------------------------------------------------------------------
# Match-level train/val/test splits
# Define with full intended ranges -- excluded videos are stripped
# automatically below, so you never need to manually skip them.
# ---------------------------------------------------------------------------
_EXPECTED_SPLIT_KEYS = {'train', 'val', 'test'}

_SPLITS_RAW: dict[str, list[int]] = {
    'train': list(range(1, 35)),
    'val':   list(range(35, 39)) + [41],
    'test':  [39, 40, 42, 43, 44],
}

assert set(_SPLITS_RAW.keys()) == _EXPECTED_SPLIT_KEYS, (
    f'SPLITS keys {set(_SPLITS_RAW.keys())} != expected {_EXPECTED_SPLIT_KEYS}'
)

# Strip excluded videos so SPLITS and EXCLUDED_VIDEOS can never desync.
SPLITS: dict[str, list[int]] = {
    name: [v for v in ids if v not in EXCLUDED_VIDEOS]
    for name, ids in _SPLITS_RAW.items()
}


