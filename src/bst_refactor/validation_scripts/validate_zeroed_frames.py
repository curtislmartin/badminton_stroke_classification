"""Analyze the distribution of zeroed frames across the pose/shuttle dataset.

Frames where MMPose failed to detect exactly 2 players on court have their
joints, court positions, AND shuttle coordinates zeroed. The BST transformer
does NOT mask these frames in attention — they participate as zero vectors.
This script quantifies that noise so you can decide whether to exclude
high-failure clips before training.

Loads *_failed.npy files from the dataset_npy/ tree and reports:
- Overall, per-split, and per-stroke-type failure rates
- Tiered clip counts at multiple thresholds
- Histogram and temporal-pattern figures

Outputs (text + PNGs) are saved to a sibling folder:
    validation_scripts/zeroed_frames_analysis_outputs/

Usage:
    python validate_zeroed_frames.py \
        --data-root /path/to/ShuttleSet_data_merged_25 \
        --taxonomy merged_25 \
        --threshold 0.5
"""
import argparse
import io
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend for HPC (no X11 display)
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

class ClipRecord(NamedTuple):
    clip_name: str        # e.g. "35_1_10_17"
    rel_path: str         # e.g. "train/Top_smash/35_1_10_17"
    split: str            # "train", "val", or "test"
    stroke_type: str      # player prefix stripped, e.g. "smash"
    player: str           # "Top", "Bottom", or "" for unprefixed (e.g. unknown)
    total_frames: int
    failed_frames: int
    fail_rate: float
    temporal_bins: np.ndarray  # shape (n_bins,), per-bin fail rates


# ---------------------------------------------------------------------------
# Tee helper — writes to both stdout and a StringIO buffer
# ---------------------------------------------------------------------------

class _Tee:
    """Minimal stdout tee that also captures output for saving to .txt."""

    def __init__(self):
        self._buf = io.StringIO()
        self._stdout = sys.stdout

    def write(self, text: str):
        self._stdout.write(text)
        self._buf.write(text)

    def flush(self):
        self._stdout.flush()

    def get_text(self) -> str:
        return self._buf.getvalue()


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

SPLITS = ("train", "val", "test")
N_TEMPORAL_BINS = 10


def scan_clips(dataset_npy_dir: Path) -> list[ClipRecord]:
    """Walk the dataset_npy/ tree and load every *_failed.npy file.

    Computes per-clip failure stats and temporal bins in a single pass.

    :param dataset_npy_dir: Path to the dataset_npy/ directory.
    :return: List of ClipRecord namedtuples, sorted by (split, stroke, clip).
    """
    records: list[ClipRecord] = []

    for split in SPLITS:
        split_dir = dataset_npy_dir / split
        if not split_dir.is_dir():
            print(f"  WARNING: split directory not found: {split_dir}")
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            # Parse player prefix.  Folders are like "Top_smash", "Bottom_lob",
            # or unprefixed like "unknown".
            folder_name = class_dir.name
            if folder_name.startswith("Top_"):
                player, stroke_type = "Top", folder_name[4:]
            elif folder_name.startswith("Bottom_"):
                player, stroke_type = "Bottom", folder_name[7:]
            else:
                player, stroke_type = "", folder_name

            for fpath in sorted(class_dir.glob("*_failed.npy")):
                clip_name = fpath.name.removesuffix("_failed.npy")
                arr = np.load(fpath)

                total = len(arr)
                if total == 0:
                    print(f"  WARNING: 0-frame clip: {fpath}")
                    records.append(ClipRecord(
                        clip_name=clip_name,
                        rel_path=f"{split}/{folder_name}/{clip_name}",
                        split=split,
                        stroke_type=stroke_type,
                        player=player,
                        total_frames=0,
                        failed_frames=0,
                        fail_rate=0.0,
                        temporal_bins=np.zeros(N_TEMPORAL_BINS),
                    ))
                    continue

                failed = int(np.sum(arr))
                fail_rate = failed / total

                # Temporal bins: split the clip into N equal segments and
                # compute the mean failure rate in each.
                bins = np.array_split(arr, N_TEMPORAL_BINS)
                temporal = np.array(
                    [float(np.mean(b)) if len(b) > 0 else np.nan for b in bins]
                )

                records.append(ClipRecord(
                    clip_name=clip_name,
                    rel_path=f"{split}/{folder_name}/{clip_name}",
                    split=split,
                    stroke_type=stroke_type,
                    player=player,
                    total_frames=total,
                    failed_frames=failed,
                    fail_rate=fail_rate,
                    temporal_bins=temporal,
                ))

    records.sort(key=lambda r: (r.split, r.stroke_type, r.clip_name))
    return records


# ---------------------------------------------------------------------------
# Text reporters
# ---------------------------------------------------------------------------

def print_overall_stats(
    records: list[ClipRecord], taxonomy: str, data_root: Path,
) -> None:
    total_frames = sum(r.total_frames for r in records)
    total_failed = sum(r.failed_frames for r in records)
    rate = total_failed / total_frames * 100 if total_frames else 0

    print("=" * 65)
    print("  Zeroed-Frame Analysis (Pose + Shuttle)")
    print("=" * 65)
    print(f"  Taxonomy:   {taxonomy}")
    print(f"  Data root:  {data_root}")
    print()
    print(f"  Overall: {total_failed:,} / {total_frames:,} frames zeroed ({rate:.2f}%)")
    print(f"           across {len(records):,} clips")
    print()
    print("  Note: joints, court positions, AND shuttle coords are all zeroed")
    print("  on these frames. The transformer does NOT mask them in attention.")
    print()


def print_per_split_stats(records: list[ClipRecord]) -> None:
    by_split: dict[str, list[ClipRecord]] = defaultdict(list)
    for r in records:
        by_split[r.split].append(r)

    print("--- Per-split breakdown ---")
    print(f"  {'Split':<8} {'Clips':>7}   {'Total Frames':>14}   "
          f"{'Failed Frames':>14}   {'Fail Rate':>10}")

    for split in SPLITS:
        recs = by_split.get(split, [])
        if not recs:
            continue
        total = sum(r.total_frames for r in recs)
        failed = sum(r.failed_frames for r in recs)
        rate = failed / total * 100 if total else 0
        print(f"  {split:<8} {len(recs):>7,}   {total:>14,}   "
              f"{failed:>14,}   {rate:>9.2f}%")
    print()


def print_per_stroke_stats(records: list[ClipRecord]) -> None:
    by_stroke: dict[str, list[ClipRecord]] = defaultdict(list)
    for r in records:
        by_stroke[r.stroke_type].append(r)

    # Sort by fail rate descending.
    stroke_stats = []
    for stroke, recs in by_stroke.items():
        total = sum(r.total_frames for r in recs)
        failed = sum(r.failed_frames for r in recs)
        rate = failed / total * 100 if total else 0
        stroke_stats.append((stroke, len(recs), failed, total, rate))
    stroke_stats.sort(key=lambda x: x[4], reverse=True)

    print("--- Per-stroke-type fail rates ---")
    print(f"  {'Stroke Type':<24} {'Clips':>6}   {'Failed / Total':>20}   "
          f"{'Fail Rate':>10}")

    for stroke, n_clips, failed, total, rate in stroke_stats:
        ratio = f"{failed:,} / {total:,}"
        print(f"  {stroke:<24} {n_clips:>6}   {ratio:>20}   {rate:>9.2f}%")
    print()


def print_tiered_clip_counts(
    records: list[ClipRecord], threshold: float,
) -> None:
    tiers = [
        ("100% failed (completely blank)", lambda r: r.fail_rate == 1.0),
        ("> 90% failed",                   lambda r: r.fail_rate > 0.9),
        ("> 75% failed",                   lambda r: r.fail_rate > 0.75),
        ("> 50% failed",                   lambda r: r.fail_rate > 0.5),
    ]

    print("--- Clip failure tiers ---")
    for label, pred in tiers:
        count = sum(1 for r in records if pred(r))
        print(f"  {label + ':':<38} {count:>6} clips")
    print()

    # 100% failed — always list all.
    blank = [r for r in records if r.fail_rate == 1.0]
    if blank:
        print(f"--- Completely blank clips (100% failed): {len(blank)} ---")
        for r in sorted(blank, key=lambda r: r.rel_path):
            print(f"  {r.rel_path}   ({r.total_frames} frames)")
    else:
        print("--- No completely blank clips ---")
    print()

    # User threshold — up to 50, sorted by fail rate desc.
    flagged = sorted(
        [r for r in records if r.fail_rate > threshold],
        key=lambda r: -r.fail_rate,
    )
    print(f"--- Clips above {threshold:.0%} threshold: {len(flagged)} ---")
    if flagged:
        show = flagged[:50]
        for r in show:
            print(f"  {r.rel_path:<55} "
                  f"{r.failed_frames}/{r.total_frames} ({r.fail_rate:.1%})")
        if len(flagged) > 50:
            print(f"  ... and {len(flagged) - 50} more (truncated)")
    print()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_fail_rate_histogram(
    records: list[ClipRecord], output_path: Path,
) -> None:
    """Histogram of per-clip fail rates."""
    rates = np.array([r.fail_rate for r in records])
    mean_rate = float(np.mean(rates))
    median_rate = float(np.median(rates))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rates, bins=20, range=(0.0, 1.0), edgecolor="black", alpha=0.75)
    ax.axvline(mean_rate, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_rate:.3f}")
    ax.set_xlabel("Clip Fail Rate")
    ax.set_ylabel("Number of Clips")
    ax.set_title("Per-clip Zeroed-Frame Rate Distribution")

    # Stats annotation.
    stats_text = (
        f"Clips: {len(records):,}\n"
        f"Mean:   {mean_rate:.4f}\n"
        f"Median: {median_rate:.4f}"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="right",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))

    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_temporal_pattern(
    records: list[ClipRecord], output_path: Path,
) -> None:
    """Line chart: mean fail rate by normalized clip position."""
    by_split: dict[str, list[np.ndarray]] = defaultdict(list)
    all_bins = []

    for r in records:
        all_bins.append(r.temporal_bins)
        by_split[r.split].append(r.temporal_bins)

    # Stack and average, ignoring NaN from single-frame edge cases.
    overall = np.nanmean(np.stack(all_bins), axis=0)
    overall_mean = float(np.nanmean(overall))

    x = np.linspace(0.05, 0.95, N_TEMPORAL_BINS)  # bin centres
    x_labels = [f"{i / N_TEMPORAL_BINS:.0%}-{(i + 1) / N_TEMPORAL_BINS:.0%}"
                for i in range(N_TEMPORAL_BINS)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, overall, color="black", linewidth=2.5, label="All", zorder=3)

    split_colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    for split in SPLITS:
        if split not in by_split:
            continue
        avg = np.nanmean(np.stack(by_split[split]), axis=0)
        ax.plot(x, avg, color=split_colors[split], linewidth=1.2, label=split)

    ax.axhline(overall_mean, color="grey", linestyle="--", linewidth=1,
               label=f"Overall mean: {overall_mean:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Normalised Clip Position")
    ax.set_ylabel("Mean Fail Rate")
    ax.set_title("Temporal Pattern of Zeroed Frames")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze the distribution of zeroed frames across "
                    "the pose/shuttle dataset.",
    )
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Path to ShuttleSet_data_{taxonomy} directory",
    )
    parser.add_argument(
        "--taxonomy", type=str, default="merged_25",
        help="Taxonomy name (used in filenames and display headers)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Fail-rate threshold for the flagged-clips list (default: 0.5)",
    )
    args = parser.parse_args()

    # Auto-discover the per-clip npy directory.  The name varies by windowing
    # strategy (e.g. "dataset_npy_between_2_hits_with_max_limits" for seq_len=100,
    # plain "dataset_npy" for seq_len=30).  We glob for directories matching
    # "dataset*npy*" that contain at least one split subfolder, and exclude the
    # collated directories (those contain stacked arrays, not per-clip files).
    candidates = [
        d for d in sorted(args.data_root.iterdir())
        if d.is_dir()
        and "npy" in d.name
        and "collated" not in d.name
        and any((d / s).is_dir() for s in SPLITS)
    ]
    if not candidates:
        print(f"ERROR: no dataset_npy* directory with split folders found in "
              f"{args.data_root}")
        print("Contents:", [d.name for d in args.data_root.iterdir() if d.is_dir()])
        sys.exit(1)
    if len(candidates) > 1:
        print(f"WARNING: multiple npy dirs found, using first: {candidates[0].name}")
        print(f"  All candidates: {[c.name for c in candidates]}")
    dataset_npy_dir = candidates[0]

    # Sydney timestamp for output filenames.
    syd_now = datetime.now(ZoneInfo("Australia/Sydney"))
    ts = syd_now.strftime("%Y%m%d_%H%M")
    tax_short = args.taxonomy.replace("_", "")  # "merged_25" -> "merged25"

    # Output dir is always a sibling folder to this script.
    output_dir = Path(__file__).resolve().parent / "zeroed_frames_analysis_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout so we can save a .txt copy.
    tee = _Tee()
    sys.stdout = tee

    try:
        # --- Scan ---
        print(f"Scanning {dataset_npy_dir} ...\n")
        records = scan_clips(dataset_npy_dir)

        if not records:
            print("ERROR: No *_failed.npy files found.")
            print(f"Checked: {dataset_npy_dir}")
            sys.exit(1)

        # --- Text reports ---
        print_overall_stats(records, args.taxonomy, args.data_root)
        print_per_split_stats(records)
        print_per_stroke_stats(records)
        print_tiered_clip_counts(records, args.threshold)

        # --- Figures ---
        hist_path = output_dir / f"fail_rate_histogram_{tax_short}_{ts}.png"
        temp_path = output_dir / f"temporal_pattern_{tax_short}_{ts}.png"

        plot_fail_rate_histogram(records, hist_path)
        print(f"Saved: {hist_path}")

        plot_temporal_pattern(records, temp_path)
        print(f"Saved: {temp_path}")

        # --- Save text copy ---
        txt_path = output_dir / f"analysis_{tax_short}_{ts}.txt"

    finally:
        # Restore real stdout before writing the .txt (so we don't recurse).
        sys.stdout = tee._stdout

    with open(txt_path, "w") as f:
        f.write(tee.get_text())
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
