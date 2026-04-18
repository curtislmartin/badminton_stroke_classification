"""Backfill the Aim UI from existing run-tracker manifests.

Walks every experiments/<run_id>/manifest.yaml, matches it to the test
log file recorded in `log_path:`, slices the log into per-serial blocks
(`=== Serial N (...) ===` headers), and mirrors each serial into Aim
with:

  - run_hash = '<run_id>_s<N>'         (stable, so re-runs update in place)
  - name     = '<run_id>_s<N>'
  - hparams  = manifest.config
  - metrics  = manifest.serials[i].metrics
  - description = the serial's test-log block
  - tags     = derived from config + best_serials + legacy flag

Idempotent. Safe to re-run whenever a training batch finished without
Aim installed, or after editing manifests. Existing Aim runs with the
same hash are reopened and overwritten rather than duplicated.

Usage (from the repo root or anywhere with run_tracker importable):
    pip install aim
    cd src/bst_refactor/stroke_classification/main_on_shuttleset
    python ../../aim_backfill.py                        # default experiments/
    python ../../aim_backfill.py path/to/experiments
    aim up                                              # UI at localhost:43800
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from run_tracker import _AIM_AVAILABLE, mirror_to_aim


SERIAL_HEADER_RE = re.compile(r'^=== Serial (\d+) \(', re.MULTILINE)


def _read_manifest(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _split_log_by_serial(log_text: str) -> dict[int, str]:
    """Carve a test_log file into {serial_no: 'full block text'}.

    Blocks start at '=== Serial N (...' headers and run until the next
    header or EOF. Preserves the leading header line so the Aim
    description is self-explanatory.
    """
    if not log_text.strip():
        return {}
    markers = list(SERIAL_HEADER_RE.finditer(log_text))
    blocks: dict[int, str] = {}
    for i, m in enumerate(markers):
        serial_no = int(m.group(1))
        start = m.start()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(log_text)
        blocks[serial_no] = log_text[start:end].rstrip()
    return blocks


def _derive_tags(manifest: dict, serial_no: int) -> list[str]:
    """Auto-tags for navigating the Aim UI.

    Combines: legacy flag, an anneal-regime label from config, and a
    'best' tag if the serial appears in manifest.best_serials. Any
    manifest-level 'tags' list is appended as-is.
    """
    tags: list[str] = []
    if manifest.get('legacy'):
        tags.append('legacy')

    cfg = manifest.get('config') or {}
    if not cfg.get('use_aux_schedule'):
        tags.append('no_aux_anneal')
    else:
        fade = cfg.get('aux_fade_end_epoch') or 0
        n_epochs = cfg.get('n_epochs') or 0
        if fade <= 1:
            tags.append('cg_ap_off_from_start')
        elif n_epochs and fade < n_epochs * 0.3:
            tags.append('anneal_aggressive')
        else:
            tags.append('anneal_gentle')

    if serial_no in (manifest.get('best_serials') or []):
        tags.append('best')

    for t in (manifest.get('tags') or []):
        if t not in tags:
            tags.append(t)

    return tags


def _resolve_log_path(manifest: dict, experiments_dir: Path) -> Path | None:
    """log_path in the manifest is stored relative to experiments_dir.parent
    (the main_on_shuttleset/ folder). Resolve to an absolute path.
    """
    rel = manifest.get('log_path')
    if not rel:
        return None
    candidate = (experiments_dir.parent / rel).resolve()
    return candidate if candidate.exists() else None


def backfill_run(run_dir: Path, experiments_dir: Path) -> int:
    manifest_path = run_dir / 'manifest.yaml'
    if not manifest_path.exists():
        return 0
    manifest = _read_manifest(manifest_path)
    serials = manifest.get('serials') or []
    if not serials:
        return 0

    log_path = _resolve_log_path(manifest, experiments_dir)
    blocks: dict[int, str] = {}
    if log_path is not None:
        blocks = _split_log_by_serial(log_path.read_text())

    run_id = manifest['run_id']
    count = 0
    for s in serials:
        serial_no = s['serial_no']
        metrics = s.get('metrics') or {}
        description = blocks.get(serial_no)
        tags = _derive_tags(manifest, serial_no)
        name = f'{run_id}_s{serial_no}'
        if mirror_to_aim(manifest, serial_no, metrics,
                         description=description, tags=tags, name=name):
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments_dir', nargs='?', default='experiments')
    args = parser.parse_args()

    if not _AIM_AVAILABLE:
        print('aim is not installed. Install with:  pip install aim')
        sys.exit(1)

    experiments_dir = Path(args.experiments_dir).resolve()
    if not experiments_dir.is_dir():
        print(f'Not a directory: {experiments_dir}')
        sys.exit(1)

    total_runs = 0
    total_serials = 0
    for run_dir in sorted(experiments_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        n = backfill_run(run_dir, experiments_dir)
        if n:
            print(f'[{run_dir.name}] mirrored {n} serial(s)')
            total_runs += 1
            total_serials += n
    print(f'Done. {total_runs} run(s), {total_serials} serial(s) mirrored.')


if __name__ == '__main__':
    main()
