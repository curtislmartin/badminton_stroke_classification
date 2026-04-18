"""Experiment tracker for BST/arch training scripts.

Writes a YAML manifest per run under experiments/<run_id>/ with
hyperparameters, git SHA, and per-serial metrics + output paths. Aim
integration is optional: if the aim package is pip-installed, hparams
and metrics are mirrored into .aim/ for UI browsing (aim up). If aim
is not installed, the YAML manifest alone is the record.

Integration cost per train script: 2 function calls.

    from run_tracker import track_run, track_serial

    run_dir, run_id = track_run(config=hyp, run_id=f'run_{timestamp}')
    # existing loop:
    for serial_no in range(1, 6):
        tb_dir = run_dir / 'tb' / f'serial_{serial_no}'
        weights_path = run_dir / 'weights' / f'serial_{serial_no}.pt'
        # ... existing training + testing ...
        track_serial(run_dir, serial_no,
                     weights_path=weights_path, tb_dir=tb_dir,
                     metrics={'macro_f1': 0.834, 'min_f1': 0.619})

config can be a @dataclass, namedtuple, dict, or any object with vars().
metrics is any flat dict of scalar values.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    import aim
    _AIM_AVAILABLE = True
except ImportError:
    _AIM_AVAILABLE = False


DEFAULT_EXPERIMENTS_DIR = Path('experiments')


def track_run(
    config: Any,
    run_id: str | None = None,
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
    project_root: Path | str | None = None,
    extra: dict | None = None,
    log_path: Path | str | None = None,
) -> tuple[Path, str]:
    """Create (or reopen) a run folder and write the initial manifest.yaml.

    :param config: hyperparameters object. Accepts @dataclass, namedtuple,
                   Mapping, or anything accepted by vars(). Stored verbatim
                   in the manifest, so values must be YAML-serializable.
    :param run_id: folder name under experiments_dir. Defaults to
                   'run_YYYYMMDD_HHMMSS'.
    :param experiments_dir: parent folder. Created if missing.
    :param project_root: for the git-SHA lookup. Defaults to cwd.
    :param extra: any additional top-level manifest fields (notes, env info).
    :param log_path: optional path to the run's stdout/test log file. Stored
                     on the manifest so aim_backfill.py can slice per-serial
                     blocks into Aim-run descriptions later.
    :return: (run_dir, run_id). Train script writes weights under
             run_dir/weights/ and TB event files under run_dir/tb/serial_N/.

    Idempotent if called with a run_id whose manifest already exists: the
    existing manifest is kept untouched and the paths are returned. Useful
    for resume flows.
    """
    if run_id is None:
        run_id = f'run_{datetime.now():%Y%m%d_%H%M%S}'

    run_dir = Path(experiments_dir) / run_id
    (run_dir / 'weights').mkdir(parents=True, exist_ok=True)
    (run_dir / 'tb').mkdir(parents=True, exist_ok=True)

    if _manifest_path(run_dir).exists():
        return run_dir, run_id

    proj_root = Path(project_root) if project_root is not None else Path.cwd()
    manifest = {
        'run_id': run_id,
        'started_at': datetime.now().isoformat(timespec='seconds'),
        'git_sha': _git_sha(proj_root),
        'git_dirty': _git_dirty(proj_root),
        'host': os.uname().nodename if hasattr(os, 'uname') else None,
        'config': _config_to_dict(config),
        'serials': [],
    }
    if log_path is not None:
        manifest['log_path'] = _relpath(log_path)
    if extra:
        manifest['extra'] = extra

    _write_manifest(run_dir, manifest)
    return run_dir, run_id


def track_serial(
    run_dir: Path | str,
    serial_no: int,
    weights_path: Path | str,
    tb_dir: Path | str | None = None,
    metrics: dict | None = None,
    extra: dict | None = None,
) -> None:
    """Append (or replace) a serial entry in the run's manifest.yaml.

    Idempotent by serial_no: calling a second time with the same serial_no
    replaces the earlier entry, so re-tests can update metrics in place.
    Also opens a lightweight aim.Run(run_hash='<run_id>_s<n>') with hparams
    + metrics if aim is installed.

    :param run_dir: the Path returned by track_run.
    :param serial_no: 1-indexed serial number within the run.
    :param weights_path: final best-checkpoint path for this serial.
    :param tb_dir: TensorBoard event directory for this serial, if any.
                   Pass log_dir from the SummaryWriter. None if no TB.
    :param metrics: flat dict of scalar values, e.g.
                    {'macro_f1': 0.834, 'min_f1': 0.619, 'accuracy': 0.846}.
    :param extra: any per-serial additional fields (best-epoch, notes, etc).
    """
    run_dir = Path(run_dir)
    manifest = _read_manifest(run_dir)

    entry = {
        'serial_no': serial_no,
        'weights_path': _relpath(weights_path),
        'tb_dir': _relpath(tb_dir) if tb_dir is not None else None,
        'metrics': dict(metrics) if metrics else {},
        'recorded_at': datetime.now().isoformat(timespec='seconds'),
    }
    if extra:
        entry['extra'] = extra

    manifest['serials'] = [
        s for s in manifest.get('serials', []) if s['serial_no'] != serial_no
    ]
    manifest['serials'].append(entry)
    manifest['serials'].sort(key=lambda s: s['serial_no'])

    _write_manifest(run_dir, manifest)

    if _AIM_AVAILABLE and metrics:
        mirror_to_aim(manifest, serial_no, metrics)


def _config_to_dict(config: Any) -> dict:
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, '_asdict'):
        return dict(config._asdict())
    if is_dataclass(config):
        return asdict(config)
    return dict(vars(config))


def _git_sha(project_root: Path) -> str | None:
    return _run_git(project_root, ['rev-parse', 'HEAD'])


def _git_dirty(project_root: Path) -> bool | None:
    out = _run_git(project_root, ['status', '--porcelain'])
    return None if out is None else bool(out)


def _run_git(project_root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ['git', '-C', str(project_root), *args],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _relpath(p: Path | str) -> str:
    """Render a path relative to cwd when possible, else absolute."""
    try:
        return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
    except (ValueError, OSError):
        return str(p)


def _manifest_path(run_dir: Path) -> Path:
    return Path(run_dir) / 'manifest.yaml'


def _read_manifest(run_dir: Path) -> dict:
    with open(_manifest_path(run_dir)) as f:
        return yaml.safe_load(f) or {}


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    with open(_manifest_path(run_dir), 'w') as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)


def mirror_to_aim(
    manifest: dict,
    serial_no: int,
    metrics: dict,
    description: str | None = None,
    tags: list[str] | None = None,
    name: str | None = None,
) -> bool:
    """Mirror one serial into Aim. Stable run_hash = '<run_id>_s<N>'.

    Idempotent by design: reopening a run with the same hash overwrites
    params / name / description rather than duplicating. Safe to call from
    both the live `track_serial` path and the standalone `aim_backfill.py`
    script without creating duplicate UI entries.

    :param description: freeform text shown in the Aim UI for this run.
                        Typically the per-serial test-log block from
                        test_logs/*.log.
    :param tags: list of tag labels (e.g., 'legacy', 'best', 'anneal_gentle').
    :param name: human-readable alias in the UI. Defaults to
                 '<run_id>_s<N>' when None.
    :return: True if the mirror succeeded, False if aim is unavailable or
             the call raised. Errors are logged to stderr and swallowed so
             a broken aim install can't take down a training run.
    """
    if not _AIM_AVAILABLE:
        return False
    try:
        run_id = manifest['run_id']
        aim_run = aim.Run(run_hash=f'{run_id}_s{serial_no}')
        aim_run.name = name if name is not None else f'{run_id}_s{serial_no}'
        aim_run['hparams'] = manifest.get('config', {})
        aim_run['run_id'] = run_id
        aim_run['serial_no'] = serial_no
        if manifest.get('git_sha'):
            aim_run['git_sha'] = manifest['git_sha']
        if manifest.get('notes'):
            aim_run['run_notes'] = manifest['notes']
        if description is not None:
            aim_run.description = description
        if tags:
            existing = set(aim_run.props.tags) if hasattr(aim_run.props, 'tags') else set()
            for t in tags:
                if t not in existing:
                    aim_run.add_tag(t)
        for k, v in (metrics or {}).items():
            try:
                aim_run.track(float(v), name=k)
            except (TypeError, ValueError):
                aim_run[k] = v
        aim_run.close()
        return True
    except Exception as e:
        print(f'[run_tracker] aim mirror failed (ignored): {e}', file=sys.stderr)
        return False
