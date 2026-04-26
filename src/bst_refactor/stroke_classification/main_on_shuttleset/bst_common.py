"""Shared scaffolding between bst_train.py and bst_infer.py.

Lifted pre-X3D-S so a third entry point (the X3D-S training script) does
not triplicate the orchestration glue. The BST model graph itself is not
refactored here; this module owns the variant table, the tee'er, the
network builder, and the data-provenance manifest helper only.
"""

import hashlib
from pathlib import Path

import torch
from torch import nn

from preparing_data.shuttleset_dataset import POSE_BONE_MULTIPLIER, get_bone_pairs
from model.bst import BST_0, BST_PPF, BST_CG, BST_AP, BST_CG_AP


# BST variant name -> pre-configured constructor (partials defined in bst.py).
# Both bst_train and bst_infer dispatch through this single mapping.
MODELS = {
    'BST_0':     BST_0,
    'BST':       BST_PPF,
    'BST_CG':    BST_CG,
    'BST_AP':    BST_AP,
    'BST_CG_AP': BST_CG_AP,
}


class Tee:
    """Mirror writes across multiple streams (terminal + file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def build_bst_network(
    model_name: str,
    *,
    n_joints: int,
    pose_style: str,
    in_channels: int,
    n_class: int,
    seq_len: int = 100,
    depth_tem: int = 2,
    depth_inter: int = 1,
    device: str = 'cuda',
) -> tuple[nn.Module, int]:
    """Construct a BST variant with feature-dim wiring shared between train and infer.

    Returns ``(net, n_bones)``. ``n_bones`` is propagated to the train loop so
    the random-translation augmentation knows how many trailing bone channels
    to leave alone; inference can ignore the second return value.

    :param in_channels: 2 for 2D (xy) keypoints, 3 for 3D (xyz).
    """
    n_bones = len(get_bone_pairs())
    extra = POSE_BONE_MULTIPLIER[pose_style]
    in_dim = (n_joints + n_bones * extra) * in_channels
    net = MODELS[model_name](
        in_dim=in_dim,
        n_class=n_class,
        seq_len=seq_len,
        depth_tem=depth_tem,
        depth_inter=depth_inter,
    ).to(device)
    return net, n_bones


def compute_data_provenance(
    clips_csv_path: Path,
    effective_ablation_id: str,
    npy_collated_dir: str,
) -> dict:
    """Manifest ``extra.data_provenance`` for ``track_run``.

    Hashes the clips CSV so the manifest pins the source-of-truth that
    produced this run's collated arrays. Fail fast if missing.
    """
    if not clips_csv_path.exists():
        raise FileNotFoundError(
            f'clips_csv does not exist: {clips_csv_path}\n'
            f'  (Run preparing_data.prepare_train_on_shuttleset to generate '
            f'the collated arrays first.)'
        )
    clips_csv_sha = hashlib.sha256(clips_csv_path.read_bytes()).hexdigest()
    return {
        'data_provenance': {
            'clips_csv_path': str(clips_csv_path),
            'clips_csv_sha256': clips_csv_sha,
            'effective_ablation_id': effective_ablation_id,
            'npy_collated_dir': npy_collated_dir,
        },
    }
