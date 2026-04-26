# BST inference script for ShuttleSet
# Loads a trained checkpoint and predicts stroke types.
# Suitable as a backend for Gradio GUI — call task.infer() to get predictions.
#
# Run from the repo root with both package roots on PYTHONPATH:
#   PYTHONPATH=src/bst_refactor:src/bst_refactor/stroke_classification \
#       python -m main_on_shuttleset.bst_infer
#
# See bst_train.py for detailed PyTorch/TF comparison comments.

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pathlib import Path

from preparing_data.shuttleset_dataset import Dataset_npy_collated
from pipeline.config import TAXONOMIES, DEFAULT_TAXONOMY, Taxonomy
from main_on_shuttleset.bst_common import build_bst_network


@torch.no_grad()  # no gradient tracking needed for inference — saves memory
def infer(
    model: nn.Module,
    loader,
    device
):
    model.eval()  # disable dropout, set batchnorm to eval mode
    pred_ls = []

    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)

        # argmax gives predicted class index; .cpu() moves result back from GPU
        pred = torch.argmax(logits, dim=1).cpu()

        pred_ls.append(pred)

    # torch.cat joins list of batch predictions into one tensor
    return torch.cat(pred_ls)


class Task:
    def __init__(self, n_joints=17) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.n_joints = n_joints

    def prepare_loader(
        self,
        npy_collated_dir: Path,
        pose_style='Jn2B',
        batch_size=128,
    ):
        your_set = Dataset_npy_collated(npy_collated_dir, 'test', pose_style)

        self.infer_loader = DataLoader(
            dataset=your_set,
            batch_size=batch_size
        )
        self.pose_style = pose_style

    def get_network_architecture(
        self,
        model_name='BST_CG_AP',
        seq_len=100,
        in_channels=2,
        taxonomy: Taxonomy = None,
    ):
        if taxonomy is None:
            taxonomy = TAXONOMIES[DEFAULT_TAXONOMY]
        self.taxonomy = taxonomy
        self.net, _n_bones = build_bst_network(
            model_name,
            n_joints=self.n_joints,
            pose_style=self.pose_style,
            in_channels=in_channels,
            n_class=taxonomy.n_classes,
            seq_len=seq_len,
            device=self.device,
        )

    def load_weight(self, weight_path: Path):
        self.net.load_state_dict(torch.load(str(weight_path), map_location=self.device, weights_only=True))

    def infer(self):
        return infer(self.net, self.infer_loader, self.device)


if __name__ == '__main__':
    # Inference example

    taxonomy = TAXONOMIES[DEFAULT_TAXONOMY]

    task = Task(n_joints=17)
    task.prepare_loader(
        # Example path; update the ablation_id suffix to whichever ablation
        # you're inferring on (e.g. une_merge_v1_split_v2_dropunk).
        # Format: npy_[3d_][seq{N}_]{ablation_id} (prefixes only when non-default).
        npy_collated_dir=Path(f'preparing_data/ShuttleSet_data_{taxonomy.name}')
                        / f"npy_{taxonomy.name}_split_v2_dropunk",
        pose_style="JnB_bone",
    )
    task.get_network_architecture(
        model_name='BST_CG_AP',
        seq_len=100,
        in_channels=2,
        taxonomy=taxonomy,
    )
    task.load_weight(Path('weight')
                     /"bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_une_merge_v1_2.pt")

    pred = task.infer()

    classes = taxonomy.class_list()
    pred_cls = [classes[e] for e in pred]
    print(pred_cls)
