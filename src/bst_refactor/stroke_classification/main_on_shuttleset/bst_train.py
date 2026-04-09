# Consolidated BST training script for ShuttleSet
# Replaces: bst_main.py, bst_main_summary_writer.py, bst_backbone_main.py
#
# PyTorch training loop overview (differs significantly from TF/Keras):
#   TF:      model.compile(optimizer, loss) -> model.fit(data)  (one line trains everything)
#   PyTorch: you write the loop yourself — iterate batches, compute loss, call backward(), step()
#   This is more verbose but gives full control over every training step.

import torch
from torch import Tensor, nn, optim  # nn = layers/models, optim = optimizers (like tf.keras.optimizers)
import torch.nn.functional as F      # F = stateless functions (one_hot, softmax, etc.)
from torch.utils.data import DataLoader  # like tf.data.Dataset — batches, shuffles, prefetches
from torch.utils.tensorboard import SummaryWriter  # TensorBoard logging (same viewer as TF)
from torcheval.metrics.functional import multiclass_f1_score

from transformers import get_cosine_schedule_with_warmup  # from HuggingFace, not a custom module

import pandas as pd
from pathlib import Path
from copy import deepcopy
from collections import namedtuple
import time
from datetime import timedelta

import sys
import os
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from preparing_data.shuttleset_dataset import prepare_npy_collated_loaders, \
                                              RandomTranslation_batch, Dataset_npy, \
                                              pad_class_labels, get_bone_pairs, \
                                              POSE_BONE_MULTIPLIER
from model.bst import BST_0, BST_PPF, BST_CG, BST_AP, BST_CG_AP
from result_utils import show_f1_results, plot_confusion_matrix
from pipeline.config import TAXONOMIES, DEFAULT_TAXONOMY, Taxonomy


# ==========================================================================
# Hyperparameters — edit these to change experiment configuration.
# namedtuple is just an immutable struct: hyp.lr, hyp.batch_size, etc.
# ==========================================================================
Hyp = namedtuple('Hyp', [
    'n_epochs', 'batch_size', 'lr', 'warm_up_step',
    'taxonomy', 'seq_len', 'early_stop_n_epochs',
    'pose_style', 'use_3d_pose', 'train_partial'
])
hyp = Hyp(
    n_epochs=1600,            # max epochs (will early-stop before this)
    early_stop_n_epochs=300,  # stop if no F1 improvement for this many epochs
    batch_size=128,
    lr=5e-4,                  # initial learning rate (cosine-annealed during training)
    warm_up_step=400,         # LR warmup steps before cosine decay begins
    taxonomy=DEFAULT_TAXONOMY, # key in TAXONOMIES: 'une_merge_v1', 'merged_25', 'raw_35', …
    seq_len=30,               # frames per sample (must match data preprocessing)
    pose_style='JnB_bone',   # 'J_only'=joints, 'JnB_bone'=joints+bones, 'Jn2B'=joints+2xbones
    use_3d_pose=False,        # True for xyz keypoints, False for xy only
    train_partial=0.25        # fraction of training set to use (1.0 = all)
)


# ==========================================================================
# Training and evaluation functions
# ==========================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    random_shift_fn,
    n_bones: int,
    loss_fn,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,  # learning rate scheduler
    device
):
    model.train()  # enable dropout + batchnorm training mode (TF: training=True)
    total_loss = 0.0

    for (human_pose, pos, shuttle), video_len, labels in loader:
        # .to(device) = move tensors to GPU/CPU. TF does this automatically;
        # PyTorch requires explicit placement for every tensor.
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)

        # Apply random translation augmentation to joints only (not bones,
        # because bone vectors are relative and translation-invariant)
        if n_bones == 0:
            human_pose = random_shift_fn(human_pose)
        else:
            joints = human_pose[:, :, :, :-n_bones, :].contiguous()
            bones = human_pose[:, :, :, -n_bones:, :]

            joints = random_shift_fn(joints)
            human_pose = torch.cat([joints, bones], dim=-2)

        # Flatten last two dims (joints/bones, xy) into one feature dim for the model
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)
        loss: Tensor = loss_fn(logits, labels)

        # PyTorch manual gradient step (TF does this inside model.fit()):
        optimizer.zero_grad()  # clear gradients from previous batch
        loss.backward()        # backpropagation: compute gradients (like tape.gradient())
        optimizer.step()       # apply gradients to weights (like optimizer.apply_gradients())
        scheduler.step()       # update learning rate according to cosine schedule

        total_loss += loss.item()  # .item() extracts Python float from single-element tensor

    train_loss = total_loss / len(loader)
    return train_loss


@torch.no_grad()  # disables gradient computation — saves memory during eval
def validate(
    model: nn.Module,
    loss_fn,
    loader,
    device,
    n_classes: int,
):
    model.eval()  # disable dropout + set batchnorm to eval mode (TF: training=False)
    total_loss = 0.0
    # Accumulate confusion matrix components across batches for per-class F1
    cum_tp = torch.zeros(n_classes)
    cum_tn = torch.zeros(n_classes)
    cum_fp = torch.zeros(n_classes)
    cum_fn = torch.zeros(n_classes)

    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)
        loss: Tensor = loss_fn(logits, labels)
        total_loss += loss.item()

        # Manual per-class TP/FP/FN/TN computation via one-hot encoding
        pred = F.one_hot(torch.argmax(logits, dim=1), n_classes).bool()
        labels_onehot = F.one_hot(labels, n_classes).bool()

        tp = torch.sum(pred & labels_onehot, dim=0)
        tn = torch.sum(~pred & ~labels_onehot, dim=0)

        fp = torch.sum(pred & ~labels_onehot, dim=0)
        fn = torch.sum(~pred & labels_onehot, dim=0)

        # .cpu() moves results back from GPU for accumulation
        cum_tp += tp.cpu()
        cum_tn += tn.cpu()
        cum_fp += fp.cpu()
        cum_fn += fn.cpu()

    val_loss = total_loss / len(loader)

    # Per-class F1, then macro average (mean across classes)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / (cum_tp + cum_fn)

    f1_score = 2 * precision * recall / (precision + recall)
    f1_score[f1_score.isnan()] = 0  # classes with no predictions get NaN -> 0

    f1_score_avg = f1_score.mean()  # macro F1: unweighted mean across all classes
    f1_score_min = f1_score.min()   # worst-performing class
    return val_loss, f1_score_avg, f1_score_min


@torch.no_grad()
def test(
    model: nn.Module,
    loader,
    device
):
    model.eval()
    pred_ls = []
    labels_ls = []
    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)

        pred = torch.argmax(logits, dim=1).cpu()

        pred_ls.append(pred)
        labels_ls.append(labels)

    return torch.cat(pred_ls), torch.cat(labels_ls)


@torch.no_grad()
def test_topk(
    model: nn.Module,
    loader,
    device,
    k=2
):
    model.eval()
    pred_ls = []
    labels_ls = []
    for (human_pose, pos, shuttle), video_len, labels in loader:
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        logits = model(human_pose, shuttle, pos, video_len)

        _, pred = torch.topk(logits, k=k, dim=1)

        pred_ls.append(pred.cpu())
        labels_ls.append(labels)

    return torch.cat(pred_ls), torch.cat(labels_ls)


# ==========================================================================
# Training loop with TensorBoard logging and early stopping
# ==========================================================================

def train_network(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    save_path: Path,
    n_bones,
    n_classes: int,
):
    writer = SummaryWriter()  # logs to ./runs/ — view with: tensorboard --logdir=runs
    random_shift_fn = RandomTranslation_batch()  # data augmentation: small xy shifts

    # label_smoothing=0.1: softens targets from [0,1] to [0.004, 0.904] to reduce overconfidence
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    # AdamW = Adam with decoupled weight decay (standard for transformers)
    # model.parameters() returns all learnable weights (TF equivalent: model.trainable_variables)
    optimizer = optim.AdamW(model.parameters(), lr=hyp.lr)
    # Cosine schedule: LR ramps up during warmup, then decays following a cosine curve
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=hyp.warm_up_step,
        num_training_steps=(hyp.n_epochs * len(train_loader)),  # total batches across all epochs
        num_cycles=0.25  # fraction of cosine cycle (0.25 = quarter-cosine decay)
    )

    best_value = 0.0
    early_stop_count = 0

    for epoch in range(1, hyp.n_epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            random_shift_fn=random_shift_fn,
            n_bones=n_bones,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        val_loss, f1_score_avg, f1_score_min = validate(
            model=model,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device,
            n_classes=n_classes,
        )
        t1 = time.time()
        print(f'Epoch({epoch}/{hyp.n_epochs}): train_loss={train_loss:.3f}, '
              f'val_loss={val_loss:.3f}, macro_f1={f1_score_avg:.3f}, min_f1={f1_score_min:.3f} '
              f'- {t1 - t0:.2f} s')

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)

        # Early stopping: if macro F1 hasn't improved for early_stop_n_epochs, stop training
        early_stop_count += 1
        if best_value < f1_score_avg:
            best_value = f1_score_avg
            # state_dict() = snapshot of all model weights as a dict (like model.get_weights() in TF)
            # deepcopy because state_dict returns references that would change as training continues
            best_state = deepcopy(model.state_dict())
            print(f'Picked! => Best value {f1_score_avg:.3f}')
            early_stop_count = 0

        if early_stop_count == hyp.early_stop_n_epochs:
            print(f'Early stop with best value {best_value:.3f}')
            break

    # Save best checkpoint and restore it into the model
    torch.save(best_state, str(save_path))  # like model.save_weights() in TF
    model.load_state_dict(best_state)       # like model.load_weights() in TF
    return model


# ==========================================================================
# Task: orchestrates data loading, model creation, training, and evaluation
# ==========================================================================

# BST variant name -> pre-configured constructor (partials defined in bst.py)
MODELS = {
    'BST_0':     BST_0,
    'BST':       BST_PPF,
    'BST_CG':    BST_CG,
    'BST_AP':    BST_AP,
    'BST_CG_AP': BST_CG_AP,
}


class Task:
    def __init__(self, n_joints=17, taxonomy: Taxonomy = None) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.n_joints = n_joints
        self.taxonomy = taxonomy or TAXONOMIES[hyp.taxonomy]

    def prepare_dataloaders(
        self,
        root_dir: Path,
        pose_style='Jn2B',
        train_partial=1.0
    ):
        self.train_loader, \
        self.val_loader, \
        self.test_loader \
            = prepare_npy_collated_loaders(
                root_dir=root_dir,
                pose_style=pose_style,
                batch_size=hyp.batch_size,
                use_cuda=self.use_cuda,
                num_workers=(0, 0, 0),
                train_partial=train_partial
            )

        self.pose_style = pose_style

    def get_network_architecture(self, model_name='BST_CG_AP', in_channels=2):
        """Create model with the right input dimensions and optional modules.
        in_channels: 2 for 2D (xy) keypoints, 3 for 3D (xyz)."""
        ModelClass = MODELS[model_name]  # pre-configured partial from bst.py
        n_bones = len(get_bone_pairs())  # 19 bone vectors for COCO 17-joint skeleton
        extra = POSE_BONE_MULTIPLIER[self.pose_style]

        self.net = ModelClass(
            in_dim=(self.n_joints + n_bones * extra) * in_channels,
            n_class=self.taxonomy.n_classes,
            seq_len=hyp.seq_len,
            depth_tem=2,       # 2 layers in temporal transformer
            depth_inter=1,     # 1 layer in interactional transformer
        ).to(self.device)  # move entire model to GPU/CPU

        self.model_name = model_name
        self.n_bones = n_bones

    def seek_network_weights(self, model_info='', serial_no=1):
        """Load existing weights if found, otherwise train from scratch.
        Weight filenames encode the full experiment config, e.g.:
        'bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged_2.pt'
        """
        model_info = f'_{model_info}' if model_info != '' else ''
        taxonomy_info = f'_{self.taxonomy.name}'
        serial_str = f'_{serial_no}' if serial_no != 1 else ''

        model_postfix = '_' + self.pose_style \
            + model_info + taxonomy_info + serial_str

        # Weight filename: 'BST_CG_AP' -> 'bst_CG_AP', 'BST_0' -> 'bst_0', 'BST' -> 'bst'
        if '_' in self.model_name:
            first, rest = self.model_name.split('_', 1)
            save_name = first.lower() + '_' + rest
        else:
            save_name = self.model_name.lower()
        save_name += model_postfix

        self.model_name += model_postfix

        weight_path = Path(f'weight/{save_name}.pt')
        if weight_path.exists():
            self.net.load_state_dict(
                torch.load(str(weight_path), map_location=self.device, weights_only=True)
            )
            return True  # weight already existed
        else:
            train_t0 = time.time()
            self.net = train_network(
                model=self.net,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
                save_path=weight_path,
                n_bones=len(get_bone_pairs()) if self.pose_style != 'J_only' else 0,
                n_classes=self.taxonomy.n_classes,
            )
            t = timedelta(seconds=int(time.time() - train_t0))
            print(f'Total training time: {t}')
            return False  # newly trained

    def test(self, show_details=False, show_confusion_matrix=False):
        pred, gt = test(self.net, self.test_loader, self.device)
        print(f'Test (num_strokes: {len(pred)}) =>')

        f1_score_each = multiclass_f1_score(
            pred, gt, num_classes=self.taxonomy.n_classes, average=None
        )
        show_f1_results(
            model_name=self.model_name,
            f1_score_each=f1_score_each,
            class_ls=pad_class_labels(self.taxonomy.class_list()),
            show_details=show_details
        )

        acc = torch.sum(pred == gt).item() / len(pred)
        print('Accuracy:', f'{acc:.3f}')

        if show_confusion_matrix:
            plot_confusion_matrix(
                y_true=gt,
                y_pred=pred,
                need_pre_argmax=False,
                model_name=self.model_name,
                font_size=6,
                save=False
            )

    def test_topk_acc(self, k=2):
        assert k > 1, 'k should be > 1'
        pred, gt = test_topk(self.net, self.test_loader, self.device, k=k)
        gt = gt.unsqueeze(1).repeat(1, k)
        acc = torch.any(pred == gt, dim=1).sum().item() / len(gt)
        print(f'Top{k} Accuracy: {acc:.3f}')

    def compare_pred_gt_on_specific_type(self, dir_path: Path):
        infer_ds = Dataset_npy(
            root_dir=dir_path,
            set_name='test_specific',
            pose_style=self.pose_style,
            seq_len=hyp.seq_len,
            taxonomy=self.taxonomy,
        )
        infer_loader = DataLoader(
            dataset=infer_ds,
            batch_size=hyp.batch_size,
        )

        pred, gt = test(self.net, infer_loader, self.device)
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()

        not_match = pred != gt
        class_ls = self.taxonomy.class_list()
        with pd.option_context('display.max_rows', None):
            df = pd.DataFrame(
                data={
                    'Ball Round': [Path(e).stem for e in infer_ds.data_branches],
                    'Pred': [class_ls[e] if b else '-' for e, b in zip(pred, not_match)],
                    'GT': [class_ls[e] if b else '-' for e, b in zip(gt, not_match)]
                }
            )
            print(df)


# ==========================================================================
# Main: train and test on ShuttleSet
# ==========================================================================

if __name__ == '__main__':
    additional_model_info = ''
    taxonomy = TAXONOMIES[hyp.taxonomy]

    str_3d = '_3d' if hyp.use_3d_pose else ''
    match hyp.seq_len:
        case 30:
            npy_collated_dir = f'dataset{str_3d}_npy_collated'
            model_info = '3d' if hyp.use_3d_pose else ''
        case 100:
            npy_collated_dir = f'dataset{str_3d}_npy_collated_between_2_hits_with_max_limits_seq_100'
            model_info = f'between_2_hits_with_max_limits_seq_100{str_3d}'
        case _:
            raise NotImplementedError

    assert 0 < hyp.train_partial <= 1, 'hyp.train_partial should be in (0, 1].'
    if hyp.train_partial != 1:
        tmp_str = f'train_partial_0p{str(hyp.train_partial)[2:]}'
        if model_info != '':
            model_info += '_' + tmp_str
        else:
            model_info += tmp_str

    if additional_model_info != '':
        if model_info != '':
            model_info += f'_{additional_model_info}'
        else:
            model_info = additional_model_info

    for serial_no in range(1, 6):
        print(f'Running serial {serial_no} ...')
        task = Task(n_joints=17, taxonomy=taxonomy)
        task.prepare_dataloaders(
            root_dir=Path(f'preparing_data/ShuttleSet_data_{taxonomy.name}')
                         /npy_collated_dir,
            pose_style=hyp.pose_style,
            train_partial=hyp.train_partial
        )
        task.get_network_architecture(model_name='BST_CG_AP', in_channels=(3 if hyp.use_3d_pose else 2))
        weight_exists = task.seek_network_weights(model_info=model_info, serial_no=serial_no)
        task.test(show_details=False, show_confusion_matrix=False)
        task.test_topk_acc(k=2)
        print('Serial', serial_no, 'done.')

        if not weight_exists:
            time.sleep(3)
