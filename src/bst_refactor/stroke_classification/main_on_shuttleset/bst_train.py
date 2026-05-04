# BST training script for ShuttleSet.
#
# Run from the repo root with both package roots on PYTHONPATH:
#   PYTHONPATH=src/bst_refactor:src/bst_refactor/stroke_classification \
#       python -m main_on_shuttleset.bst_train
#
# PyTorch training loop overview (differs significantly from TF/Keras):
#   TF:      model.compile(optimizer, loss) -> model.fit(data)  (one line trains everything)
#   PyTorch: you write the loop yourself — iterate batches, compute loss, call backward(), step()
#   This is more verbose but gives full control over every training step.

import torch
from torch import Tensor, nn, optim  # nn = layers/models, optim = optimizers (like tf.keras.optimizers)
import torch.nn.functional as F      # F = stateless functions (one_hot, softmax, etc.)
from torch.utils.tensorboard import SummaryWriter  # TensorBoard logging (same viewer as TF)
from torcheval.metrics.functional import multiclass_f1_score

from transformers import get_cosine_schedule_with_warmup  # from HuggingFace, not a custom module

from pathlib import Path
from copy import deepcopy
from collections import namedtuple
from contextlib import redirect_stdout
import math
import time
from datetime import datetime, timedelta
import sys

import yaml

from preparing_data.shuttleset_dataset import prepare_npy_collated_loaders, \
                                              RandomTranslation_batch, \
                                              pad_class_labels
from result_utils import show_f1_results, plot_confusion_matrix
from pipeline.config import (
    TAXONOMIES,
    Taxonomy,
    derive_ablation_id,
    derive_npy_collated_dir_basename,
)
from run_tracker import track_run, track_serial
from main_on_shuttleset.bst_common import (
    Tee,
    build_bst_network,
    compute_data_provenance,
    derive_active_classes_from_labels,
)
from main_on_shuttleset.loss.adaptive_focal import (
    AdaptiveFocalLoss,
    accumulate_class_counts,
    per_class_f1_from_counts,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CLIPS_CSV = REPO_ROOT / 'notebooks' / 'clips_master.csv'


# ==========================================================================
# Hyperparameters — edit these to change experiment configuration.
# Active LR + aux schedule rationale: scratch/architecture_notes/arch_1_directions.md.
# Dated retune history: scratch/architecture_notes/historical_bst.md section 3.
# ==========================================================================
# ablation_id tags the collated dir so multiple ablations don't collide;
# it defaults to a tuple of (taxonomy, split_column, drop_unknown) when None.
Hyp = namedtuple('Hyp', [
    'n_epochs', 'batch_size', 'lr', 'warm_up_step',
    'taxonomy', 'seq_len', 'early_stop_n_epochs',
    'pose_style', 'use_3d_pose', 'train_partial',
    'use_aux_schedule', 'aux_fade_end_epoch',
    'clips_csv', 'split_column', 'drop_unknown', 'ablation_id',
    'label_smoothing', 'class_weights', 'adaptive_focal',
    'expected_active_classes',
])
hyp = Hyp(
    n_epochs=80,
    early_stop_n_epochs=40,
    batch_size=128,
    lr=5e-4,
    warm_up_step=100,
    taxonomy='une_merge_v1_nosides',
    seq_len=100,
    pose_style='JnB_bone',
    use_3d_pose=False,
    train_partial=1.0,
    use_aux_schedule=True,
    aux_fade_end_epoch=15,
    clips_csv=str(DEFAULT_CLIPS_CSV),
    split_column='split_v2',
    drop_unknown=True,
    ablation_id='wipe_drop',
    label_smoothing=0.0,  # CDB-F1 cell forces LS=0; LS softens targets so confident-correct samples have p_t < 1.0, contaminating focal's per-sample hardness signal
    # Manual per-class CE weights for the wrist_smash <-> smash confusion-pair smoke test.
    # Pair-balanced (both at 2.0) so the gradient has no directional bias toward one class:
    # tests whether loss-side reweighting alone can move the wrist_smash F1 floor without
    # stealing recall from smash. Weights renormalised to mean 1.0 inside the loss build so
    # overall loss scale stays comparable to uniform CE. Set to None for uniform CE.
    class_weights=None,
    # Class-F1-driven adaptive focal loss (CDB-F1). Mutually exclusive with
    # class_weights, and forces label_smoothing=0 (LS contaminates focal's
    # hardness estimate). None disables; pass a dict to engage:
    #   adaptive_focal={
    #       'tau': 1.0, 'gamma': 1.0, 'momentum': 0.9,
    #       'warm_up_epochs': 5, 'f1_floor': 0.0,
    #       # Optional pair-cap rules for known confusion pairs the scalar CDB
    #       # signal can't model. Each rule enforces alpha[numer] >= ratio *
    #       # alpha[denom] after the standard renormalisation, with the bump
    #       # absorbed across the other (n - 2) classes so mean alpha stays 1.0.
    #       'pair_caps': [
    #           {'numer': 'smash', 'denom': 'wrist_smash', 'ratio': 0.7},
    #       ],
    #   }
    # Full design + paper-verified equations: scratch/architecture_notes/class_f1_focal_design.md.
    adaptive_focal={
        # First-run sweet spot from run_20260501_164658: tau=1, gamma=1.
        # All four CDB knob variants (gamma=0, tau=0.5, pair-cap, gamma=2)
        # traded wrist_smash back for smash without macro moving, so this
        # combo holds the floor-lift sweet spot (+8.7 pp wrist_smash on the
        # LS=0.1 baseline). Active default for the capacity-bump runs.
        'tau': 1.0,
        'gamma': 1.0,
        'momentum': 0.9,
        'warm_up_epochs': 5,
        'f1_floor': 0.0,
    },
    # Optional belt-and-braces lever: when non-None, the empirical active class list
    # derived from train labels gets asserted equal to this. Mismatch raises with
    # both lists side-by-side. Default None means "trust the data".
    expected_active_classes=None,
)


# ==========================================================================
# Training and evaluation functions
# ==========================================================================

def aux_schedule_factor(epoch: int, fade_end_epoch: int) -> float:
    """Cosine warm-start-to-fade schedule for CG/AP auxiliary modules.

    Factor is 1.0 at epoch 1, 0.5 at mid-fade, and 0.0 at fade_end_epoch.
    Stays pinned at 0.0 for all epochs beyond fade_end_epoch, giving the
    transformer backbone a pure-solo phase to find its own best representation.

    Decoupling fade_end from n_epochs matters when the historical peak F1
    falls well inside the schedule: setting fade_end_epoch near (or before)
    that peak guarantees CG/AP contribution is meaningfully reduced in the
    peak region, so the experiment actually tests the hypothesis rather than
    running a near-baseline with a mild perturbation.

    :param epoch: current epoch, 1-indexed (matches the training loop).
    :param fade_end_epoch: epoch at which factor first reaches 0.0; stays 0 after.
    :return: scalar in [0, 1].
    """
    if fade_end_epoch <= 1 or epoch >= fade_end_epoch:
        return 0.0 if epoch >= fade_end_epoch else 1.0
    progress = (epoch - 1) / (fade_end_epoch - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(
    model: nn.Module,
    loader,
    random_shift_fn,
    n_bones: int,
    n_classes: int,
    loss_fn,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,  # learning rate scheduler
    device,
) -> tuple[float, Tensor, Tensor, Tensor]:
    """Train for one epoch, accumulating per-class TP / FP / FN alongside loss.

    Per-class counts feed ``AdaptiveFocalLoss.update_alpha`` at the call site.
    They're cheap (three batched ``bincount`` calls per batch) and stay
    accumulated even when the loss has no use for them, so the train loop
    keeps a uniform return signature regardless of which loss is active.

    :return: ``(train_loss, tp, fp, fn)``; counts are length-``n_classes``
        int64 tensors on ``device``.
    """
    model.train()  # enable dropout + batchnorm training mode (TF: training=True)
    total_loss = 0.0
    tp = torch.zeros(n_classes, dtype=torch.long, device=device)
    fp = torch.zeros(n_classes, dtype=torch.long, device=device)
    fn = torch.zeros(n_classes, dtype=torch.long, device=device)

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

        # Per-class confusion counts on argmax preds. no_grad() because preds
        # are detached labels; nothing here needs an autograd graph.
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            batch_tp, batch_fp, batch_fn = accumulate_class_counts(
                preds, labels, n_classes,
            )
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn

    train_loss = total_loss / len(loader)
    return train_loss, tp, fp, fn


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

    # Only classes present in the val set count toward macro/min. Generic
    # zero-support guard: any class with no ground-truth this epoch would
    # otherwise score F1=0 by construction, dragging macro down by 1/n
    # and pinning min at 0.
    present = (cum_tp + cum_fn) > 0
    if present.any():
        f1_score_avg = f1_score[present].mean()
        f1_score_min = f1_score[present].min()
    else:
        f1_score_avg = torch.tensor(0.0)
        f1_score_min = torch.tensor(0.0)
    return val_loss, f1_score_avg, f1_score_min, f1_score, present


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
    class_ls: list[str],
    taxonomy: Taxonomy,
    tb_dir: Path | None = None,
):
    # tb_dir lands the event files under experiments/<run_id>/tb/serial_N/ so
    # TB folders pair with the run they came from. Default SummaryWriter() writes
    # to ./runs/<host_time>/, which is what older runs used.
    writer = SummaryWriter(log_dir=str(tb_dir)) if tb_dir is not None else SummaryWriter()
    random_shift_fn = RandomTranslation_batch()  # data augmentation: small xy shifts

    # label_smoothing softens targets from [0,1] to reduce overconfidence.
    # BST paper / TemPose default is 0.1; we sweep this knob to test
    # whether it's bottlenecking the small-support classes that lose
    # ground when the cleaner Phase-2 pose data lifts the head of the
    # F1 distribution. See scratch/architecture_notes/hparams_sweep_speculations.md.
    #
    # class_weights: optional manual per-class loss multipliers. Used as a
    # smoke test for whether loss-side reweighting can move the bottleneck
    # F1 classes (wrist_smash + its confusion partner smash). Renormalised to
    # mean 1.0 so the overall loss magnitude stays comparable to uniform CE
    # (keeps LR / grad-clip behaviour aligned across cells). None = uniform.
    #
    # adaptive_focal: class-F1-driven CDB-loss with optional focal modulation.
    # Replaces the static class_weights lever with an EMA-smoothed per-class
    # weight that re-prioritises classes whose train F1 stays low. Mutually
    # exclusive with class_weights and forces label_smoothing=0 (LS softens
    # targets, contaminating focal's per-sample hardness estimate).
    if hyp.adaptive_focal is not None:
        if hyp.class_weights:
            raise ValueError(
                'adaptive_focal and class_weights are mutually exclusive; '
                'set one of them to None.'
            )
        if hyp.label_smoothing != 0.0:
            raise ValueError(
                'adaptive_focal requires label_smoothing=0.0 (LS softens '
                'targets so confident-correct samples have p_t < 1.0, '
                "contaminating focal's per-sample hardness signal). "
                f'Got label_smoothing={hyp.label_smoothing}.'
            )
        af_cfg = hyp.adaptive_focal
        loss_fn = AdaptiveFocalLoss(
            n_classes=n_classes,
            class_names=class_ls,
            tau=af_cfg.get('tau', 1.0),
            gamma=af_cfg.get('gamma', 1.0),
            momentum=af_cfg.get('momentum', 0.9),
            warm_up_epochs=af_cfg.get('warm_up_epochs', 5),
            f1_floor=af_cfg.get('f1_floor', 0.0),
            pair_caps=af_cfg.get('pair_caps'),
            device=device,
        )
        # Print resolved pair_caps as triples (rather than the dict spec) so the
        # log shows the index lookup succeeded against the active class list.
        pair_cap_str = (
            ', '.join(
                f'{class_ls[n]}/{class_ls[d]}>={r:.2f}'
                for n, d, r in loss_fn.pair_caps
            )
            if loss_fn.pair_caps else 'none'
        )
        print(
            f"[loss] adaptive focal (CDB-F1): "
            f"tau={loss_fn.tau}, gamma={loss_fn.gamma}, "
            f"momentum={loss_fn.momentum}, "
            f"warm_up_epochs={loss_fn.warm_up_epochs}, "
            f"f1_floor={loss_fn.f1_floor}, "
            f"pair_caps=[{pair_cap_str}]"
        )
    elif hyp.class_weights:
        weights = torch.ones(n_classes, device=device)
        for cls_name, multiplier in hyp.class_weights.items():
            if cls_name not in class_ls:
                raise ValueError(
                    f"class_weights key '{cls_name}' not in active class list "
                    f"{class_ls}. (Full taxonomy {taxonomy.name!r} has "
                    f"{len(taxonomy.class_list())} classes; this run uses "
                    f"{len(class_ls)}.)"
                )
            weights[class_ls.index(cls_name)] = multiplier
        weights = weights * (n_classes / weights.sum())  # renormalise mean to 1.0
        print(f"[loss] class-weighted CE (renormalised, mean=1.0):")
        for i, c in enumerate(class_ls):
            print(f"    {c:25s} weight={weights[i].item():.3f}")
        loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=hyp.label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp.label_smoothing)
    # AdamW = Adam with decoupled weight decay (standard for transformers)
    # model.parameters() returns all learnable weights (TF equivalent: model.trainable_variables)
    optimizer = optim.AdamW(model.parameters(), lr=hyp.lr)
    # Cosine schedule: LR ramps up during warmup, then decays following a cosine curve.
    # HF formula: lr_factor = 0.5 * (1 + cos(pi * 2 * num_cycles * progress))
    #   num_cycles=0.5 -> LR ends at 0 (full standard cosine descent)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=hyp.warm_up_step,
        num_training_steps=(hyp.n_epochs * len(train_loader)),  # total batches across all epochs
        num_cycles=0.5
    )

    # Track top-2 of each metric (for HParams summary + verifying early-stop vs crash)
    best_macro = second_macro = 0.0
    best_macro_epoch = second_macro_epoch = 0
    best_min = second_min = 0.0
    best_min_epoch = second_min_epoch = 0
    best_val_loss, best_val_loss_epoch = float('inf'), 0
    early_stop_count = 0

    for epoch in range(1, hyp.n_epochs+1):
        # Auxiliary module schedule: cosine fade of CG/AP from 1.0 -> 0.0 across the run.
        # When disabled, factor stays at 1.0 -> identical to unscheduled BST_CG_AP.
        if hyp.use_aux_schedule:
            aux_factor = aux_schedule_factor(epoch, hyp.aux_fade_end_epoch)
        else:
            aux_factor = 1.0
        model.set_schedule_factors(cg_factor=aux_factor, ap_factor=aux_factor)

        t0 = time.time()
        train_loss, train_tp, train_fp, train_fn = train_one_epoch(
            model=model,
            loader=train_loader,
            random_shift_fn=random_shift_fn,
            n_bones=n_bones,
            n_classes=n_classes,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        # End-of-epoch per-class train F1 feeds AdaptiveFocalLoss; otherwise
        # the values are still computed (cheap) and logged to TB for context.
        train_per_class_f1 = per_class_f1_from_counts(train_tp, train_fp, train_fn)
        if isinstance(loss_fn, AdaptiveFocalLoss):
            loss_fn.update_alpha(train_per_class_f1)

        val_loss, f1_score_avg, f1_score_min, f1_per_class, present = validate(
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

        if isinstance(loss_fn, AdaptiveFocalLoss):
            # Top-3 / bot-3 alpha summary so the operator can eyeball whether
            # the loss is reweighting toward the struggling classes each epoch.
            alpha_np = loss_fn.alpha.detach().cpu().numpy()
            order = alpha_np.argsort()
            print('  alpha bot3: ' + ' '.join(
                f'{class_ls[i]}={alpha_np[i]:.2f}' for i in order[:3]
            ))
            print('  alpha top3: ' + ' '.join(
                f'{class_ls[i]}={alpha_np[i]:.2f}' for i in order[-3:][::-1]
            ))

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('F1/Val_macro', f1_score_avg, epoch)
        writer.add_scalar('F1/Val_min', f1_score_min, epoch)
        # Train F1 macro/min summaries mirror the val pair above, so the
        # train-vs-val gap reads off two scalars per epoch instead of needing
        # to re-aggregate the per-class arrays. .mean()/.min() over the
        # length-n_classes tensor of active-class F1s, .item() unwraps to float.
        writer.add_scalar('F1_train/macro', train_per_class_f1.mean().item(), epoch)
        writer.add_scalar('F1_train/min', train_per_class_f1.min().item(), epoch)
        writer.add_scalar('Schedule/aux_factor', aux_factor, epoch)
        for i, c in enumerate(class_ls):
            writer.add_scalar(f'F1_train/{c}', train_per_class_f1[i].item(), epoch)
            if isinstance(loss_fn, AdaptiveFocalLoss):
                writer.add_scalar(f'Alpha/{c}', loss_fn.alpha[i].item(), epoch)

        curr_macro, curr_min = f1_score_avg.item(), f1_score_min.item()

        # Early stop + snapshot best weights (piggybacks on new-best detection)
        early_stop_count += 1
        if curr_macro > best_macro:
            second_macro, second_macro_epoch = best_macro, best_macro_epoch
            best_macro, best_macro_epoch = curr_macro, epoch
            # state_dict() = snapshot of all model weights as a dict (like model.get_weights() in TF)
            # deepcopy because state_dict returns references that would change as training continues
            best_state = deepcopy(model.state_dict())
            print(f'Picked! => Best value {curr_macro:.3f}')
            # Compact per-class snapshot on new-best epochs: top-5 and bot-5
            # of present classes, one line each. Full per-class breakdown
            # lands in the test-time log at the end of each serial.
            present_idx = present.nonzero(as_tuple=True)[0].tolist()
            scored = sorted(
                [(class_ls[i], f1_per_class[i].item()) for i in present_idx],
                key=lambda t: t[1],
            )
            print('  val top5: ' + ' '.join(
                f'{n}={v:.2f}' for n, v in reversed(scored[-5:])
            ))
            print('  val bot5: ' + ' '.join(
                f'{n}={v:.2f}' for n, v in scored[:5]
            ))
            early_stop_count = 0
        elif curr_macro > second_macro:
            second_macro, second_macro_epoch = curr_macro, epoch

        if curr_min > best_min:
            second_min, second_min_epoch = best_min, best_min_epoch
            best_min, best_min_epoch = curr_min, epoch
        elif curr_min > second_min:
            second_min, second_min_epoch = curr_min, epoch

        best_val_loss, best_val_loss_epoch = min(
            (best_val_loss, best_val_loss_epoch), (val_loss, epoch)
        )

        if early_stop_count == hyp.early_stop_n_epochs:
            print(f'Early stop with best value {best_macro:.3f}')
            break

    # Save best checkpoint and restore it into the model. Done before TB
    # hparam logging so a logging failure doesn't lose the trained weights.
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(save_path))  # like model.save_weights() in TF
    model.load_state_dict(best_state)       # like model.load_weights() in TF

    # HParams summary: one row per run, sortable in TB's HParams tab.
    # stopped_epoch - best_macro_epoch == early_stop_n_epochs confirms clean early-stop.
    # Coerce non-scalar values (dicts, None, etc.) to strings; TB's add_hparams
    # only accepts int / float / str / bool / Tensor.
    def _to_hparam_value(value):
        if isinstance(value, (int, float, str, bool)) or torch.is_tensor(value):
            return value
        return str(value)

    writer.add_hparams(
        hparam_dict={k: _to_hparam_value(v) for k, v in hyp._asdict().items()},
        metric_dict={
            'best/macro_f1':        best_macro,
            'best/macro_f1_epoch':  best_macro_epoch,
            'best/macro_f1_2nd':    second_macro,
            'best/macro_f1_2nd_ep': second_macro_epoch,
            'best/min_f1':          best_min,
            'best/min_f1_epoch':    best_min_epoch,
            'best/min_f1_2nd':      second_min,
            'best/min_f1_2nd_ep':   second_min_epoch,
            'best/val_loss':        best_val_loss,
            'best/val_loss_epoch':  best_val_loss_epoch,
            'stopped_epoch':        epoch,
        },
        run_name='.',
        global_step=epoch,
    )
    writer.close()

    return model


# ==========================================================================
# Task: orchestrates data loading, model creation, training, and evaluation
# ==========================================================================


class Task:
    def __init__(self, n_joints=17, taxonomy: Taxonomy = None,
                 weight_dir: Path = Path('weight')) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.n_joints = n_joints
        self.taxonomy = taxonomy or TAXONOMIES[hyp.taxonomy]
        # Active label space gets derived from labels.npy in
        # prepare_dataloaders; architecture is a function of the data here,
        # not of any flag. Stays None until the dataloaders return.
        self.n_active_classes: int | None = None
        self.active_class_list: list[str] | None = None
        # Where to save/load weights for this run. Caller should pass a
        # per-invocation subdir (e.g. weight/run_YYYYMMDD_HHMMSS) so fresh
        # runs never collide with older weights — see __main__ setup.
        self.weight_dir = weight_dir

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
        self._derive_active_classes_from_loaded_labels()

    def _derive_active_classes_from_loaded_labels(self) -> None:
        """Inspect loaded labels (post-train_partial slicing), build active
        class list + remap, write back. Architecture is sized to whatever
        train can teach. val/test must be subsets of train's present classes;
        rogue val/test labels raise ValueError with a clear message.

        ``labels.npy`` on disk stays untouched; existing collated dirs and
        weight files keep working without recollation.
        """
        train_ds = self.train_loader.dataset
        val_ds = self.val_loader.dataset
        test_ds = self.test_loader.dataset

        active, _remap, remapped = derive_active_classes_from_labels(
            taxonomy=self.taxonomy,
            train_labels=train_ds.labels,
            validation_label_arrays={
                'val':  val_ds.labels,
                'test': test_ds.labels,
            },
        )
        train_ds.labels = remapped['train']
        val_ds.labels = remapped['val']
        test_ds.labels = remapped['test']
        self.active_class_list = active
        self.n_active_classes = len(active)

    def get_network_architecture(self, model_name='BST_CG_AP', in_channels=2):
        """Create the model at the active head dim and ground its inputs.

        :param in_channels: 2 for 2D (xy) keypoints, 3 for 3D (xyz).

        Output dim is ``n_active_classes``, derived empirically from the
        train labels in ``_derive_active_classes_from_loaded_labels``.
        Whenever the data has fewer classes than the full taxonomy
        (e.g. unknown empty after the writer drop), the head shrinks to
        match — no ghost output channel.
        """
        self.net, self.n_bones = build_bst_network(
            model_name,
            n_joints=self.n_joints,
            pose_style=self.pose_style,
            in_channels=in_channels,
            n_class=self.n_active_classes,
            seq_len=hyp.seq_len,
            device=self.device,
        )
        self.model_name = model_name

    def seek_network_weights(self, model_info='', serial_no=1, tb_dir: Path | None = None):
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

        weight_path = self.weight_dir / f'{save_name}.pt'
        self.weight_path = weight_path
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
                n_bones=self.n_bones,
                n_classes=self.n_active_classes,
                class_ls=self.active_class_list,
                taxonomy=self.taxonomy,
                tb_dir=tb_dir,
            )
            t = timedelta(seconds=int(time.time() - train_t0))
            print(f'Total training time: {t}')
            return False  # newly trained

    def test(self, show_details=False, show_confusion_matrix=False) -> dict:
        pred, gt = test(self.net, self.test_loader, self.device)
        print(f'Test (num_strokes: {len(pred)}) =>')

        f1_score_each = multiclass_f1_score(
            pred, gt, num_classes=self.n_active_classes, average=None
        )

        # Mirror validate(): generic zero-support guard. Any active class
        # with no ground truth in the test set would otherwise score F1=0
        # by construction, dragging macro down by 1/n and pinning min at 0.
        present = torch.bincount(gt, minlength=self.n_active_classes) > 0
        present_idx = present.nonzero(as_tuple=True)[0].tolist()
        class_ls = self.active_class_list

        show_f1_results(
            model_name=self.model_name,
            f1_score_each=f1_score_each[present_idx] if present_idx else f1_score_each,
            class_ls=pad_class_labels(
                [class_ls[i] for i in present_idx] if present_idx else class_ls
            ),
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

        if present_idx:
            macro_f1 = float(f1_score_each[present_idx].mean().item())
            min_f1 = float(f1_score_each[present_idx].min().item())
            per_class_f1 = {
                class_ls[i]: float(f1_score_each[i].item()) for i in present_idx
            }
        else:
            macro_f1 = 0.0
            min_f1 = 0.0
            per_class_f1 = {}

        return {
            'macro_f1':     macro_f1,
            'min_f1':       min_f1,
            'accuracy':     float(acc),
            'num_strokes':  int(len(pred)),
            'per_class_f1': per_class_f1,
        }

    def test_topk_acc(self, k=2) -> dict:
        assert k > 1, 'k should be > 1'
        pred, gt = test_topk(self.net, self.test_loader, self.device, k=k)
        gt = gt.unsqueeze(1).repeat(1, k)
        acc = torch.any(pred == gt, dim=1).sum().item() / len(gt)
        print(f'Top{k} Accuracy: {acc:.3f}')
        return {f'top{k}_accuracy': float(acc)}


# ==========================================================================
# Per-run arch validation and manifest enrichment
# ==========================================================================

def _validate_and_record_arch(
    *,
    run_dir: Path,
    task: 'Task',
    taxonomy: Taxonomy,
    hyp: 'Hyp',
    resumed_manifest_arch: dict | None,
    tee,
) -> None:
    """First-serial post-prepare hook: surface arch invariants and update manifest.

    Runs once per run, after the first serial's ``prepare_dataloaders``
    has populated the active class list from the data. Builds the loud
    ``[arch]`` printout (captured by the tee'd log), checks
    ``hyp.expected_active_classes`` if set, checks resume consistency
    against the loaded manifest's ``extra.arch`` if resuming, then
    rewrites the manifest's ``extra.arch`` block in place.

    :param run_dir: experiments/<run_id>/ for this run.
    :param task: Task whose ``n_active_classes`` and ``active_class_list``
        have been populated.
    :param taxonomy: full taxonomy under which the run is operating.
    :param hyp: the active Hyp tuple (for the optional
        ``expected_active_classes`` lever).
    :param resumed_manifest_arch: ``extra.arch`` block read from the
        existing manifest when ``resume_from`` is set, else None.
    :param tee: file-like object writing to terminal + log_path so the
        ``[arch]`` print lands in both.
    """
    arch_block = {
        'n_classes_full':    taxonomy.n_classes,
        'n_active_classes':  task.n_active_classes,
        'has_unknown':       taxonomy.has_unknown,
        'unknown_first':     taxonomy.unknown_first,
        'active_class_list': task.active_class_list,
    }

    with redirect_stdout(tee):
        print(
            f'[arch] taxonomy={taxonomy.name}, '
            f'has_unknown={taxonomy.has_unknown}, '
            f'unknown_first={taxonomy.unknown_first}'
        )
        print(
            f'       full taxonomy n_classes={taxonomy.n_classes}, '
            f'n_active_classes={task.n_active_classes} '
            f'(derived from train labels post train_partial)'
        )
        print(f'       active class_list: {task.active_class_list}')

    expected = getattr(hyp, 'expected_active_classes', None)
    if expected is not None and list(expected) != list(task.active_class_list):
        raise ValueError(
            f'hyp.expected_active_classes != empirical active list.\n'
            f'  expected ({len(expected)}): {expected}\n'
            f'  empirical ({task.n_active_classes}): {task.active_class_list}\n'
            f'  Either fix the expectation, fix the dir, or set '
            f'expected_active_classes=None to trust the data.'
        )

    if resumed_manifest_arch is not None:
        prev_n = resumed_manifest_arch.get('n_active_classes')
        prev_list = resumed_manifest_arch.get('active_class_list')
        if prev_n != task.n_active_classes or prev_list != task.active_class_list:
            raise ValueError(
                f'Resume manifest disagrees with live derivation:\n'
                f'  manifest: n_active={prev_n}, list={prev_list}\n'
                f'  live:     n_active={task.n_active_classes}, '
                f'list={task.active_class_list}\n'
                f'  Has the collated dir contents changed since the original run?'
            )

    manifest_path = run_dir / 'manifest.yaml'
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}
    manifest.setdefault('extra', {})['arch'] = arch_block
    with open(manifest_path, 'w') as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)


# ==========================================================================
# Main: train and test on ShuttleSet
# ==========================================================================

if __name__ == '__main__':
    taxonomy = TAXONOMIES[hyp.taxonomy]

    # Collated dir naming via shared helper (mirrored on the prepare_train
    # writer side); see ``pipeline.config.derive_npy_collated_dir_basename``.
    if hyp.seq_len not in (30, 100):
        raise NotImplementedError(f'Unsupported hyp.seq_len={hyp.seq_len!r}; expected 30 or 100.')
    effective_ablation_id = derive_ablation_id(
        taxonomy.name, hyp.split_column, hyp.drop_unknown, hyp.ablation_id,
    )
    npy_collated_dir = derive_npy_collated_dir_basename(
        taxonomy_name=taxonomy.name,
        split_column=hyp.split_column,
        drop_unknown=hyp.drop_unknown,
        use_3d_pose=hyp.use_3d_pose,
        seq_len=hyp.seq_len,
        ablation_id=hyp.ablation_id,
    )

    # Weights filename suffix. Independent of the collated-dir name; encodes
    # config knobs that change per run (seq_len-derived window tag, 3d flag,
    # train_partial). Empty string is a valid value (seq_len=30, 2D, full data).
    str_3d = '_3d' if hyp.use_3d_pose else ''
    model_info_parts: list[str] = []
    if hyp.seq_len == 100:
        model_info_parts.append(f'between_2_hits_with_max_limits_seq_100{str_3d}')
    elif hyp.use_3d_pose:
        model_info_parts.append('3d')
    assert 0 < hyp.train_partial <= 1, 'hyp.train_partial should be in (0, 1].'
    if hyp.train_partial != 1:
        model_info_parts.append(f'train_partial_0p{str(hyp.train_partial)[2:]}')
    model_info = '_'.join(model_info_parts)

    # ----------------------------------------------------------------------
    # Per-run experiment folder (tracked via run_tracker).
    # Every invocation mints a fresh experiments/run_<timestamp>/ with:
    #   manifest.yaml          (hyperparams, git SHA, per-serial metrics)
    #   weights/<save_name>.pt (best checkpoint per serial)
    #   tb/serial_N/           (TB event files per serial)
    # Old flat weight/<name>.pt layout caused silent training-skips after
    # hyperparam changes, so the cache is scoped to the run folder.
    #
    # To re-test an existing run without retraining, set resume_from to its
    # folder name (e.g. 'run_20260417_091933'). The cache then finds saved
    # weights under experiments/<resume_from>/weights/ and skips training.
    # Only tracker-era runs resume cleanly; legacy weight/run_*/ folders
    # need their .pt files copied into experiments/<id>/weights/ first.
    # Leave as None for normal fresh-train behaviour.
    #
    # Resume mutates the run's manifest.yaml in place on serial 1
    # (`_validate_and_record_arch` rewrites the file with the live arch
    # derivation, so historical formatting / ordering does not survive).
    # The script auto-backs the manifest up to
    # ``manifest.yaml.<timestamp>.bak`` next to the live file before any
    # work begins, so the original is always recoverable. Test logs are
    # safe regardless: each invocation gets a fresh timestamped log file.
    # ----------------------------------------------------------------------
    resume_from: str | None = None

    timestamp = f'{datetime.now():%Y%m%d_%H%M%S}'
    run_id = resume_from or f'run_{timestamp}'

    # Test output is auto-teed to a timestamped log file so metrics are never
    # lost to a dropped terminal. Training stdout stays on terminal only; TB
    # captures it. One log file per script invocation, all serials inside.
    # Uses the fresh invocation timestamp (not run_id) so resumed re-tests
    # don't overwrite the original run's log file.
    #
    # Anchor test_logs/ and experiments/ to this file's directory so the
    # write paths don't depend on cwd. Lets `python -m main_on_shuttleset.bst_train`
    # land outputs next to the script regardless of where it was invoked from.
    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / 'test_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'test_{timestamp}.log'
    experiments_dir = script_dir / 'experiments'

    # Resume mutates manifest.yaml on serial 1 (see _validate_and_record_arch).
    # Snapshot the original alongside the live file so the historical
    # formatting / ordering is always recoverable, no user action required.
    if resume_from:
        src_manifest = experiments_dir / resume_from / 'manifest.yaml'
        if src_manifest.exists():
            backup = src_manifest.parent / f'{src_manifest.name}.{timestamp}.bak'
            backup.write_bytes(src_manifest.read_bytes())
            print(f'[resume] backed up original manifest to {backup.name}')
        print(
            f'[resume] resuming {resume_from!r}; manifest.yaml will be '
            f'rewritten in place on serial 1.'
        )

    extra = compute_data_provenance(
        clips_csv_path=Path(hyp.clips_csv),
        effective_ablation_id=effective_ablation_id,
        npy_collated_dir=npy_collated_dir,
    )
    # extra['arch'] stays unset here. _validate_and_record_arch populates
    # it on the first serial after prepare_dataloaders has run the live
    # derivation against the actual training data (post train_partial).
    run_dir, run_id = track_run(
        config=hyp, run_id=run_id, log_path=log_path, extra=extra,
        experiments_dir=experiments_dir,
    )
    weight_dir = run_dir / 'weights'

    collated_root = (
        Path(__file__).resolve().parent.parent
        / f'preparing_data/ShuttleSet_data_{taxonomy.name}'
        / npy_collated_dir
    )

    # Resume cross-check: capture the original manifest's arch block now so
    # the live derivation can be compared against it on serial 1.
    resumed_manifest_arch: dict | None = None
    if resume_from:
        manifest_path = run_dir / 'manifest.yaml'
        if manifest_path.exists():
            with open(manifest_path) as f:
                existing = yaml.safe_load(f) or {}
            resumed_manifest_arch = existing.get('extra', {}).get('arch')
        if resumed_manifest_arch is None:
            print(
                '[arch:resume] resuming pre-fix run; load_state_dict will '
                'fail on shape mismatch for v1/nosides/raw_35 dropunk weights.'
            )

    with open(log_path, 'w') as log_f:
        tee = Tee(sys.stdout, log_f)
        for serial_no in range(1, 6):
            print(f'Running serial {serial_no} ...')
            task = Task(
                n_joints=17, taxonomy=taxonomy, weight_dir=weight_dir,
            )
            task.prepare_dataloaders(
                root_dir=collated_root,
                pose_style=hyp.pose_style,
                train_partial=hyp.train_partial
            )

            # First-serial-only: surface the live arch derivation, run the
            # expected_active_classes lever and the resume cross-check, and
            # write the arch block into the manifest. Subsequent serials use
            # the same dir, so the derivation is deterministic and we just
            # don't repeat the manifest write.
            if serial_no == 1:
                _validate_and_record_arch(
                    run_dir=run_dir,
                    task=task,
                    taxonomy=taxonomy,
                    hyp=hyp,
                    resumed_manifest_arch=resumed_manifest_arch,
                    tee=tee,
                )

            task.get_network_architecture(model_name='BST_CG_AP', in_channels=(3 if hyp.use_3d_pose else 2))

            tb_dir = run_dir / 'tb' / f'serial_{serial_no}'
            weight_exists = task.seek_network_weights(
                model_info=model_info, serial_no=serial_no, tb_dir=tb_dir,
            )

            with redirect_stdout(tee):
                print(f'\n=== Serial {serial_no} ({task.model_name}) ===')
                test_metrics = task.test(show_details=True, show_confusion_matrix=False)
                topk_metrics = task.test_topk_acc(k=2)

            track_serial(
                run_dir=run_dir,
                serial_no=serial_no,
                weights_path=task.weight_path,
                tb_dir=tb_dir,
                metrics={**test_metrics, **topk_metrics},
            )

            print('Serial', serial_no, 'done.')

            if not weight_exists:
                time.sleep(3)

    print(f'\nTest log saved to: {log_path}')
    print(f'Run manifest:    {run_dir / "manifest.yaml"}')
