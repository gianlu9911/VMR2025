#!/usr/bin/env python3
"""
Sequential fine-tuning (images-only) - NO separate head / LwF support

"""

import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

# project imports (same as your repo)
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model

from typing import Sequence, Optional, List
from torch.utils.data import Sampler
import math
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------- BalancedBatchSampler (fixed) --------------------------
class BalancedBatchSampler(Sampler):
    """
    Yields full-batches (lists of indices) and therefore is intended to be
    passed to DataLoader as `batch_sampler=BalancedBatchSampler(...)`.
    """
    def __init__(self,
                 labels: Sequence[int] | torch.Tensor,
                 batch_size: int,
                 oversample: bool = False,
                 shuffle: bool = True,
                 replacement: bool = False,
                 seed: Optional[int] = None,
                 drop_last: bool = True,
                 verbose: bool = False):
        # unify labels to numpy array of ints
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().long().numpy()
        else:
            labels = np.asarray(labels, dtype=np.int64)

        self.labels = labels
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.oversample = bool(oversample)
        self.replacement = bool(replacement)
        self.drop_last = bool(drop_last)
        self.verbose = bool(verbose)

        self.classes = np.sort(np.unique(self.labels)).tolist()
        self.num_classes = len(self.classes)
        if self.num_classes == 0:
            raise ValueError("No classes found in labels.")
        if self.batch_size % self.num_classes != 0:
            raise ValueError("batch_size must be a multiple of number of classes (got batch_size=%d, num_classes=%d)" % (self.batch_size, self.num_classes))

        self.per_class = self.batch_size // self.num_classes

        # build per-class index arrays (numpy ints)
        self.class_to_indices = {}
        for cls in self.classes:
            idxs = np.nonzero(self.labels == cls)[0].astype(np.int64)
            if idxs.size == 0:
                raise ValueError(f"Class {cls} has no samples.")
            self.class_to_indices[int(cls)] = idxs

        lens = [len(v) for v in self.class_to_indices.values()]
        self.min_class_len = min(lens)
        self.max_class_len = max(lens)

        # compute number of batches (careful about zero)
        if self.oversample:
            # need enough batches to cover the largest class
            self.num_batches = math.ceil(self.max_class_len / float(self.per_class))
        else:
            # limited by smallest class
            self.num_batches = self.min_class_len // self.per_class

        if self.num_batches <= 0:
            raise ValueError("Computed zero batches for BalancedBatchSampler: reduce per_class or supply more data. (min_class_len=%d, per_class=%d)" % (self.min_class_len, self.per_class))

        # RNGs: keep both numpy RNG and torch Generator for compatibility
        self.seed = None if seed is None else int(seed)
        self.torch_gen = torch.Generator()
        if self.seed is not None:
            self.torch_gen.manual_seed(self.seed)
            self.np_rng = np.random.RandomState(self.seed)
            random.seed(self.seed)
        else:
            self.np_rng = np.random.RandomState()

    def __len__(self) -> int:
        return int(self.num_batches)

    def __iter__(self):
        per_class_block = self.num_batches * self.per_class
        class_rows = {}

        for cls in self.classes:
            idxs = self.class_to_indices[int(cls)]
            n = len(idxs)

            if self.oversample:
                if self.replacement:
                    # use torch randint for speed/reproducibility
                    choices = torch.randint(low=0, high=n, size=(per_class_block,), generator=self.torch_gen).numpy()
                    selected = idxs[choices]
                else:
                    reps = math.ceil(per_class_block / n)
                    # use numpy RNG for permutation/tiling
                    permuted = np.concatenate([self.np_rng.permutation(idxs) for _ in range(reps)])[:per_class_block]
                    selected = permuted
            else:
                if self.shuffle:
                    perm = self.np_rng.permutation(idxs)
                else:
                    perm = idxs.copy()
                if perm.shape[0] < per_class_block:
                    reps = math.ceil(per_class_block / perm.shape[0])
                    perm = np.tile(perm, reps)[:per_class_block]
                selected = perm[:per_class_block]

            selected = np.asarray(selected, dtype=np.int64).reshape(self.num_batches, self.per_class)
            class_rows[int(cls)] = selected

        for batch_idx in range(self.num_batches):
            parts = [class_rows[int(cls)][batch_idx] for cls in self.classes]
            batch = np.concatenate(parts, axis=0).tolist()
            if self.shuffle:
                # shuffle batch indices in-place with np RNG
                self.np_rng.shuffle(batch)
            yield batch

        if self.verbose:
            used_counts = {int(cls): int(min(self.num_batches * self.per_class, len(self.class_to_indices[int(cls)]))) for cls in self.classes}
            logger.info(f"[BalancedBatchSampler] num_batches={self.num_batches}, per_class={self.per_class}, samples_used_per_class={used_counts}")


# -------------------------- utilities --------------------------

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_classifier_param_list(module: nn.Module):
    """Try to find the final classifier params (common cases like resnet.fc)."""
    if hasattr(module, "resnet") and hasattr(module.resnet, "fc"):
        return list(module.resnet.fc.parameters())
    if hasattr(module, "fc") and isinstance(module.fc, nn.Module):
        return list(module.fc.parameters())
    linear_modules = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if len(linear_modules) > 0:
        return list(linear_modules[-1].parameters())
    return []


def _extract_labels_from_dataset(dataset):
    """
    Robustly extract labels from a dataset.
    Tries fast paths: dataset.targets / dataset.labels / Subset.dataset.targets,
    otherwise falls back to indexing every sample (slower but reliable).
    """
    from torch.utils.data import Subset

    # Helper to check if obj is sequence-like (not counting strings)
    def is_sequence_like(x):
        if x is None:
            return False
        if isinstance(x, (str, bytes)):
            return False
        try:
            len(x)
            return True
        except Exception:
            return False

    # If it's a Subset, consider the underlying dataset first
    if isinstance(dataset, Subset):
        base = dataset.dataset
        idxs = np.asarray(dataset.indices)
        for attr in ("targets", "labels"):
            if hasattr(base, attr):
                attr_val = getattr(base, attr)
                # If attr_val is a torch.Tensor, convert to numpy
                if isinstance(attr_val, torch.Tensor):
                    labels_arr = attr_val.cpu().numpy()
                else:
                    labels_arr = np.asarray(attr_val)
                # If the attribute is sequence-like and 1-D, apply subset indices
                if is_sequence_like(labels_arr) and getattr(labels_arr, "ndim", 1) >= 1:
                    try:
                        return labels_arr[idxs].tolist()
                    except Exception:
                        # fallback to slow path
                        break
                else:
                    # attribute exists but isn't a per-sample sequence -> fall back
                    break
        # fall through to slow path below for Subset

    else:
        # Non-subset: try common attributes directly
        for attr in ("targets", "labels"):
            if hasattr(dataset, attr):
                attr_val = getattr(dataset, attr)
                if isinstance(attr_val, torch.Tensor):
                    labels_arr = attr_val.cpu().numpy()
                else:
                    labels_arr = np.asarray(attr_val)
                if is_sequence_like(labels_arr) and getattr(labels_arr, "ndim", 1) >= 1:
                    return labels_arr.tolist()
                # if attr exists but not sequence-like, break and fall back to indexing

    # Last resort: index every item (slow but safe)
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            lab = item[1]
            if isinstance(lab, torch.Tensor):
                lab = int(lab.item())
            labels.append(int(lab))
        else:
            raise RuntimeError("Unable to extract label from dataset item; dataset must return (x, y).")
    return labels


# -------------------------- evaluation --------------------------

def evaluate_on_images(backbone, test_loader, criterion, device, test_name="test", save_dir="./logs"):
    backbone.eval()
    device = torch.device(device if isinstance(device, str) else device)
    backbone.to(device)

    os.makedirs(save_dir, exist_ok=True)
    rows = []
    losses = []
    sample_idx = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Evaluating {test_name}", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device).long()

            outputs = backbone(imgs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            probs_cpu = probs.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            batch_size = labels_cpu.shape[0]
            num_probs = probs_cpu.shape[1] if probs_cpu.ndim == 2 else 0
            for i in range(batch_size):
                row = {
                    "idx": sample_idx,
                    "label": int(labels_cpu[i]),
                    "pred": int(preds_cpu[i])
                }
                for k in range(num_probs):
                    row[f"prob_class_{k}"] = float(probs_cpu[i, k])
                rows.append(row)
                sample_idx += 1

    avg_loss = float(np.mean(losses)) if len(losses) > 0 else float('nan')
    preds_all = np.array([r['pred'] for r in rows], dtype=int) if len(rows) > 0 else np.array([], dtype=int)
    labels_all = np.array([r['label'] for r in rows], dtype=int) if len(rows) > 0 else np.array([], dtype=int)
    accuracy = float((preds_all == labels_all).sum() / labels_all.shape[0]) if labels_all.size > 0 else float('nan')

    csv_out = os.path.join(save_dir, f"{test_name}_predictions.csv")
    if len(rows) > 0:
        df_preds = pd.DataFrame(rows)
        cols = [c for c in ["idx", "label", "pred"] if c in df_preds.columns]
        prob_cols = sorted([c for c in df_preds.columns if c.startswith("prob_class_")])
        cols.extend(prob_cols)
        df_preds = df_preds[cols]
    else:
        df_preds = pd.DataFrame(columns=["idx", "label", "pred"])

    df_preds.to_csv(csv_out, index=False)

    summary_out = os.path.join(save_dir, f"{test_name}_summary.txt")
    with open(summary_out, "w") as f:
        f.write(f"test_name: {test_name}\n")
        f.write(f"num_samples: {len(labels_all)}\n")
        f.write(f"avg_loss: {avg_loss:.6f}\n")
        f.write(f"accuracy: {accuracy:.6f}\n")

    logger.info(f"[evaluate] Saved predictions CSV to {csv_out} and summary to {summary_out}")
    return avg_loss, accuracy


# -------------------------- training + LwF --------------------------

def train_and_eval(args):
    set_global_seed(args.seed)

    # configure deterministic / benchmark according to flag
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) == 0:
        raise ValueError("No tasks specified. Provide --tasks like 'stylegan1,stylegan_xl,sdv1_4'")

    # load backbone and KEEP its classifier
    backbone = load_pretrained_model(PRETRAINED_MODELS[args.initial_backbone])
    backbone.to(device)

    checkpoint_dir = "./checkpoint_lwf"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("./logs_lwf", exist_ok=True)

    test_domains = {
        "real_vs_stylegan1": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan1'], split='test_set'),
        "real_vs_stylegan2": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan2'], split='test_set'),
        "real_vs_sdv1_4": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['sdv1_4'], split='test_set'),
        "real_vs_stylegan3": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan3'], split='test_set'),
        "real_vs_styleganxl": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan_xl'], split='test_set'),
    }

    results = []
    prev_model = None
    prev_ckpt = None

    for step, task in enumerate(tasks, start=1):
        logger.info("\n" + "="*60)
        logger.info(f"Task {step}/{len(tasks)}: train on {task}")

        # prepare datasets/loaders
        train_dataset = RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR[task])

        # optionally limit number of training samples per task
        if args.max_train_samples and args.max_train_samples > 0:
            from torch.utils.data import Subset
            n_total = len(train_dataset)
            k = min(int(args.max_train_samples), n_total)
            rng = np.random.RandomState(int(args.seed))
            indices = rng.choice(n_total, size=k, replace=False)
            train_dataset = Subset(train_dataset, indices.tolist())
            logger.info(f"Using subset of training data: {k}/{n_total} samples (seed={args.seed})")

        # ---------- prepare train loader ----------
        if args.use_balanced_sampler:
            # Try to extract labels efficiently from dataset
            labels = _extract_labels_from_dataset(train_dataset)
            sampler = BalancedBatchSampler(labels=labels,
                                           batch_size=args.batch_size,
                                           oversample=args.sampler_oversample,
                                           shuffle=True,
                                           replacement=args.sampler_replacement,
                                           seed=args.seed,
                                           drop_last=True,
                                           verbose=False)
            # IMPORTANT: BalancedBatchSampler yields full batches -> pass as batch_sampler
            train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=args.num_workers)
            logger.info("Using BalancedBatchSampler (as batch_sampler) for training loader")
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            logger.info(f"Using standard DataLoader for training with batch size {args.batch_size}")

        logger.info("Training the entire backbone")
        for p in backbone.parameters():
            p.requires_grad = True

        # optimizer: choose params that require grad
        trainable = [p for p in backbone.parameters() if p.requires_grad]
        if len(trainable) == 0:
            raise RuntimeError("No trainable parameters found. Check --train_fc or model structure.")

        # handle classifier param group if desired
        classifier_params = _get_classifier_param_list(backbone)
        param_groups = []

        if len(classifier_params) > 0:
            # identify classifier params that are trainable and others
            cls_param_ids = {id(p) for p in classifier_params}
            cls_params = [p for p in trainable if id(p) in cls_param_ids]
            other_params = [p for p in trainable if id(p) not in cls_param_ids]
            if len(cls_params) > 0:
                param_groups.append({'params': other_params, 'lr': args.backbone_lr})
                param_groups.append({'params': cls_params, 'lr': args.classifier_lr})
            else:
                param_groups.append({'params': trainable, 'lr': args.backbone_lr})
        else:
            param_groups.append({'params': trainable, 'lr': args.backbone_lr})

        optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma) if args.use_scheduler else None
        criterion = nn.CrossEntropyLoss().to(device)

        # create AMP scaler only if using CUDA and user enabled AMP
        use_cuda = ('cuda' in str(device).lower()) and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if (args.use_amp and use_cuda) else None

        # If LwF is enabled and we have a previous model, prepare teacher (move to device only while used)
        teacher = None
        if prev_model is not None:
            # prev_model is stored on CPU if --teacher_on_cpu is set
            teacher = prev_model.to(device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            logger.info("LwF: using previous task model as teacher for distillation")

        # optionally resume from prev_ckpt (if provided)
        if prev_ckpt is not None and os.path.exists(prev_ckpt):
            try:
                ck = torch.load(prev_ckpt, map_location='cpu')
                if 'backbone_state' in ck:
                    backbone.load_state_dict(ck['backbone_state'], strict=False)
                logger.info(f"Loaded previous checkpoint {prev_ckpt}")
            except Exception as e:
                logger.warning(f"[warning] couldn't load prev ckpt: {e}")

        # training loop
        logger.info(f"Starting training for {args.epochs_per_task} epochs on {len(train_dataset)} samples")
        start_time = time.time()

        T = float(args.lwf_temp)
        alpha = float(args.lwf_alpha)
        kd_loss_fn = nn.KLDivLoss(reduction='batchmean')

        for epoch in range(args.epochs_per_task):
            backbone.train()
            losses = []
            correct = 0
            total = 0

            for batch in tqdm(train_loader, desc=f"Train {task} (ep {epoch+1})", leave=False):
                # when using batch_sampler, DataLoader yields (batch_x, batch_y)
                if args.use_balanced_sampler:
                    imgs, labels = batch
                else:
                    imgs, labels = batch

                imgs = imgs.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = backbone(imgs)
                        loss_ce = criterion(outputs, labels)

                        if teacher is not None:
                            with torch.no_grad():
                                teacher_logits = teacher(imgs)
                            log_p = F.log_softmax(outputs / T, dim=1)
                            q = F.softmax(teacher_logits / T, dim=1)
                            loss_kd = kd_loss_fn(log_p, q) * (T * T)
                            loss = (1.0 - alpha) * loss_ce + alpha * loss_kd
                        else:
                            loss = loss_ce

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = backbone(imgs)
                    loss_ce = criterion(outputs, labels)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits = teacher(imgs)
                        log_p = F.log_softmax(outputs / T, dim=1)
                        q = F.softmax(teacher_logits / T, dim=1)
                        loss_kd = kd_loss_fn(log_p, q) * (T * T)
                        loss = (1.0 - alpha) * loss_ce + alpha * loss_kd
                    else:
                        loss = loss_ce

                    loss.backward()
                    optimizer.step()

                losses.append(loss.item())
                preds = outputs.detach().argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            if scheduler is not None:
                scheduler.step()

            train_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
            train_acc = float(correct / total) if total > 0 else 0.0
            if (epoch+1) % max(1, args.log_every) == 0:
                logger.info(f"Epoch [{epoch+1}/{args.epochs_per_task}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        elapsed = time.time() - start_time
        logger.info(f"Finished task {task} training in {elapsed/60:.2f} minutes")

        # save last checkpoint (backbone state + optimizer)
        last_ckpt = os.path.join(checkpoint_dir, f'ft_step{step}_{task}_last.pth')
        torch.save({'backbone_state': backbone.state_dict(), 'optimizer_state': optimizer.state_dict(), 'args': vars(args), 'task': task}, last_ckpt)
        logger.info(f"Saved last checkpoint: {last_ckpt}")

        # evaluation on test sets (images -> backbone (logits))
        row = {"task": task}
        test_accs = []
        for name, dataset in test_domains.items():
            test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            loss, acc = evaluate_on_images(backbone, test_loader, criterion, device, test_name=f"step{step}_{name}", save_dir="./logs_finetuning")
            row[name + '_acc'] = acc
            test_accs.append(acc if not np.isnan(acc) else 0.0)

        mean_acc = float(np.mean(test_accs)) if len(test_accs) > 0 else 0.0
        best_ckpt = os.path.join(checkpoint_dir, f'ft_step{step}_{task}_best.pth')
        torch.save({'backbone_state': backbone.state_dict(), 'args': vars(args), 'task': task}, best_ckpt)
        prev_ckpt = best_ckpt
        logger.info(f"Saved best checkpoint for task {task}: {best_ckpt} (mean_test_acc={mean_acc:.4f})")

        # If using LwF, create a frozen copy of current model to use as teacher for next task
        prev_model = copy.deepcopy(backbone).eval()
        if args.teacher_on_cpu:
            prev_model.to('cpu')
        for p in prev_model.parameters():
            p.requires_grad = False

        results.append(row)
        df = pd.DataFrame(results)
        df.to_csv(os.path.join("./logs_lwf", "sequential_lwf_results.csv"), index=False)

    logger.info("All tasks finished")
    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--initial_backbone', type=str, default='stylegan1', choices=list(PRETRAINED_MODELS.keys()))
    parser.add_argument('--tasks', type=str, default='stylegan1,stylegan2,sdv1_4,stylegan3,stylegan_xl')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs_per_task', type=int, default=10)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--classifier_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--scheduler_step', type=int, default=5)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--use_amp', action='store_true')

    # LwF args
    parser.add_argument('--lwf_alpha', type=float, default=0.5, help='Weight for distillation loss (0..1)')
    parser.add_argument('--lwf_temp', type=float, default=2.0, help='Temperature for distillation')

    # New flags from patch
    parser.add_argument('--use_balanced_sampler', action='store_true', default=False, help='Use BalancedBatchSampler for training loader')
    parser.add_argument('--sampler_oversample', action='store_true', help='BalancedBatchSampler oversample smaller classes')
    parser.add_argument('--sampler_replacement', action='store_true', help='BalancedBatchSampler sampling with replacement when oversampling')
    parser.add_argument('--teacher_on_cpu', action='store_true', default=True, help='Keep previous-task teacher on CPU to save GPU memory')
    parser.add_argument('--max_train_samples', type=int, default=0, help='If >0, use at most this many training samples per task (random subset)')
    parser.add_argument('--deterministic', action='store_true', default=False, help='Set cudnn.deterministic True (slow) for reproducibility')

    args = parser.parse_args()

    results = train_and_eval(args)
    print(results)
