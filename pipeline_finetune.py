#!/usr/bin/env python3
"""
End-to-end sequential fine-tuning (images-only) — no separate head.

Features:
 - Always fine-tunes the backbone (backbone must output 2-class logits).
 - --max_train_samples: limit samples per task.
 - --balanced_subset: when used with --max_train_samples, create a balanced
   subset with (approximately) equal samples per class.
 - --balanced_batch_sampler: use a BalancedBatchSampler to produce class-balanced batches.
 - --oversample: when using balanced_batch_sampler, allow oversampling of smaller classes.

Usage examples:
  python sequential_finetune_no_features.py --max_train_samples 100 --balanced_subset
  python sequential_finetune_no_features.py --balanced_batch_sampler --oversample
"""

import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
# change visible devices if needed
#s.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import math
from typing import Sequence, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Sampler
import pandas as pd
from tqdm import tqdm
import logging

# project imports (adjust to your repo)
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model

logger = logging.getLogger(__name__)


class BalancedBatchSampler(Sampler[List[int]]):
    """
    Balanced batch sampler.

    - labels: 1D sequence of integer class labels.
    - batch_size will be adjusted to be a multiple of number of classes (if needed).
    - oversample: if True, up-sample smaller classes so we can produce as many
      batches as the largest class supports (uses ceil to ensure coverage).
    - replacement: when oversampling, whether to sample with replacement when needed.
    - seed: RNG seed for reproducibility.
    - drop_last: whether to drop a final incomplete batch (default True).
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
        if isinstance(labels, torch.Tensor):
            self.labels = labels.cpu().long().numpy()
        else:
            self.labels = np.asarray(labels, dtype=np.int64)

        if self.labels.size == 0:
            raise ValueError("BalancedBatchSampler requires non-empty labels array.")

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

        # Ensure batch size at least num_classes
        if self.batch_size < self.num_classes:
            self.batch_size = self.num_classes

        # Force batch_size to be multiple of num_classes by rounding DOWN (then warn)
        if self.batch_size % self.num_classes != 0:
            new_bs = (self.batch_size // self.num_classes) * self.num_classes
            if new_bs == 0:
                new_bs = self.num_classes
            if new_bs != self.batch_size:
                print(f"[BalancedBatchSampler] Warning: batch_size {self.batch_size} not divisible by num_classes {self.num_classes}. "
                      f"Adjusting batch_size -> {new_bs}")
                self.batch_size = new_bs

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

        # compute num_batches:
        # - if oversample: allow as many batches as *ceil(max_class_len / per_class)*
        # - else: limit by smallest class (floor)
        if self.per_class == 0:
            self.num_batches = 0
        else:
            if self.oversample:
                self.num_batches = int(math.ceil(self.max_class_len / float(self.per_class)))
            else:
                self.num_batches = int(self.min_class_len // self.per_class)

        # RNG generator for reproducibility (use numpy RandomState for repeatability)
        self.generator = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    def __len__(self) -> int:
        return int(self.num_batches)

    def __iter__(self):
        if self.num_batches == 0:
            return
            yield  # make generator

        per_class_block = self.num_batches * self.per_class
        class_rows = {}  # cls -> ndarray (num_batches, per_class)

        for cls in self.classes:
            idxs = self.class_to_indices[int(cls)]
            n = len(idxs)
            if self.oversample:
                # need per_class_block items
                if self.replacement:
                    # sample indices with replacement
                    choices = self.generator.randint(0, n, size=(per_class_block,))
                    selected = idxs[choices]
                else:
                    # shuffle and tile cycles until we reach required length (no replacement)
                    reps = int(math.ceil(per_class_block / n))
                    permuted_blocks = []
                    for r in range(reps):
                        permuted_blocks.append(self.generator.permutation(idxs))
                    permuted = np.concatenate(permuted_blocks)[:per_class_block]
                    selected = permuted
            else:
                # do not oversample: shuffle and cut (no replacement)
                if self.shuffle:
                    perm = self.generator.permutation(idxs)
                else:
                    perm = idxs.copy()
                if perm.shape[0] < per_class_block:
                    # tile to reach required length (fallback) — shouldn't usually happen because num_batches was computed from min class len
                    reps = int(math.ceil(per_class_block / perm.shape[0]))
                    perm = np.tile(perm, reps)[:per_class_block]
                selected = perm[:per_class_block]

            # reshape to (num_batches, per_class)
            selected = np.asarray(selected, dtype=np.int64).reshape(self.num_batches, self.per_class)
            class_rows[int(cls)] = selected

        # Now yield batches: for each batch i concat class_rows[:, i]
        for batch_idx in range(self.num_batches):
            parts = [class_rows[int(cls)][batch_idx] for cls in self.classes]
            batch = np.concatenate(parts, axis=0).tolist()
            if self.shuffle:
                # use generator.shuffle (in-place)
                self.generator.shuffle(batch)
            yield batch

        if self.verbose:
            used_counts = {int(cls): int(min(self.num_batches * self.per_class, len(self.class_to_indices[int(cls)]))) for cls in self.classes}
            logger.info(f"[BalancedBatchSampler] num_batches={self.num_batches}, per_class={self.per_class}, samples_used_per_class(approx)={used_counts}")


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_on_images(backbone, test_loader, criterion, device, test_name="test", save_dir="./logs"):
    """Run backbone on images from test_loader and save per-sample CSV + summary."""
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

            outputs = backbone(imgs)   # backbone must produce logits
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)

            probs_cpu = probs.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            batch_size = labels_cpu.shape[0]
            num_probs = probs_cpu.shape[1] if probs_cpu.ndim == 2 else 0
            for i in range(batch_size):
                rows.append({
                    "idx": sample_idx,
                    "label": int(labels_cpu[i]),
                    "pred": int(preds_cpu[i]),
                    "prob_real": float(probs_cpu[i, 0]) if num_probs >= 1 else 0.0,
                    "prob_fake": float(probs_cpu[i, 1]) if num_probs >= 2 else 0.0
                })
                sample_idx += 1

    avg_loss = float(np.mean(losses)) if len(losses) > 0 else float('nan')
    preds_all = np.array([r['pred'] for r in rows], dtype=int) if len(rows) > 0 else np.array([], dtype=int)
    labels_all = np.array([r['label'] for r in rows], dtype=int) if len(rows) > 0 else np.array([], dtype=int)
    accuracy = float((preds_all == labels_all).sum() / labels_all.shape[0]) if labels_all.size > 0 else float('nan')

    csv_out = os.path.join(save_dir, f"{test_name}_predictions.csv")
    df_preds = pd.DataFrame(rows, columns=["idx", "label", "pred", "prob_real", "prob_fake"]) if len(rows) > 0 else pd.DataFrame(columns=["idx", "label", "pred", "prob_real", "prob_fake"]) 
    df_preds.to_csv(csv_out, index=False)

    summary_out = os.path.join(save_dir, f"{test_name}_summary.txt")
    with open(summary_out, "w") as f:
        f.write(f"test_name: {test_name}\n")
        f.write(f"num_samples: {len(labels_all)}\n")
        f.write(f"avg_loss: {avg_loss:.6f}\n")
        f.write(f"accuracy: {accuracy:.6f}\n")

    print(f"[evaluate] Saved predictions CSV to {csv_out} and summary to {summary_out}")
    return avg_loss, accuracy


def build_balanced_subset_indices(labels_array: np.ndarray, total_samples: int, rng: np.random.RandomState):
    """
    Build an index list of length total_samples that is balanced across classes.
    If a class has fewer examples than required, sample with replacement for that class.
    Remainder (when total_samples % num_classes != 0) is distributed one-by-one across classes.
    """
    classes = np.sort(np.unique(labels_array)).tolist()
    num_classes = len(classes)
    if num_classes == 0:
        raise ValueError("No classes found in labels_array when building balanced subset.")

    base = total_samples // num_classes
    remainder = total_samples - base * num_classes

    chosen_indices = []
    for i, cls in enumerate(classes):
        cls_idxs = np.where(labels_array == cls)[0]
        req = base + (1 if i < remainder else 0)
        if len(cls_idxs) == 0:
            raise ValueError(f"Class {cls} has no samples, cannot build balanced subset.")
        replace = req > len(cls_idxs)
        picked = rng.choice(cls_idxs, size=req, replace=replace)
        chosen_indices.append(picked)
    chosen_indices = np.concatenate(chosen_indices)
    rng.shuffle(chosen_indices)
    return chosen_indices.tolist()


def get_labels_array_from_dataset(dataset) -> np.ndarray:
    """
    Try common dataset attributes for labels (fast). If not present, fallback to iterating.
    Returns an int64 numpy array of labels.
    """
    # Common attr names
    for attr in ("targets", "labels", "y", "label_array", "labels_array"):
        if hasattr(dataset, attr):
            arr = getattr(dataset, attr)
            if isinstance(arr, (list, np.ndarray)):
                return np.asarray(arr, dtype=np.int64)
            if isinstance(arr, torch.Tensor):
                return arr.cpu().numpy().astype(np.int64)

    # If dataset is a Subset, try to access underlying dataset's attribute if available
    if isinstance(dataset, Subset):
        inner = dataset.dataset
        if hasattr(inner, "targets"):
            arr = getattr(inner, "targets")
            indices = dataset.indices if hasattr(dataset, "indices") else None
            arr_np = np.asarray(arr, dtype=np.int64)
            if indices is not None:
                return arr_np[np.asarray(indices, dtype=np.int64)]
            return arr_np

    # Fallback slow path: iterate dataset once (still deterministic since seed is set)
    n = len(dataset)
    labels = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        item = dataset[i]
        if isinstance(item, tuple) or isinstance(item, list):
            labels[i] = int(item[1])
        else:
            # if dataset returns dict
            labels[i] = int(item.get("label", 0))
    return labels


def train_and_eval(args):
    set_global_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) == 0:
        raise ValueError("No tasks specified. Provide --tasks like 'stylegan1,stylegan2,sdv1_4,stylegan3,stylegan_xl'")

    # load backbone (must output logits directly)
    backbone = load_pretrained_model(PRETRAINED_MODELS[args.initial_backbone])
    backbone.to(device)

    checkpoint_dir = "./checkpoint_finetune"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("./logs_finetune", exist_ok=True)

    results = []
    prev_ckpt = None

    rng = np.random.RandomState(args.seed)

    for step, task in enumerate(tasks, start=1):
        print('=' * 80)
        print(f"Task {step}/{len(tasks)}: train on {task}")

        # prepare full dataset
        train_dataset = RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR[task], split='train_set', num_training_samples=args.max_train_samples)
        total_train = len(train_dataset)
        print(f"Full dataset size: {total_train}")

        # Option A: build a balanced subset (exactly max_train_samples, balanced across classes)
        dataset_for_loader = train_dataset
        if args.max_train_samples > 0 and args.balanced_subset:
            # build labels array
            labels_array = get_labels_array_from_dataset(train_dataset)
            chosen_idx = build_balanced_subset_indices(labels_array, args.max_train_samples, rng)
            train_subset = Subset(train_dataset, chosen_idx)
            dataset_for_loader = train_subset
            print(f"Using balanced subset of {len(train_subset)} samples (balanced_subset).")
        else:
            if args.max_train_samples > 0 and not args.balanced_subset:
                # simple random subset
                if args.max_train_samples < total_train:
                    indices = list(range(total_train))
                    rng.shuffle(indices)
                    subset_idx = indices[:args.max_train_samples]
                    train_subset = Subset(train_dataset, subset_idx)
                    dataset_for_loader = train_subset
                    print(f"Using random subset of {len(train_subset)} samples (max_train_samples).")
                else:
                    print("max_train_samples >= dataset size, using full dataset.")

        # Decide DataLoader creation
        effective_batch_size = args.batch_size
        if args.balanced_batch_sampler:
            labels_array = get_labels_array_from_dataset(dataset_for_loader)
            num_classes = len(np.unique(labels_array))
            if num_classes == 0:
                raise ValueError("No classes found in training labels for balanced_batch_sampler.")

            # ensure batch_size divisible by num_classes; BalancedBatchSampler will adjust if needed
            if args.batch_size % num_classes != 0:
                adjusted_bs = (args.batch_size // num_classes) * num_classes
                if adjusted_bs == 0:
                    adjusted_bs = num_classes
                print(f"[train] Warning: batch_size {args.batch_size} not divisible by num_classes {num_classes}. Adjusting batch_size -> {adjusted_bs}")
                effective_batch_size = adjusted_bs

            sampler = BalancedBatchSampler(labels_array,
                                           batch_size=effective_batch_size,
                                           oversample=args.oversample,
                                           shuffle=True,
                                           replacement=args.replacement,
                                           seed=args.seed,
                                           drop_last=True,
                                           verbose=False)

            pin_memory = True if ("cuda" in str(device).lower()) else False
            train_loader = DataLoader(dataset_for_loader, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=pin_memory)
            print(f"Using BalancedBatchSampler: batch_size={effective_batch_size}, oversample={args.oversample}")
        else:
            pin_memory = True if ("cuda" in str(device).lower()) else False
            train_loader = DataLoader(dataset_for_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
            print(f"Using standard DataLoader: batch_size={args.batch_size}, samples={len(dataset_for_loader)}")

        # Always fine-tune full backbone (no head)
        print("Fine-tuning full backbone parameters")
        for p in backbone.parameters():
            p.requires_grad = True

        backbone_params = [p for p in backbone.parameters() if p.requires_grad]
        if len(backbone_params) == 0:
            raise RuntimeError("No parameters to optimize. Check backbone implementation.")

        optimizer = torch.optim.Adam(backbone_params, lr=args.backbone_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma) if args.use_scheduler else None
        criterion = nn.CrossEntropyLoss().to(device)
        scaler = torch.cuda.amp.GradScaler() if (args.use_amp and torch.cuda.is_available()) else None

        # optionally resume previous backbone checkpoint (backbone_state)
        if prev_ckpt is not None:
            try:
                ck = torch.load(prev_ckpt, map_location='cpu')
                if 'backbone_state' in ck:
                    backbone.load_state_dict(ck['backbone_state'], strict=False)
                print(f"Loaded previous checkpoint {prev_ckpt}")
            except Exception as e:
                print(f"[warning] couldn't load prev ckpt: {e}")

        # training loop
        n_samples_in_loader = len(train_loader.dataset) if not args.balanced_batch_sampler else f"{len(train_loader)} batches x {effective_batch_size}"
        print(f"Starting training for {args.epochs_per_task} epochs on {n_samples_in_loader} samples (loader)")
        start_time = time.time()
        for epoch in range(args.epochs_per_task):
            backbone.train()
            losses = []
            correct = 0
            total = 0

            for imgs, labels in tqdm(train_loader, desc=f"Train {task} (ep {epoch+1})", leave=False):
                imgs = imgs.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = backbone(imgs)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = backbone(imgs)
                    loss = criterion(outputs, labels)
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
            if (epoch + 1) % max(1, args.log_every) == 0:
                print(f"Epoch [{epoch+1}/{args.epochs_per_task}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        elapsed = time.time() - start_time
        print(f"Finished task {task} training in {elapsed/60:.2f} minutes")

        # save last checkpoint (backbone + optimizer + RNG states)
        last_ckpt = os.path.join(checkpoint_dir, f'ft_step{step}_{task}_last.pth')
        torch.save({
            'backbone_state': backbone.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
            'rng_state': rng.get_state(),
            'np_random_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, last_ckpt)
        print(f"Saved last checkpoint: {last_ckpt}")

        # evaluation on test sets
        test_domains = {
            "real_vs_stylegan1": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan1'], split='test_set', num_training_samples=args.max_train_samples),
            "real_vs_stylegan2": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan2'], split='test_set', num_training_samples=args.max_train_samples),
            "real_vs_sdv1_4": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['sdv1_4'], split='test_set', num_training_samples=args.max_train_samples),
            "real_vs_stylegan3": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan3'], split='test_set', num_training_samples=args.max_train_samples),
            "real_vs_styleganxl": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan_xl'], split='test_set', num_training_samples=args.max_train_samples),
            "real_vs_sdv2_1": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['sdv2_1'], split='test_set', num_training_samples=args.max_train_samples),
        }

        row = {"task": task}
        test_accs = []
        for name, dataset in test_domains.items():
            test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            loss, acc = evaluate_on_images(backbone, test_loader, criterion, device, test_name=f"step{step}_{name}", save_dir="./logs_finetuning")
            row[name + '_acc'] = acc
            test_accs.append(acc if not np.isnan(acc) else 0.0)

        mean_acc = float(np.mean(test_accs)) if len(test_accs) > 0 else 0.0
        best_ckpt = os.path.join(checkpoint_dir, f'ft_step{step}_{task}_best.pth')
        torch.save({
            'backbone_state': backbone.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        }, best_ckpt)
        prev_ckpt = best_ckpt
        print(f"Saved best checkpoint for task {task}: {best_ckpt} (mean_test_acc={mean_acc:.4f})")

        results.append(row)
        df = pd.DataFrame(results)
        df.to_csv(os.path.join("./logs_finetuning", "sequential_finetune_results.csv"), index=False)

    print("All tasks finished")
    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--initial_backbone', type=str, default='stylegan1', choices=list(PRETRAINED_MODELS.keys()))
    parser.add_argument('--tasks', type=str, default='stylegan1,stylegan2,sdv1_4,stylegan3,stylegan_xl,sdv2_1',)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs_per_task', type=int, default=1)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--scheduler_step', type=int, default=5)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--max_train_samples', type=int, default=100,
                        help='Limit the number of training samples per task (-1 for all)')
    parser.add_argument('--balanced_subset', action='store_true', 
                        help='When used with --max_train_samples, create an exactly balanced subset across classes')
    parser.add_argument('--balanced_batch_sampler', action='store_true', default=False,
                        help='Use BalancedBatchSampler to produce class-balanced batches from the dataset/subset')
    parser.add_argument('--oversample', action='store_true',
                        help='When using balanced_batch_sampler, allow oversampling of smaller classes to match largest class')
    parser.add_argument('--replacement', action='store_true',
                        help='When oversampling, sample with replacement where needed')
    args = parser.parse_args()
    orders = [
        "stylegan1, stylegan2, sdv1_4, stylegan3, stylegan_xl, sdv2_1",
        "stylegan1, stylegan2, stylegan3, stylegan_xl, sdv1_4, sdv2_1",
        "sdv1_4, sdv2_1, stylegan1, stylegan2, stylegan3, stylegan_xl",
        #random order from stylegan2
        "stylegan2, stylegan3,  sdv2_1,stylegan1,stylegan_xl, sdv1_4"
    ]
    for o in orders:
        args.tasks = o
        results = train_and_eval(args)
        print(results)

        order_str = o.replace(" ", "").replace(",", "_")
        path = f"logs_finetune/new_sequential_fd_results_{order_str}.csv"

        # scrive header solo se il file NON esiste
        write_header = not os.path.exists(path)

        results.to_csv(
            path,
            mode="a",          # append
            header=write_header,
            index=False
        )
