#!/usr/bin/env python3
"""
End-to-end sequential fine-tuning (images-only)

- Trains the backbone end-to-end by default.
- If --train_fc is provided, freezes the backbone and trains only the final linear head.
- No feature extraction or anchor code anywhere.
- Evaluation runs images through the current backbone+head (no saved features).

Usage examples:
  # fine-tune entire backbone
  python sequential_finetune_no_features.py --finetune --epochs_per_task 5

  # train only the final FC head (backbone frozen)
  python sequential_finetune_no_features.py --train_fc --epochs_per_task 5

"""

import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm

# project imports (same as your repo)
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model
from src.utils import (get_device)

from typing import Sequence, Optional, List
import torch
from torch.utils.data import Sampler
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

class BalancedBatchSampler(Sampler[List[int]]):
    """
    Balanced batch sampler.

    Behavior:
      - Divides batch_size evenly across classes: per_class = batch_size // num_classes.
        (Requires batch_size % num_classes == 0.)
      - Two strategies for number of batches per epoch:
          * oversample=False (default): num_batches = min_class_len // per_class
            (ensures no sample from any class is reused)
          * oversample=True: num_batches = max_class_len // per_class
            (smaller classes are up-sampled / repeated to reach that many batches)
      - shuffle: whether to shuffle class indices each epoch (default True).
      - replacement: if True, sampling for upsampling uses replacement (fast).
                     If False and oversample=True, the sampler cycles through
                     shuffled indices and pads by sampling without replacement where needed.
      - seed: optional int seed to make per-epoch sampling reproducible. If None,
              a generator with non-deterministic seed is used.

    Args:
        labels: 1D sequence or 1D torch.Tensor of integer class labels (CPU or any).
        batch_size: integer, must be a multiple of number of unique classes.
        oversample: If True, up-sample small classes to allow more batches (default False).
        shuffle: shuffle class indices per epoch (default True).
        replacement: use sampling with replacement when padding/oversampling (default False).
        seed: optional integer seed for reproducibility.
        drop_last: if True, drop trailing samples that don't form a full balanced batch (default True).
        verbose: if True, will log an informational summary each epoch.
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
            raise ValueError("batch_size must be a multiple of number of classes")

        self.per_class = self.batch_size // self.num_classes

        # build per-class index arrays (numpy ints)
        self.class_to_indices = {}
        for cls in self.classes:
            idxs = np.nonzero(self.labels == cls)[0].astype(np.int64)
            if idxs.size == 0:
                raise ValueError(f"Class {cls} has no samples.")
            self.class_to_indices[int(cls)] = idxs

        # compute num_batches for epoch depending on oversample/drop_last
        lens = [len(v) for v in self.class_to_indices.values()]
        self.min_class_len = min(lens)
        self.max_class_len = max(lens)

        if self.oversample:
            # allow as many batches as the largest class can provide
            self.num_batches = self.max_class_len // self.per_class
        else:
            # limit by the smallest class to avoid reuse
            self.num_batches = self.min_class_len // self.per_class

        # If drop_last is False and no oversampling, we can still produce one extra partial batch per class
        if not self.drop_last and not self.oversample:
            # floor division gives full batches; if some leftover exists, we create one more batch by padding (not balanced) -> we avoid this complexity for safety
            # So keep behavior consistent: we still use num_batches as above (drop partial)
            pass

        # RNG generator for reproducibility
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(int(seed))
        else:
            # leave unseeded (non-deterministic)
            self.generator = torch.Generator()

    def __len__(self) -> int:
        return int(self.num_batches)

    def __iter__(self):
        """
        Build per-class selection matrices of shape (num_batches, per_class) for each class
        and then yield balanced batches by concatenating corresponding rows across classes.
        """
        # For each class produce `num_batches * per_class` indices (possibly shuffling / repeating)
        per_class_block = self.num_batches * self.per_class
        class_rows = {}  # cls -> ndarray (num_batches, per_class)

        for cls in self.classes:
            idxs = self.class_to_indices[int(cls)]
            n = len(idxs)

            if self.oversample:
                # need to produce per_class_block indices; allow repeats
                if self.replacement:
                    # sample with replacement directly (fast)
                    choices = torch.randint(low=0, high=n, size=(per_class_block,), generator=self.generator).numpy()
                    selected = idxs[choices]
                else:
                    # shuffle and tile cycles until we reach required length
                    reps = math.ceil(per_class_block / n)
                    # produce reps permutations concatenated
                    permuted = np.concatenate([np.random.permutation(idxs) for _ in range(reps)])[:per_class_block]
                    selected = permuted
            else:
                # do not oversample: only use available samples (shuffle then cut)
                if self.shuffle:
                    perm = np.random.permutation(idxs)
                else:
                    perm = idxs.copy()
                # take exactly per_class_block or as many available (it should be >= per_class_block by num_batches definition)
                if perm.shape[0] < per_class_block:
                    # Fallback: if not enough (shouldn't happen when num_batches computed properly), pad by repeating
                    reps = math.ceil(per_class_block / perm.shape[0])
                    perm = np.tile(perm, reps)[:per_class_block]
                selected = perm[:per_class_block]

            # reshape to (num_batches, per_class)
            selected = np.asarray(selected, dtype=np.int64).reshape(self.num_batches, self.per_class)
            class_rows[int(cls)] = selected

        # Now yield batches: for each batch i concat class_rows[:, i]
        for batch_idx in range(self.num_batches):
            parts = [class_rows[int(cls)][batch_idx] for cls in self.classes]
            # interleave classes or simply concatenate? We concatenate per-class blocks for simplicity.
            batch = np.concatenate(parts, axis=0).tolist()
            if self.shuffle:
                # shuffle within the batch so order not grouped by class
                np.random.shuffle(batch)
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


class LinearHead(nn.Module):
    def __init__(self, feat_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def infer_feat_dim(backbone, device, sample_shape=(1, 3, 224, 224)):
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(sample_shape).to(device)
        feats = backbone(dummy)
        if feats.ndim == 1:
            return int(feats.shape[0])
        return int(feats.shape[1])


def evaluate_on_images(backbone, test_loader, criterion, device, test_name="test", save_dir="./logs"):
    """Run backbone+head on images from test_loader and save per-sample CSV + summary."""
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
        f.write(f"test_name: {test_name}\n"),
        f.write(f"num_samples: {len(labels_all)}\n"),
        f.write(f"avg_loss: {avg_loss:.6f}\n"),
        f.write(f"accuracy: {accuracy:.6f}\n")

    print(f"[evaluate] Saved predictions CSV to {csv_out} and summary to {summary_out}")
    return avg_loss, accuracy


def train_and_eval(args):
    set_global_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) == 0:
        raise ValueError("No tasks specified. Provide --tasks like 'stylegan1,stylegan_xl,sdv1_4'")

    # load backbone and remove final classifier head if present
    backbone = load_pretrained_model(PRETRAINED_MODELS[args.initial_backbone])
    if hasattr(backbone, "resnet") and hasattr(backbone.resnet, "fc"):
        backbone.resnet.fc = nn.Identity()
    backbone.to(device)

    checkpoint_dir = "./checkpoint_finetune"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("./logs_finetuning", exist_ok=True)

    test_names = [
        "real_vs_stylegan1",
        "real_vs_stylegan2",
        "real_vs_styleganxl",
        "real_vs_sdv1_4"
    ]
    results = []
    prev_ckpt = None

    for step, task in enumerate(tasks, start=1):
        print('' + '='*60)
        print(f"Task {step}/{len(tasks)}: train on {task}")

        # prepare datasets/loaders
        
        train_dataset = RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR[task])
        #labels_array = np.array([train_dataset[i][1] for i in range(len(train_dataset))], dtype=np.int64)
        #print('Labels array prepared')
        #try:
            #sampler = BalancedBatchSampler(labels_array, batch_size=args.batch_size)
            #train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=args.num_workers)
            #print(f"Using BalancedBatchSampler for training with batch size {args.batch_size}")
        #except Exception:
            #print("[warning] BalancedBatchSampler failed, using standard DataLoader instead")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        print(f"Using standard DataLoader for training with batch size {args.batch_size}")




        if args.train_fc:
            # only train last layer
            print("--train_fc set: freezing backbone, training only final FC head")
            for p in backbone.parameters():
                p.requires_grad = False
            for p in head.parameters():
                p.requires_grad = True
        else:
            # train whole model
            print("Fine-tuning full backbone")
            for p in backbone.parameters():
                p.requires_grad = True

    

        # optimizer: separate groups
        backbone_params = [p for p in backbone.parameters() if p.requires_grad]
        param_groups = []
        if len(backbone_params) > 0:
            param_groups.append({'params': backbone_params, 'lr': args.backbone_lr})
        optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma) if args.use_scheduler else None
        criterion = nn.CrossEntropyLoss().to(device)
        scaler = torch.cuda.amp.GradScaler() if (args.use_amp and torch.cuda.is_available()) else None

        # optionally resume
        if prev_ckpt is not None:
            try:
                ck = torch.load(prev_ckpt, map_location='cpu')
                if 'backbone_state' in ck:
                    backbone.load_state_dict(ck['backbone_state'], strict=False)

                print(f"Loaded previous checkpoint {prev_ckpt}")
            except Exception as e:
                print(f"[warning] couldn't load prev ckpt: {e}")

        # training loop
        print(f"Starting training for {args.epochs_per_task} epochs on {len(train_dataset)} samples")
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
                        feats = backbone(imgs)
                        outputs = head(feats)
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
            if (epoch+1) % max(1, args.log_every) == 0:
                print(f"Epoch [{epoch+1}/{args.epochs_per_task}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        elapsed = time.time() - start_time
        print(f"Finished task {task} training in {elapsed/60:.2f} minutes")

        # save last checkpoint
        last_ckpt = os.path.join(checkpoint_dir, f'ft_step{step}_{task}_last.pth')
        torch.save({'state_dict': backbone.state_dict()}, last_ckpt)

        print(f"Saved last checkpoint: {last_ckpt}")

        # evaluation on test sets (images -> backbone -> head)
        test_domains = {
            "real_vs_stylegan1": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan1'], split='test_set'),
            "real_vs_stylegan2": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan2'], split='test_set'),
            "real_vs_styleganxl": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan_xl'], split='test_set'),
            "real_vs_sdv1_4": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['sdv1_4'], split='test_set')
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
        torch.save({'backbone_state': backbone.state_dict()}, best_ckpt)
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
    parser.add_argument('--tasks', type=str, default='stylegan1,stylegan2,stylegan_xl,sdv1_4')
    parser.add_argument('--train_fc', action='store_true', help='If set, freeze backbone and train only final FC head')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs_per_task', type=int, default=10)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--classifier_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--scheduler_step', type=int, default=5)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--use_amp', action='store_true')
    args = parser.parse_args()

    results = train_and_eval(args)
    print(results)
