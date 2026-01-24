#!/usr/bin/env python3
"""
Sequential fine-tuning with built-in feature distillation (teacher = previous-step model).

Simplifications you requested:
 - Distillation is always enabled (no --distill flag).
 - Teacher is always taken from previous-step checkpoint (no --teacher_from_prev flag).
 - Feature normalization for distillation is always ON (no --distill_feat_norm flag).
 - If helper functions are identical to the baseline, we try to import them from
   `sequential_finetune_no_features` (your original file). If that import fails we
   fall back to local definitions to remain robust.

Loss used per minibatch:
    L = CE(student, y) + distill_lambda * || normalize(fs) - normalize(ft) ||^2

"""

import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import math
from typing import Sequence, Optional, List, Tuple, Dict

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


from pipeline_finetune import (
        BalancedBatchSampler,
        set_global_seed,
        evaluate_on_images,
        build_balanced_subset_indices,
        get_labels_array_from_dataset,
    )
# Distillation helpers (small, local)

def find_final_linear_module(model: torch.nn.Module) -> Tuple[Optional[str], Optional[torch.nn.Module]]:
    final_name = None
    final_mod = None
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            final_name = name
            final_mod = mod
    return final_name, final_mod


def attach_feature_hook(model: torch.nn.Module) -> Tuple[Optional[torch.utils.hooks.RemovableHandle], Dict[str, torch.Tensor]]:
    feat_dict: Dict[str, torch.Tensor] = {}
    name, final_mod = find_final_linear_module(model)
    if final_mod is None:
        return None, feat_dict
    def _hook(module, input, output):
        feat = input[0]
        feat_flat = feat.view(feat.size(0), -1) if feat.dim() > 2 else feat
        feat_dict['feat'] = feat_flat
    handle = final_mod.register_forward_hook(_hook)
    return handle, feat_dict


def train_and_eval(args):
    set_global_seed(args.seed)
    device = torch.device('cuda')
    print(f"Using device: {device}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) == 0:
        raise ValueError("No tasks specified.")

    # load initial backbone
    backbone = load_pretrained_model(PRETRAINED_MODELS[args.initial_backbone])
    backbone.to(device)

    checkpoint_dir = "./checkpoint_fd"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("./logs_fd", exist_ok=True)

    results = []
    prev_ckpt = None

    rng = np.random.RandomState(args.seed)

    # distillation containers
    teacher = None
    teacher_handle = None
    teacher_feat = {}
    student_handle = None
    student_feat = {}

    for step, task in enumerate(tasks, start=1):
        print('=' * 80)
        print(f"Task {step}/{len(tasks)}: train on {task}")

        train_dataset = RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR[task])
        total_train = len(train_dataset)
        print(f"Full dataset size: {total_train}")

        dataset_for_loader = train_dataset
        if args.max_train_samples > 0 and args.balanced_subset:
            labels_array = get_labels_array_from_dataset(train_dataset)
            chosen_idx = build_balanced_subset_indices(labels_array, args.max_train_samples, rng)
            train_subset = Subset(train_dataset, chosen_idx)
            dataset_for_loader = train_subset
            print(f"Using balanced subset of {len(train_subset)} samples (balanced_subset).")
        else:
            if args.max_train_samples > 0 and not args.balanced_subset:
                if args.max_train_samples < total_train:
                    indices = list(range(total_train))
                    rng.shuffle(indices)
                    subset_idx = indices[:args.max_train_samples]
                    train_subset = Subset(train_dataset, subset_idx)
                    dataset_for_loader = train_subset
                    print(f"Using random subset of {len(train_subset)} samples (max_train_samples).")
                else:
                    print("max_train_samples >= dataset size, using full dataset.")

        effective_batch_size = args.batch_size
        if args.balanced_batch_sampler:
            labels_array = get_labels_array_from_dataset(dataset_for_loader)
            num_classes = len(np.unique(labels_array))
            if num_classes == 0:
                raise ValueError("No classes found in training labels for balanced_batch_sampler.")
            if args.batch_size % num_classes != 0:
                adjusted_bs = (args.batch_size // num_classes) * num_classes
                if adjusted_bs == 0:
                    adjusted_bs = num_classes
                print(f"[train] Warning: batch_size {args.batch_size} not divisible by num_classes {num_classes}. Adjusting batch_size -> {adjusted_bs}")
                effective_batch_size = adjusted_bs
            sampler = BalancedBatchSampler(labels_array, batch_size=effective_batch_size, oversample=args.oversample, shuffle=True, replacement=args.replacement, seed=args.seed, drop_last=True, verbose=False)
            pin_memory = True if ("cuda" in str(device).lower()) else False
            train_loader = DataLoader(dataset_for_loader, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=pin_memory)
            print(f"Using BalancedBatchSampler: batch_size={effective_batch_size}, oversample={args.oversample}")
        else:
            pin_memory = True if ("cuda" in str(device).lower()) else False
            train_loader = DataLoader(dataset_for_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
            print(f"Using standard DataLoader: batch_size={args.batch_size}, samples={len(dataset_for_loader)}")

        # fine-tune full backbone
        for p in backbone.parameters():
            p.requires_grad = True
        backbone_params = [p for p in backbone.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(backbone_params, lr=args.backbone_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma) if args.use_scheduler else None
        criterion = nn.CrossEntropyLoss().to(device)
        scaler = torch.cuda.amp.GradScaler() if (args.use_amp and torch.cuda.is_available()) else None

        # resume student from prev_ckpt if available (sequential fine-tuning)
        if prev_ckpt is not None:
            try:
                ck = torch.load(prev_ckpt, map_location='cpu')
                if 'backbone_state' in ck:
                    backbone.load_state_dict(ck['backbone_state'], strict=False)
                print(f"Loaded previous checkpoint {prev_ckpt} (student resumed)")
            except Exception as e:
                print(f"[warning] couldn't load prev ckpt for student: {e}")

        # ------------------ Teacher setup (ALWAYS: teacher = prev_ckpt if available) ------------------
        # remove old teacher hooks
        if teacher_handle is not None:
            try:
                teacher_handle.remove()
            except Exception:
                pass
            teacher_handle = None
            teacher = None
            teacher_feat = {}

        if prev_ckpt is not None:
            try:
                print(f"[distill] Using prev_ckpt as teacher: {prev_ckpt}")
                teacher = load_pretrained_model(PRETRAINED_MODELS[args.initial_backbone])
                ck = torch.load(prev_ckpt, map_location='cpu')
                if 'backbone_state' in ck:
                    teacher.load_state_dict(ck['backbone_state'], strict=False)
                else:
                    teacher.load_state_dict(ck, strict=False)
                teacher.to(device)
                teacher.eval()
                for p in teacher.parameters():
                    p.requires_grad = False
                teacher_handle, teacher_feat = attach_feature_hook(teacher)
                if teacher_handle is None:
                    print("[distill] Warning: couldn't attach feature hook to teacher; falling back to logits for distillation.")
            except Exception as e:
                print(f"[distill] Failed to load teacher from prev_ckpt: {e}")
                teacher = None
                teacher_handle = None
                teacher_feat = {}
        else:
            # no prev_ckpt available: optional fallback teacher_backbone
            if args.teacher_backbone is not None:
                print(f"[distill] No prev_ckpt, loading fallback teacher backbone {args.teacher_backbone}")
                teacher = load_pretrained_model(PRETRAINED_MODELS[args.teacher_backbone])
                teacher.to(device)
                teacher.eval()
                for p in teacher.parameters():
                    p.requires_grad = False
                teacher_handle, teacher_feat = attach_feature_hook(teacher)
                if teacher_handle is None:
                    print("[distill] Warning: couldn't attach feature hook to fallback teacher; falling back to logits for distillation.")

        # attach hook to student
        if student_handle is not None:
            try:
                student_handle.remove()
            except Exception:
                pass
            student_handle = None
            student_feat = {}
        student_handle, student_feat = attach_feature_hook(backbone)
        if student_handle is None:
            print("[distill] Warning: couldn't attach feature hook to student; will fallback to logits for distillation.")

        # training loop
        n_samples_in_loader = len(train_loader.dataset) if not args.balanced_batch_sampler else f"{len(train_loader)} batches x {effective_batch_size}"
        print(f"Starting training for {args.epochs_per_task} epochs on {n_samples_in_loader} samples (loader)")
        start_time = time.time()
        for epoch in range(args.epochs_per_task):
            backbone.train()
            epoch_losses = []
            epoch_cls_losses = []
            epoch_dist_losses = []
            correct = 0
            total = 0
            for imgs, labels in tqdm(train_loader, desc=f"Train {task} (ep {epoch+1})", leave=False):
                imgs = imgs.to(device)
                labels = labels.to(device).long()
                optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        student_feat.clear()
                        student_out = backbone(imgs)
                        loss_ce = criterion(student_out, labels)
                        loss_dist = torch.tensor(0.0, device=device)
                        if teacher is not None:
                            with torch.no_grad():
                                teacher_feat.clear()
                                teacher_out = teacher(imgs)
                            s_feat = student_feat.get('feat', None)
                            t_feat = teacher_feat.get('feat', None)
                            if s_feat is None or t_feat is None:
                                s_feat = student_out.view(student_out.size(0), -1)
                                t_feat = teacher_out.view(teacher_out.size(0), -1)
                            # ALWAYS normalize features before MSE (per your request)
                            s_feat = F.normalize(s_feat, p=2, dim=1)
                            t_feat = F.normalize(t_feat, p=2, dim=1)
                            t_feat = t_feat.detach()
                            if s_feat.size(1) != t_feat.size(1):
                                s_dim = s_feat.size(1)
                                t_dim = t_feat.size(1)
                                if s_dim > t_dim:
                                    pad = torch.zeros((t_feat.size(0), s_dim - t_dim), device=device)
                                    t_feat = torch.cat([t_feat, pad], dim=1)
                                else:
                                    t_feat = t_feat[:, :s_dim]
                            loss_dist = F.mse_loss(s_feat, t_feat)
                        loss = loss_ce + args.distill_lambda * loss_dist
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    student_feat.clear()
                    student_out = backbone(imgs)
                    loss_ce = criterion(student_out, labels)
                    loss_dist = torch.tensor(0.0, device=device)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_feat.clear()
                            teacher_out = teacher(imgs)
                        s_feat = student_feat.get('feat', None)
                        t_feat = teacher_feat.get('feat', None)
                        if s_feat is None or t_feat is None:
                            s_feat = student_out.view(student_out.size(0), -1)
                            t_feat = teacher_out.view(teacher_out.size(0), -1)
                        s_feat = F.normalize(s_feat, p=2, dim=1)
                        t_feat = F.normalize(t_feat, p=2, dim=1)
                        t_feat = t_feat.detach()
                        if s_feat.size(1) != t_feat.size(1):
                            s_dim = s_feat.size(1)
                            t_dim = t_feat.size(1)
                            if s_dim > t_dim:
                                pad = torch.zeros((t_feat.size(0), s_dim - t_dim), device=device)
                                t_feat = torch.cat([t_feat, pad], dim=1)
                            else:
                                t_feat = t_feat[:, :s_dim]
                        loss_dist = F.mse_loss(s_feat, t_feat)
                    loss = loss_ce + args.distill_lambda * loss_dist
                    loss.backward()
                    optimizer.step()
                epoch_losses.append(loss.item())
                epoch_cls_losses.append(float(loss_ce.item()))
                epoch_dist_losses.append(float(loss_dist.item()) if isinstance(loss_dist, torch.Tensor) else 0.0)
                preds = student_out.detach().argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            if scheduler is not None:
                scheduler.step()
            train_loss = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else 0.0
            train_acc = float(correct / total) if total > 0 else 0.0
            mean_ce = float(np.mean(epoch_cls_losses)) if len(epoch_cls_losses) > 0 else 0.0
            mean_dist = float(np.mean(epoch_dist_losses)) if len(epoch_dist_losses) > 0 else 0.0
            if (epoch + 1) % max(1, args.log_every) == 0:
                print(f"Epoch [{epoch+1}/{args.epochs_per_task}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, CE: {mean_ce:.4f}, Dist: {mean_dist:.6f}")
        elapsed = time.time() - start_time
        print(f"Finished task {task} training in {elapsed/60:.2f} minutes")

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
            loss, acc = evaluate_on_images(backbone, test_loader, criterion, device, test_name=f"step{step}_{name}", save_dir="./logs_fd")
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
        df.to_csv(os.path.join("./logs_fd", "sequential_fd_results.csv"), index=False)

    try:
        if teacher_handle is not None:
            teacher_handle.remove()
        if student_handle is not None:
            student_handle.remove()
    except Exception:
        pass

    print("All tasks finished")
    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--initial_backbone', type=str, default='stylegan1', choices=list(PRETRAINED_MODELS.keys()))
    parser.add_argument('--tasks', type=str, default='stylegan1,stylegan2,sdv1_4,stylegan3,stylegan_xl,sdv2_1',
                        help='Comma-separated list of tasks to train on sequentially')
    parser.add_argument('--batch_size', type=int, default=64)
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
    parser.add_argument('--max_train_samples', type=int, default=-1,
                        help='Limit the number of training samples per task (-1 for all)')
    parser.add_argument('--balanced_subset', action='store_true', 
                        help='When used with --max_train_samples, create an exactly balanced subset across classes')
    parser.add_argument('--balanced_batch_sampler', action='store_true', default=False,
                        help='Use BalancedBatchSampler to produce class-balanced batches from the dataset/subset')
    parser.add_argument('--oversample', action='store_true',
                        help='When using balanced_batch_sampler, allow oversampling of smaller classes to match largest class')
    parser.add_argument('--replacement', action='store_true',
                        help='When oversampling, sample with replacement where needed')

    # keep teacher_backbone as optional fallback (used only if prev_ckpt missing)
    parser.add_argument('--teacher_backbone', type=str, default=None, choices=list(PRETRAINED_MODELS.keys()) + [None],
                        help='Fallback teacher name if no prev_ckpt available')
    parser.add_argument('--distill_lambda', type=float, default=1.0, help='Weight for feature distillation loss (MSE)')

    args = parser.parse_args()
    orders = [
        "stylegan1, stylegan2, sdv1_4, stylegan3, stylegan_xl, sdv2_1",
        "stylegan1, stylegan2, stylegan3, stylegan_xl, sdv1_4, sdv2_1",
        "sdv1_4, sdv2_1, stylegan1, stylegan2, stylegan3, stylegan_xl",
        #random order from stylegan2
        "stylegan2, stylegan3,  sdv2_1,stylegan1,stylegan_xl, sdv1_4"
    ]
    import os

    for o in orders:
        args.tasks = o
        results = train_and_eval(args)

        order_str = o.replace(" ", "").replace(",", "_")
        path = f"logs_fd/new_sequential_fd_results_{order_str}.csv"

        # scrive header solo se il file NON esiste
        write_header = not os.path.exists(path)

        results.to_csv(
            path,
            mode="a",          # append
            header=write_header,
            index=False
        )

