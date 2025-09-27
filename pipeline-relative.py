#!/usr/bin/env python3
"""
Sequential fine-tuning pipeline (improved / robust version)

Main fixes/improvements:
- deterministic seeding for torch/numpy/random
- safer BalancedBatchSampler usage (labels -> numpy int array)
- checkpoint loading with strict=False (warns on mismatches)
- save best checkpoint per task (based on aggregated test acc)
- minor evaluate CSV bugfixes and defensive checks
- optional LR scheduler and simple logging
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
from src.utils import (get_device, BalancedBatchSampler, RelativeRepresentation,
                       RelClassifier, plot_features_with_anchors, extract_and_save_features,
                       train_one_epoch)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if cuda available, set cuda seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic flags (may slow down but increases reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(classifier, test_loader, criterion, device,
             rel_module=None, test_name="test", save_dir="./logs"):
    """
    Evaluate classifier on test_loader. Saves per-sample CSV and a small summary txt.
    Returns (avg_loss, accuracy).
    """
    classifier.eval()
    device = torch.device(device if isinstance(device, str) else device)
    classifier.to(device)

    os.makedirs(save_dir, exist_ok=True)
    preds_all = []
    labels_all = []
    probs_all = []
    losses = []
    sample_idx = 0
    rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {test_name}", leave=False)):
            # expect batch: (features, labels)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                feats, labels = batch[0], batch[1]
            else:
                raise ValueError("Expected test_loader to yield (features, labels) pairs")

            feats = feats.to(device)
            labels = labels.to(device).long()

            outputs = classifier(feats)  # classifier should accept raw features
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            # collect
            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())
            probs_all.append(probs.cpu())

            # build rows for CSV
            probs_cpu = probs.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            batch_size = labels_cpu.shape[0]
            num_probs = probs_cpu.shape[1] if probs_cpu.ndim == 2 else 0
            for i in range(batch_size):
                row = {
                    "idx": sample_idx,
                    "label": int(labels_cpu[i]),
                    "pred": int(preds_cpu[i]),
                    # guard against unexpected number of classes
                    "prob_real": float(probs_cpu[i, 0]) if num_probs >= 1 else 0.0,
                    "prob_fake": float(probs_cpu[i, 1]) if num_probs >= 2 else 0.0
                }
                rows.append(row)
                sample_idx += 1

    # concat results
    preds_all = torch.cat(preds_all).numpy() if len(preds_all) > 0 else np.array([], dtype=int)
    labels_all = torch.cat(labels_all).numpy() if len(labels_all) > 0 else np.array([], dtype=int)
    probs_all = torch.cat(probs_all).numpy() if len(probs_all) > 0 else np.zeros((0, 2))

    # compute metrics
    if labels_all.size == 0:
        avg_loss = float(np.nan)
        accuracy = float(np.nan)
    else:
        avg_loss = float(np.mean(losses))
        accuracy = float((preds_all == labels_all).sum() / labels_all.shape[0])

    # Save CSV
    csv_out = os.path.join(save_dir, f"{test_name}_predictions.csv")
    df_preds = pd.DataFrame(rows, columns=["idx", "label", "pred", "prob_real", "prob_fake"])
    df_preds.to_csv(csv_out, index=False)

    # also save a small summary txt
    summary_out = os.path.join(save_dir, f"{test_name}_summary.txt")
    with open(summary_out, "w") as f:
        f.write(f"test_name: {test_name}\n")
        f.write(f"num_samples: {len(labels_all)}\n")
        f.write(f"avg_loss: {avg_loss:.6f}\n")
        f.write(f"accuracy: {accuracy:.6f}\n")
    print(f"[evaluate] Saved predictions CSV to {csv_out} and summary to {summary_out}")

    return avg_loss, accuracy


def train_and_eval_sequential(args):
    # set seeds
    set_global_seed(args.seed)

    device = get_device(args.device)
    print(f"Using device: {device}")

    # tasks order (domains to train sequentially on)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) == 0:
        raise ValueError("No tasks specified. Provide --tasks like 'stylegan1,stylegan_xl,sdv1_4'")

    # use a single backbone (loaded from PRETRAINED_MODELS starting point)
    initial_backbone_name = args.initial_backbone
    if initial_backbone_name not in PRETRAINED_MODELS:
        raise KeyError(f"initial_backbone '{initial_backbone_name}' not found in PRETRAINED_MODELS keys: {list(PRETRAINED_MODELS.keys())}")
    backbone = load_pretrained_model(PRETRAINED_MODELS[initial_backbone_name])
    # strip fc if present
    if hasattr(backbone, "resnet") and hasattr(backbone.resnet, "fc"):
        backbone.resnet.fc = nn.Identity()
    backbone.to(device)
    backbone.eval()

    feature_dir = f"./feature_{initial_backbone_name}"
    checkpoint_dir = "./checkpoint_sequential"
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    # Prepare table to collect accuracies
    test_names = [
        "real_vs_stylegan1",
        "real_vs_stylegan2",
        "real_vs_styleganxl",
        "real_vs_sdv1_4"
    ]
    results_df = pd.DataFrame(columns=["task"] + [c + "_acc" for c in test_names])

    prev_classifier_ckpt = None

    # Keep RNG reproducible for anchor sampling
    rng = torch.Generator()
    rng.manual_seed(args.seed)

    for step, task in enumerate(tasks, start=1):
        print("\n" + "="*60)
        print(f"Task {step}/{len(tasks)}: Train on '{task}' (starting from previous checkpoint: {bool(prev_classifier_ckpt)})")

        # --- Extract / load training features for this task ---
        feat_file = os.path.join(feature_dir, f"train_real_vs_{task}_features.pt")
        if args.force_recompute_features or not os.path.exists(feat_file):
            print(f"Extracting training features for task {task} -> {feat_file}")
            train_dataset = RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR[task])
            loader = DataLoader(train_dataset, batch_size=args.feature_batch_size, shuffle=False,
                                num_workers=args.num_workers)
            feats_full, labels_full, feat_time = extract_and_save_features(backbone, loader, feat_file, device)
        else:
            print(f"Loading cached training features from {feat_file}")
            data = torch.load(feat_file, map_location='cpu')
            feats_full, labels_full = data['features'], data['labels']

        # optional subsample of training data
        if args.num_train_samples is not None and args.num_train_samples < len(feats_full):
            indices = torch.randperm(len(feats_full))[:args.num_train_samples]
            feats = feats_full[indices]
            labels = labels_full[indices]
        else:
            feats = feats_full
            labels = labels_full

        # ensure labels are CPU tensors
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu()
        print(f"Training samples: {len(feats)} (real={(labels==0).sum().item()}, fake={(labels==1).sum().item()})")

        # --- Anchors from real training features ---
        real_mask = (labels == 0)
        real_feats = feats[real_mask]
        if real_feats.size(0) == 0:
            raise RuntimeError("No real samples available to form anchors for task: " + task)

        num_requested = args.num_anchors
        if num_requested > len(real_feats):
            print(f"[warning] Requested {num_requested} anchors but only {len(real_feats)} reals; sampling WITH replacement.")
            idx = torch.randint(low=0, high=len(real_feats), size=(num_requested,), generator=rng)
        else:
            idx = torch.randperm(len(real_feats), generator=rng)[:num_requested]
        anchors = real_feats[idx]
        print(f"Anchors: {anchors.size(0)}")

        rel_module = RelativeRepresentation(anchors.to(device))

        # --- Prepare training DataLoader with BalancedBatchSampler ---
        feat_dataset = TensorDataset(feats, labels)
        # BalancedBatchSampler implementations often accept numpy arrays of labels
        labels_for_sampler = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
        labels_for_sampler = labels_for_sampler.astype(np.int64)
        sampler = BalancedBatchSampler(labels_for_sampler, batch_size=args.batch_size)
        feat_loader = DataLoader(feat_dataset, batch_sampler=sampler, num_workers=0)

        # --- Initialize classifier; if previous checkpoint exists, load weights ---
        classifier = RelClassifier(rel_module, anchors.size(0), num_classes=2).to(device)
        if prev_classifier_ckpt is not None:
            try:
                ckpt = torch.load(prev_classifier_ckpt, map_location='cpu')
                state = ckpt.get('state_dict', ckpt)
                # try load with strict=False in case representation (anchors) changed slightly
                # will warn if keys are missing
                missing_keys, unexpected_keys = classifier.load_state_dict(state, strict=False)
                print(f"Loaded previous classifier weights from {prev_classifier_ckpt} (strict=False).")
                if missing_keys:
                    print(f"[warning] Missing keys when loading checkpoint: {missing_keys}")
                if unexpected_keys:
                    print(f"[warning] Unexpected keys in checkpoint: {unexpected_keys}")
            except Exception as e:
                print(f"[warning] Failed loading previous checkpoint {prev_classifier_ckpt}: {e}. Continuing from scratch.")

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
        scheduler = None
        if args.use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

        # --- Train ---
        best_task_ckpt = None
        best_task_score = -np.inf
        start_time = time.time()
        for epoch in range(args.epochs_per_task):
            train_loss, train_acc = train_one_epoch(classifier, feat_loader, criterion, optimizer, device)
            if scheduler is not None:
                scheduler.step()
            if (epoch+1) % max(1, args.log_every) == 0:
                print(f"Task {task} Epoch [{epoch+1}/{args.epochs_per_task}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        elapsed = time.time() - start_time
        print(f"Task {task} training finished in {elapsed/60:.2f} minutes")

        # temporarily save 'last' checkpoint
        last_ckpt_path = os.path.join(checkpoint_dir, f'seq_finetuned_step{step}_{task}_last.pth')
        torch.save({'state_dict': classifier.state_dict()}, last_ckpt_path)
        print(f"Saved last classifier checkpoint: {last_ckpt_path}")

        # --- Prepare test sets and evaluate ---
        test_domains = {
            "real_vs_stylegan1": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan1'], split='test_set'),
            "real_vs_stylegan2": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan2'], split='test_set'),
            "real_vs_styleganxl": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['stylegan_xl'], split='test_set'),
            "real_vs_sdv1_4": RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR['sdv1_4'], split='test_set')
        }

        row = {"task": task}
        # Evaluate on each test domain and accumulate mean accuracy for deciding "best"
        test_accs = []
        for name, dataset in test_domains.items():
            feat_file_test = os.path.join(feature_dir, f"test_{name}_features.pt")
            print(f"Preparing test features for {name} -> {feat_file_test}")
            if args.force_recompute_features or not os.path.exists(feat_file_test):
                loader = DataLoader(dataset, batch_size=args.feature_batch_size, shuffle=False,
                                    num_workers=args.num_workers)
                feats_test, labels_test, feat_time = extract_and_save_features(backbone, loader, feat_file_test, device, split='test_set')
            else:
                data = torch.load(feat_file_test, map_location='cpu')
                feats_test, labels_test = data['features'], data['labels']

            # Plot (optional)
            plot_save_path = os.path.join("./logs", f"feature_plot_seq_step{step}_{name}.png")
            try:
                plot_features_with_anchors(feats_test[labels_test==0], feats_test[labels_test==1], anchors.cpu(),
                                           method=args.plot_method, save_path=plot_save_path, subsample=args.plot_subsample)
            except Exception as e:
                print(f"[warning] Failed to plot features for {name}: {e}")

            # Evaluate
            test_dataset = TensorDataset(feats_test, labels_test)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            loss, acc = evaluate(classifier, test_loader, criterion, device, rel_module=rel_module, test_name=f"step{step}_{name}", save_dir="./logs")
            row[name + '_acc'] = acc
            test_accs.append(acc if not np.isnan(acc) else 0.0)

        # pick checkpoint to pass to next task:
        mean_acc = float(np.mean(test_accs)) if len(test_accs) > 0 else 0.0
        # if this run beat previous best for this step, save as best
        best_ckpt_path = os.path.join(checkpoint_dir, f'seq_finetuned_step{step}_{task}_best.pth')
        torch.save({'state_dict': classifier.state_dict(), 'mean_test_acc': mean_acc}, best_ckpt_path)
        prev_classifier_ckpt = best_ckpt_path
        print(f"Saved best classifier checkpoint for task '{task}': {best_ckpt_path} (mean_test_acc={mean_acc:.4f})")

        # Append row to results DataFrame and save intermediate CSV
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        csv_out = os.path.join("./logs", "sequential_task_results.csv")
        results_df.to_csv(csv_out, index=False)
        print(f"Saved results table to {csv_out}")

    print("\nSequential pipeline finished.")
    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help="Device string accepted by get_device")
    parser.add_argument('--initial_backbone', type=str, default='stylegan1',
                        choices=list(PRETRAINED_MODELS.keys()))
    parser.add_argument('--tasks', type=str, default='stylegan1,stylegan2,stylegan_xl,sdv1_4',
                        help="Comma-separated list of dataset domains to train sequentially on.")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--feature_batch_size', type=int, default=1024,
                        help='Batch size used during feature extraction')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs_per_task', type=int, default=10,
                        help='Number of training epochs per task')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_samples', type=int, default=None)
    parser.add_argument('--num_anchors', type=int, default=5000)
    parser.add_argument('--plot_method', type=str, default='pca', choices=['pca', 'tsne'])
    parser.add_argument('--plot_subsample', type=int, default=5000)
    parser.add_argument('--force_recompute_features', action='store_true')
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--scheduler_step', type=int, default=5)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    args = parser.parse_args()
    results = train_and_eval_sequential(args)
    print(results)
