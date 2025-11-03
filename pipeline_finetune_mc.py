#!/usr/bin/env python3
"""
Sequential fine-tuning (binary training per-step, multiclass evaluation).

Assumes dataset provides correct labels.
No remapping, no balanced subset logic unless explicitly requested via CLI.
"""
from torch.utils.data import DataLoader, Dataset, Subset

import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm

from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSyntheticDataloaderMC as RealSynthethicDataloader
from src.net import load_pretrained_model

# --- Domain mapping / global class ids ---
ALL_DOMAINS = ['real', 'stylegan1', 'stylegan2', 'sdv1_4', 'stylegan3', 'stylegan_xl', 'sdv2_1']
CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(ALL_DOMAINS)}
NUM_CLASSES = len(ALL_DOMAINS)
print(f"[config] NUM_CLASSES = {NUM_CLASSES}; mapping = {CLASS_NAME_TO_ID}")

# --- Utilities ---
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def replace_last_linear(module: nn.Module, new_out_features: int) -> bool:
    """Find last nn.Linear and replace with new_out_features."""
    last_linear = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        print("[replace_last_linear] No nn.Linear found.")
        return False

    for parent_name, parent in module.named_modules():
        for child_name, child in parent.named_children():
            if child is last_linear:
                in_f = child.in_features
                new_linear = nn.Linear(in_f, new_out_features)
                nn.init.kaiming_uniform_(new_linear.weight, a=math.sqrt(5))
                if new_linear.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_linear.weight)
                    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(new_linear.bias, -bound, bound)
                setattr(parent, child_name, new_linear)
                print(f"[replace_last_linear] Replaced {parent_name}.{child_name} -> nn.Linear({in_f}, {new_out_features})")
                return True
    return False

# --- Evaluation ---
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_on_images(backbone, test_loader, criterion, device, test_name="test", save_dir="./logs_mc_finetuning"):
    backbone.eval()
    device = torch.device(device if isinstance(device, str) else device)
    backbone.to(device)
    os.makedirs(save_dir, exist_ok=True)

    rows, losses = [], []
    sample_idx = 0
    all_preds = []
    all_targets = []

    try:
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {test_name}", leave=False):
                # robust batch unpacking
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    imgs, labels = batch[0], batch[1]
                elif isinstance(batch, dict):
                    imgs = batch.get('image') or batch.get('img') or batch.get('images')
                    labels = batch.get('label') or batch.get('target') or batch.get('labels')
                    if imgs is None or labels is None:
                        # fallback: first two values
                        vals = list(batch.values())
                        imgs, labels = vals[0], vals[1]
                else:
                    raise ValueError("Unknown batch format from dataloader.")

                imgs, labels = imgs.to(device), labels.to(device).long()

                outputs = backbone(imgs)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                probs = F.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)

                probs_cpu = probs.cpu().numpy()
                preds_cpu = preds.cpu().numpy()
                labels_cpu = labels.cpu().numpy()

                all_preds.extend(preds_cpu.tolist())
                all_targets.extend(labels_cpu.tolist())

                for i in range(labels_cpu.shape[0]):
                    row = {"idx": sample_idx, "label": int(labels_cpu[i]), "pred": int(preds_cpu[i])}
                    for c in range(probs_cpu.shape[1]):
                        row[f"prob_class_{c}"] = float(probs_cpu[i, c])
                    rows.append(row)
                    sample_idx += 1

        if len(rows) == 0:
            print(f"[evaluate_on_images] WARNING: no samples evaluated for {test_name}")
            return float('nan'), float('nan')

        avg_loss = float(np.mean(losses))
        accuracy = float(accuracy_score(all_targets, all_preds))

        # Save predictions CSV + summary
        df_preds = pd.DataFrame(rows)
        preds_csv = os.path.join(save_dir, f"{test_name}_predictions.csv")
        df_preds.to_csv(preds_csv, index=False)

        summary_txt = os.path.join(save_dir, f"{test_name}_summary.txt")
        with open(summary_txt, "w") as f:
            f.write(f"test_name: {test_name}\nnum_samples: {len(rows)}\navg_loss: {avg_loss:.6f}\naccuracy: {accuracy:.6f}\n")
        print(f"[evaluate] Saved predictions CSV and summary for {test_name}")

        # Try to infer class names from dataset
        dataset = test_loader.dataset
        class_names = None
        if hasattr(dataset, 'classes'):
            class_names = list(dataset.classes)
        elif hasattr(dataset, 'class_to_idx'):
            inv = {v:k for k,v in dataset.class_to_idx.items()}
            class_names = [inv[i] if i in inv else str(i) for i in range(max(inv.keys())+1)]
        else:
            unique_labels = sorted(list(set(all_targets + all_preds)))
            class_names = [str(c) for c in unique_labels]

        # confusion matrix (ensure label order 0..N-1)
        # confusion matrix (force 7 classes)
        FIXED_NUM_CLASSES = 7
        class_names = [str(i) for i in range(FIXED_NUM_CLASSES)]
        n_classes = FIXED_NUM_CLASSES
        labels_range = list(range(FIXED_NUM_CLASSES))

        cm = confusion_matrix(all_targets, all_preds, labels=labels_range)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(os.path.join(save_dir, f"{test_name}_confusion_matrix_raw.csv"))

        # normalized (by true row)
        with np.errstate(all='ignore'):
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
        cmn_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
        cmn_df.to_csv(os.path.join(save_dir, f"{test_name}_confusion_matrix_normalized.csv"))

        # plot raw
        plt.figure(figsize=(max(6, n_classes*0.5), max(5, n_classes*0.5)))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{test_name} - Confusion matrix (raw)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{test_name}_confusion_matrix_raw.png"), dpi=150)
        plt.close()

        # plot normalized
        plt.figure(figsize=(max(6, n_classes*0.5), max(5, n_classes*0.5)))
        sns.heatmap(cmn_df, annot=True, fmt='.2f', cmap='Blues', cbar=True)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{test_name} - Confusion matrix (normalized)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{test_name}_confusion_matrix_normalized.png"), dpi=150)
        plt.close()

        # classification report
        try:
            report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report_dict).transpose()
            report_df.to_csv(os.path.join(save_dir, f"{test_name}_classification_report.csv"))
        except Exception as e:
            print(f"[evaluate_on_images] Warning: couldn't compute classification_report: {e}")

        print(f"[evaluate_on_images] {test_name}: loss={avg_loss:.4f}, acc={accuracy:.4f} - saved to {save_dir}")
        return float(avg_loss), float(accuracy)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[evaluate_on_images] ERROR during evaluation {test_name}: {e}")
        return float('nan'), float('nan')


# --- Training loop ---
def train_and_eval(args):
    set_global_seed(args.seed)
    device = torch.device('cuda')
    print(f"Using device: {device}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    backbone = load_pretrained_model(PRETRAINED_MODELS[args.initial_backbone])
    replace_last_linear(backbone, NUM_CLASSES)
    backbone.to(device)

    checkpoint_dir = "./checkpoint_finetune_mc"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("./logs_mc_finetuning", exist_ok=True)
    results = []

    for step, task in enumerate(tasks, start=1):
        print('=' * 80)
        print(f"Task {step}/{len(tasks)}: train on real + {task}")
        try:
            # --- TRAIN DATASET ---
            train_dataset = RealSynthethicDataloader('real', task)
            total_train = len(train_dataset)
            print(f"  train_dataset size: {total_train}")

            rng = np.random.RandomState(args.seed)
            if args.max_train_samples is not None and args.max_train_samples > 0 and args.max_train_samples < total_train:
                max_n = int(args.max_train_samples)
                if args.balanced_subset:
                    print(f"  Building balanced subset with {max_n} train samples")
                    label_list = getattr(train_dataset, 'targets', None) or getattr(train_dataset, 'labels', None)
                    label_to_idxs = {}
                    if label_list is not None and len(label_list) == total_train:
                        for idx, lbl in enumerate(label_list):
                            label_to_idxs.setdefault(int(lbl), []).append(idx)
                    else:
                        for idx in range(total_train):
                            try:
                                item = train_dataset[idx]
                                lbl = item[1] if isinstance(item, (list, tuple)) else item.get('label')
                                lbl = int(lbl)
                                label_to_idxs.setdefault(lbl, []).append(idx)
                            except Exception:
                                continue
                    classes = sorted(label_to_idxs.keys())
                    per_class = max_n // max(1, len(classes))
                    chosen = []
                    for lbl in classes:
                        idxs = label_to_idxs[lbl]
                        rng.shuffle(idxs)
                        chosen.extend(idxs[:per_class])
                    chosen = chosen[:max_n]
                else:
                    print(f"  Building random subset with {max_n} train samples")
                    chosen = rng.choice(total_train, size=max_n, replace=False).tolist()
                train_dataset = Subset(train_dataset, chosen)
                print(f"  Using subset of train_dataset -> {len(train_dataset)} samples")

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

            # --- TRAIN LOOP ---
            optimizer = torch.optim.Adam(backbone.parameters(), lr=args.backbone_lr, weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss().to(device)
            scaler = torch.cuda.amp.GradScaler() if (args.use_amp and torch.cuda.is_available()) else None

            print("START TRAINING")
            for epoch in range(args.epochs_per_task):
                backbone.train()
                losses, correct, total = [], 0, 0
                for batch in tqdm(train_loader, leave=False):
                    imgs, labels = batch if isinstance(batch, (tuple, list)) else (batch['image'], batch['label'])
                    imgs, labels = imgs.to(device), labels.to(device).long()

                    optimizer.zero_grad()
                    if scaler:
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
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                print(f"Epoch [{epoch+1}/{args.epochs_per_task}] - Loss: {np.mean(losses):.4f}, Acc: {correct/total:.4f}")

            # save last checkpoint
            last_ckpt = os.path.join(checkpoint_dir, f'ft_step{step}_{task}_last.pth')
            torch.save({'backbone_state': backbone.state_dict()}, last_ckpt)
            print(f"Saved checkpoint: {last_ckpt}")

            # --- TEST SETS ---
            test_results, test_accs = {}, []
            for domain in ALL_DOMAINS[1:]:
                test_dataset = RealSynthethicDataloader('real', domain, split='test_set')
                total_test = len(test_dataset)
                print(f"    test_dataset [{domain}] size: {total_test}")

                if args.max_test_samples is not None and args.max_test_samples > 0 and args.max_test_samples < total_test:
                    max_t = int(args.max_test_samples)
                    if args.balanced_test_subset:
                        print(f"    Building balanced test subset ({max_t} samples)")
                        label_list = getattr(test_dataset, 'targets', None) or getattr(test_dataset, 'labels', None)
                        label_to_idxs = {}
                        if label_list is not None and len(label_list) == total_test:
                            for idx, lbl in enumerate(label_list):
                                label_to_idxs.setdefault(int(lbl), []).append(idx)
                        else:
                            for idx in range(total_test):
                                try:
                                    item = test_dataset[idx]
                                    lbl = item[1] if isinstance(item, (list, tuple)) else item.get('label')
                                    lbl = int(lbl)
                                    label_to_idxs.setdefault(lbl, []).append(idx)
                                except Exception:
                                    continue
                        classes = sorted(label_to_idxs.keys())
                        per_class = max_t // max(1, len(classes))
                        chosen = []
                        for lbl in classes:
                            idxs = label_to_idxs[lbl]
                            rng.shuffle(idxs)
                            chosen.extend(idxs[:per_class])
                        chosen = chosen[:max_t]
                    else:
                        print(f"    Building random test subset ({max_t} samples)")
                        chosen = rng.choice(total_test, size=max_t, replace=False).tolist()
                    test_dataset = Subset(test_dataset, chosen)
                    print(f"    Using subset of test_dataset -> {len(test_dataset)} samples")

                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                loss, acc = evaluate_on_images(backbone, test_loader, criterion, device,
                                               test_name=f"step{step}_test_{domain}",
                                               save_dir="./logs_finetuning_mc")
                test_results[domain + "_acc"] = acc
                test_accs.append(acc if not np.isnan(acc) else 0.0)

            mean_acc = float(np.mean(test_accs))
            results.append({"task": task, "step": step, "mean_test_acc": mean_acc, **test_results})
            pd.DataFrame(results).to_csv("./logs_finetuning_mc/sequential_finetune_mc_results.csv", index=False)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[train_and_eval] ERROR on task {task}: {e} -- continuing with next task.")
            continue


    print("All tasks finished")
    return pd.DataFrame(results)

# --- Entry point ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_backbone', type=str, default='stylegan1', choices=list(PRETRAINED_MODELS.keys()))
    parser.add_argument('--tasks', type=str, default='stylegan1,stylegan2,sdv1_4,stylegan3,stylegan_xl,sdv2_1')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs_per_task', type=int, default=1)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--max_train_samples', type=int, default=-1,
                        help='Limit the number of training samples per task (-1 for all).')
    parser.add_argument('--balanced_subset', action='store_true',
                        help='When used with --max_train_samples, create an exactly balanced subset across classes.')
    parser.add_argument('--max_test_samples', type=int, default=-1,
                        help='Limit the number of test samples per domain (-1 for all).')
    parser.add_argument('--balanced_test_subset', action='store_true',
                        help='When used with --max_test_samples, create a balanced test subset.')


    args = parser.parse_args()

    results = train_and_eval(args)
    print(results)
