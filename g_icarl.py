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

from copy import deepcopy
import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
# change visible devices if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import math
from typing import Sequence, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pandas as pd
from tqdm import tqdm
from logger import *

# project imports (adjust to your repo)
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model


import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, max_per_class):
        self.data = {}  # class_id -> list of (img, label)

    def add(self, imgs, labels):
        for img, y in zip(imgs, labels):
            self.data.setdefault(int(y), []).append((img.cpu(), y))

    def get_dataset(self):
        all_data = []
        for samples in self.data.values():
            all_data.extend(samples)
        return all_data

def icarl_loss(logits, labels, old_logits, lambda_dist):
    ce = F.cross_entropy(logits, labels)

    if old_logits is None:
        return ce

    dist = F.binary_cross_entropy_with_logits(
        logits[:, :old_logits.size(1)],
        torch.sigmoid(old_logits)
    )

    return ce + lambda_dist * dist

def compute_class_means(model, dataset):
    means = {}
    for x, y in dataset:
        f = model.forward_features(x.unsqueeze(0))
        means.setdefault(int(y), []).append(f)

    for c in means:
        means[c] = torch.mean(torch.cat(means[c]), dim=0)

    return means

class ICaRLNet(nn.Module):
    def __init__(self, backbone, feature_dim):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

        # classifier TEMPORANEO (solo per training)
        self.classifier = None

    def forward_features(self, x):
        f = self.backbone(x)
        f = F.normalize(f, p=2, dim=1)  # fondamentale per NCM
        return f

    def set_classifier(self, num_classes):
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        assert self.classifier is not None, "Classifier not set"
        f = self.forward_features(x)
        return self.classifier(f)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_on_images_old(model, test_loader, device, test_name="test"):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Evaluating {test_name}", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device).long()

            logits = model(imgs)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    print(f"[{test_name}] Accuracy: {acc:.4f}")
    return acc

import torch
import numpy as np
import os
from tqdm import tqdm

def evaluate_on_images(model, test_loader, device, test_name="test", save_dir="results"):
    model.eval()
    model.to(device)

    all_logits = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Evaluating {test_name}", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device).long()

            logits = model(imgs)
            
            # Salvataggio logits e labels (portandoli su CPU per non saturare la VRAM)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Concatenazione dei risultati
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Salvataggio su disco
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"logits_{test_name}.npy")
    np.save(save_path, all_logits)
    # Opzionale: salva anche le labels per confronto
    np.save(save_path.replace("logits_", "labels_"), all_labels)

    acc = correct / total if total > 0 else 0.0
    print(f"[{test_name}] Accuracy: {acc:.4f} - Logits salvate in: {save_path}")
    
    return acc, all_logits

def train_one_epoch_icarl(
    model,
    old_model,
    dataloader,
    optimizer,
    device,
    lambda_dist: float
):
    """
    Train ICaRL model for ONE epoch.
    Uses:
      - Cross-Entropy on current labels
      - Distillation loss on old classes (if old_model is provided)
    """

    model.train()
    if old_model is not None:
        old_model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_dist = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(dataloader, desc="Train iCaRL", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        # Forward current model
        logits = model(imgs)

        # --- Cross-Entropy loss (new + old classes) ---
        ce_loss = F.cross_entropy(logits, labels)

        # --- Distillation loss ---
        if old_model is not None:
            with torch.no_grad():
                old_logits = old_model(imgs)

            # sigmoid distillation (iCaRL paper)
            dist_loss = F.binary_cross_entropy_with_logits(
                logits[:, :old_logits.size(1)],
                torch.sigmoid(old_logits)
            )
        else:
            dist_loss = torch.tensor(0.0, device=device)

        loss = ce_loss + lambda_dist * dist_loss
        loss.backward()
        optimizer.step()

        # stats
        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_dist += dist_loss.item() if old_model is not None else 0.0

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    stats = {
        "loss": total_loss / len(dataloader),
        "ce_loss": total_ce / len(dataloader),
        "dist_loss": total_dist / len(dataloader),
        "acc": correct / total if total > 0 else 0.0
    }

    return stats



def train_and_eval(args):
    set_global_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) == 0:
        raise ValueError("No tasks specified. Provide --tasks like 'stylegan1,stylegan2,sdv1_4,stylegan3,stylegan_xl'")

    # load backbone (must output logits directly)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"Using tasks: {tasks}")
    backbone = load_pretrained_model(PRETRAINED_MODELS[tasks[0]])
    in_features = backbone.resnet.fc.in_features
    backbone.resnet.fc = nn.Identity()

    backbone.to(device)
    icarl_net = ICaRLNet(backbone, in_features)
    
    icarl_net.set_classifier(2)
    icarl_net.to(device)

    old_model = None

    checkpoint_dir = "./checkpoint_icarl"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("./logs_icarl", exist_ok=True)

    results = []
    prev_ckpt = None

    for step, task in enumerate(tasks, start=1):
        print('=' * 80)
        print(f"Task {step}/{len(tasks)}: train on {task}")

        # prepare full dataset
        train_dataset = RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR[task], num_training_samples=args.max_train_samples)
        total_train = len(train_dataset)
        print(f"Full dataset size: {total_train}")


        
        
        pin_memory = False
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

        optimizer = torch.optim.Adam(icarl_net.parameters(), lr=args.backbone_lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss().to(device)

        if prev_ckpt is not None:
            ckpt = torch.load(prev_ckpt, map_location='cpu')
            icarl_net.load_state_dict(ckpt['state_dict'])
            icarl_net.to(device)
            print(f"Loaded previous checkpoint {prev_ckpt}")


        # training loop
        start_time = time.time()
        for epoch in range(args.epochs_per_task):
            backbone.train()
            losses = []
            correct = 0
            total = 0

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            train_stats = train_one_epoch_icarl(icarl_net, old_model, train_loader, optimizer, device, args.lamda_dist)
            train_loss = train_stats["loss"]
            train_acc = train_stats["acc"]
            print(f"Epoch [{epoch+1}/{args.epochs_per_task}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        elapsed = time.time() - start_time
        print(f"Finished task {task} training in {elapsed/60:.2f} minutes")

        # save last checkpoint (backbone + optimizer + RNG states)
        last_ckpt = os.path.join(checkpoint_dir, f'ft_step{step}_{task}_last.pth')
        torch.save({'state_dict': icarl_net.state_dict()}, last_ckpt)
        prev_ckpt = last_ckpt

        print(f"Saved last checkpoint: {last_ckpt}")

        old_model = deepcopy(icarl_net)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False
        old_model.to(device)

        # evaluation on test sets
        test_domains = {}

        for domain in args.tasks.split(","):
            test_domains[domain] = RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR[domain], split='test_set', num_training_samples=args.max_train_samples)


        row = {"task": task}
        test_accs = []
        for name, dataset in test_domains.items():
            test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            acc, _ = evaluate_on_images(
                icarl_net,
                test_loader,
                device,test_name=f"step{step}_{name}")  
            print(f"Test_acc={acc})")
            row[name + '_acc'] = acc
            test_accs.append(acc if not np.isnan(acc) else 0.0)

        mean_acc = float(np.mean(test_accs)) if len(test_accs) > 0 else 0.0

        print(f"Mean_test_acc={mean_acc:.4f})")

        results.append(row)
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(f"./logs_icarl", "sequential_finetune_{args.order_str}.csv"), index=False)

    print("All tasks finished")
    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tasks', type=str, default='stylegan1,stylegan2,sdv1_4,stylegan3,stylegan_xl,sdv2_1',)
    parser.add_argument('--batch_size', type=int, default=126)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs_per_task', type=int, default=1)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Limit the number of training samples per task (None for all)')
    parser.add_argument('--order', type=str, default='stylegan1,stylegan2,sdv1_4,stylegan3,stylegan_xl,sdv2_1',)
    
    ### icarl parameters
    parser.add_argument('--lamda_dist', type=float, default=0.1)

    args = parser.parse_args()
    orders = [
        "stylegan1,stylegan2,sdv1_4,stylegan3,stylegan_xl,sdv2_1",
        #"stylegan1,stylegan2,stylegan3,stylegan_xl,sdv1_4,sdv2_1",
        #"sdv1_4,sdv2_1,stylegan1,stylegan2,stylegan3,stylegan_xl",
        #"stylegan2,stylegan3,sdv2_1,stylegan1,stylegan_xl,sdv1_4"
    ]
    for o in orders:
        args.tasks = o
        results = train_and_eval(args)
        print(results)

        order_str = o.replace(" ", "").replace(",", "_")
        args.order = order_str
        path = f"logs_icarl/new_sequential_results_{order_str}.csv"

        # scrive header solo se il file NON esiste
        write_header = not os.path.exists(path)

        results.to_csv(
            path,
            mode="a",          # append
            header=write_header,
            index=False
        )
        a,b,fa = logging(f"logs_icarl/new_sequential_results_{order_str}.csv")
        # write to csv the a,b,f value in append mode
        with open(f"logs_icarl/new_sequential_results_{order_str}.csv", "a") as f:
            f.write(f"ACC:{a:.4f},BWT:{b:.4f},FWT:{fa:.4f}\n")
        # delete lgs_ingore folder
        os.system("rm -rf ./logs_ingore")
        os.system("rm -rf pymp-*")

