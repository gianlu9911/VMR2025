#!/usr/bin/env python3
# fine_tune_joint_fixed.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

from torchvision import transforms

import argparse
import warnings
warnings.filterwarnings("ignore")

# ---- your project imports (assumed to exist) ----
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.net import ResNet50BC, load_pretrained_model  # ResNet50BC is used here


# -----------------------------
# Balanced Joint Sampler
# -----------------------------
import math
import os
import numpy as np
from torch.utils.data import Sampler

class BalancedJointSampler(Sampler):
    """
    Balanced sampler: half real, half fake (fake split equally across provided fake_dirs).
    Prints effective counts of real and fake samples per source.
    """

    def __init__(self, dataset, batch_size, fake_sources, allow_oversample=False, seed=42):
        assert batch_size % 2 == 0, "batch_size must be even"
        if seed is not None:
            np.random.seed(seed)

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.fake_dirs = [os.path.normpath(d) for d in fake_sources]
        self.allow_oversample = bool(allow_oversample)

        # per-batch counts
        self.real_per_batch = self.batch_size // 2
        self.fake_per_batch = self.batch_size // 2

        # distribute fakes evenly across sources
        n_sources = len(self.fake_dirs)
        base_each, rem = divmod(self.fake_per_batch, n_sources)
        self.fake_per_source_list = [base_each + (1 if i < rem else 0) for i in range(n_sources)]

        # collect indices
        self.real_indices = [i for i, p in enumerate(dataset.images) if dataset.labels[p] == 0]

        self.fake_indices = {}
        for fdir in self.fake_dirs:
            idxs = [i for i, p in enumerate(dataset.images)
                    if dataset.labels[p] == 1 and os.path.normpath(fdir) in os.path.normpath(p)]
            self.fake_indices[fdir] = idxs

        # compute number of batches
        if not self.allow_oversample:
            possible_batches_real = len(self.real_indices) // self.real_per_batch
            possible_batches_fake = min(
                (len(self.fake_indices[fdir]) // take) if take > 0 else 0
                for fdir, take in zip(self.fake_dirs, self.fake_per_source_list)
            )
            self.num_batches = min(possible_batches_real, possible_batches_fake)
        else:
            self.num_batches = math.ceil(len(self.real_indices) / self.real_per_batch)

        # Effective totals
        self.total_real = self.num_batches * self.real_per_batch
        self.total_fake = self.num_batches * self.fake_per_batch
        self.total_fake_per_source = {
            fdir: self.num_batches * take
            for fdir, take in zip(self.fake_dirs, self.fake_per_source_list)
        }

        # Print a detailed report
        print(f"[BalancedJointSampler] mode={'oversample' if self.allow_oversample else 'floor'}")
        print(f"  -> batches: {self.num_batches}")
        print(f"  -> Real_total: {self.total_real} (from {len(self.real_indices)} available)")
        for fdir in self.fake_dirs:
            used = self.total_fake_per_source[fdir]
            avail = len(self.fake_indices[fdir])
            print(f"  -> Fake source {os.path.basename(fdir)}: {used} (from {avail} available)")
        print(f"  -> Fake_total (all sources): {self.total_fake}")

        if self.num_batches <= 0:
            raise ValueError("Not enough samples to create a single balanced batch with these settings.")

    def __iter__(self):
        real_needed = self.num_batches * self.real_per_batch
        if len(self.real_indices) >= real_needed and not self.allow_oversample:
            real_seq = np.random.permutation(self.real_indices)[:real_needed]
        else:
            real_seq = np.random.choice(self.real_indices, real_needed, replace=True)

        fake_seqs = {}
        for fdir, per_src in zip(self.fake_dirs, self.fake_per_source_list):
            needed = self.num_batches * per_src
            src_idxs = self.fake_indices.get(fdir, [])
            if len(src_idxs) >= needed and not self.allow_oversample:
                fake_seqs[fdir] = np.random.permutation(src_idxs)[:needed]
            else:
                fake_seqs[fdir] = np.random.choice(src_idxs, needed, replace=True)

        for b in range(self.num_batches):
            start_r = b * self.real_per_batch
            end_r = start_r + self.real_per_batch
            batch = list(real_seq[start_r:end_r])

            for fdir, per_src in zip(self.fake_dirs, self.fake_per_source_list):
                start_f = b * per_src
                end_f = start_f + per_src
                batch.extend(fake_seqs[fdir][start_f:end_f].tolist())

            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# -----------------------------
# Dataset: Real + multiple Fake sources
# -----------------------------
class RealSyntheticDataset(Dataset):
    def __init__(self, real_dir=None, fake_dirs=None, split='train_set',
                 balance_fake_to_real=False, seed=42, image_size=224):
        """
        real_dir: path to real images parent folder (contains 'train_set' / 'test_set' folders)
        fake_dirs: list of fake image parent folders (each contains 'train_set' / 'test_set')
        split: 'train_set' or 'test_set'
        """
        random.seed(seed)
        self.images = []
        self.labels = {}
        self.source_counts = {}

        # Real images
        if real_dir:
            pattern_png = os.path.join(real_dir, split, "*.png")
            pattern_jpg = os.path.join(real_dir, split, "*.jpg")
            real_imgs = sorted(glob(pattern_png)) + sorted(glob(pattern_jpg))
            self.source_counts["real"] = len(real_imgs)
            self.images.extend(real_imgs)
            self.labels.update({p: 0 for p in real_imgs})
        else:
            self.source_counts["real"] = 0

        # Fake images (from multiple directories)
        all_fake = []
        if fake_dirs:
            for i, fdir in enumerate(fake_dirs):
                search_dir = os.path.join(fdir, split)
                print(f"Scanning fake dir: {search_dir}")
                fake_png = sorted(glob(os.path.join(search_dir, "*.png")))
                fake_jpg = sorted(glob(os.path.join(search_dir, "*.jpg")))
                fake_i = list(dict.fromkeys(fake_png + fake_jpg))  # deduplicate while preserving order
                self.source_counts[f"fake_{i}"] = len(fake_i)
                print(f"  -> found {len(fake_i)} images in fake_{i}")
                all_fake.extend(fake_i)

        # Optionally balance fake count to real count
        if balance_fake_to_real and self.source_counts.get("real", 0) > 0:
            max_fake = self.source_counts["real"]
            if len(all_fake) > max_fake:
                all_fake = random.sample(all_fake, max_fake)
            self.source_counts["used_fake"] = len(all_fake)
        else:
            self.source_counts["used_fake"] = len(all_fake)

        # Merge
        self.images.extend(all_fake)
        self.labels.update({p: 1 for p in all_fake})  # fake label = 1

        self.len = len(self.images)
        print(f"Dataset split='{split}': real={self.source_counts['real']}, used_fake={self.source_counts['used_fake']}, total={self.len}")

        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = self.labels[img_path]
        return x, int(y)


# -----------------------------
# Training and Evaluation helpers
# -----------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_acc, num_samples = 0.0, 0.0, 0

    for imgs, labels in tqdm(dataloader, desc="Train batches"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean().item()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += acc * batch_size
        num_samples += batch_size

    return running_loss / num_samples, running_acc / num_samples


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, dataloader, criterion, device, test_name="test_set", save_dir="./logs/joint"):
    """
    Evaluate model and save confusion matrix under `save_dir`.
    Returns (avg_loss, avg_acc).
    """
    model.eval()
    val_loss, val_acc, num_samples = 0.0, 0.0, 0
    all_preds, all_labels = [], []

    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=f"Eval {test_name}"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).float().mean().item()

            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size
            val_acc += acc * batch_size
            num_samples += batch_size

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    if num_samples == 0:
        return float("nan"), float("nan")

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real","Fake"], yticklabels=["Real","Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {test_name}")
    cm_path = os.path.join(save_dir, f"confusion_matrix_{test_name}.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    avg_loss = val_loss / num_samples
    avg_acc = val_acc / num_samples
    print(f"[{test_name}] - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    return avg_loss, avg_acc


# -----------------------------
# Fine-tuning pipeline
# -----------------------------
def fine_tune(args):
    # Device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Directories
    checkpoint_dir = "./checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logs_dir = "./logs/joint"
    os.makedirs(logs_dir, exist_ok=True)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model: attempt to load pretrained backbone if requested, otherwise create ResNet50BC
    if args.use_pretrained and args.backbone in PRETRAINED_MODELS:
        print("Loading pretrained model:", args.backbone)
        model = load_pretrained_model(PRETRAINED_MODELS[args.backbone])
        # try to replace final classification head to 2 classes
        if hasattr(model, "resnet") and hasattr(model.resnet, "fc"):
            in_f = model.resnet.fc.in_features
            model.resnet.fc = nn.Linear(in_f, 2)
        elif hasattr(model, "fc"):
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, 2)
        else:
            # if model returns raw features, add a small classifier wrapper
            model = nn.Sequential(model, nn.Linear(getattr(model, "num_features", 512), 2))
    else:
        model = ResNet50BC()
        # try to ensure final fc has 2 outputs
        if hasattr(model, "resnet") and hasattr(model.resnet, "fc"):
            in_f = model.resnet.fc.in_features
            model.resnet.fc = nn.Linear(in_f, 2)
        elif hasattr(model, "fc"):
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, 2)

    model = model.to(device)
    print("Model prepared for 2-class classification (fine-tuning).")

    # Dataset directories
    real_dir = IMAGE_DIR.get('real')
    fake_dirs_all = [
        IMAGE_DIR.get("stylegan1"),
        IMAGE_DIR.get("stylegan2"),
        IMAGE_DIR.get("stylegan_xl"),
        IMAGE_DIR.get("sdv1_4"),
    ]
    # keep only existing ones
    fake_dirs_all = [d for d in fake_dirs_all if d is not None]
    if len(fake_dirs_all) == 0:
        raise RuntimeError("No fake directories found in IMAGE_DIR. Please check config.IMAGE_DIR")

    # Build training dataset (all fakes)
    train_dataset = RealSyntheticDataset(real_dir=real_dir, fake_dirs=fake_dirs_all, split="train_set",
                                         balance_fake_to_real=False, seed=args.seed, image_size=args.image_size)

    # Use balanced sampler that enforces half real / half fake and equal distribution among fake sources
    sampler = BalancedJointSampler(train_dataset, batch_size=args.batch_size,
                               fake_sources=("stylegan1", "stylegan2", "styleganxl", "sdv1_4"))
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())

    print(f"Training on {len(train_dataset)} images (real + all fakes). Number of balanced batches per epoch: {len(sampler)}")

    # Criterion & optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train loop
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed/60:.2f} minutes")

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"finetuned_joint_resnet50bc.pth")
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # ------------------------------
    # Evaluation on separate test splits: each fake source individually
    # ------------------------------
    tests = {
        "real_vs_stylegan1": [IMAGE_DIR.get("stylegan1")] if IMAGE_DIR.get("stylegan1") else [],
        "real_vs_stylegan2": [IMAGE_DIR.get("stylegan2")] if IMAGE_DIR.get("stylegan2") else [],
        "real_vs_styleganxl": [IMAGE_DIR.get("stylegan_xl")] if IMAGE_DIR.get("stylegan_xl") else [],
        "real_vs_sdv1_4": [IMAGE_DIR.get("sdv1_4")] if IMAGE_DIR.get("sdv1_4") else [],
    }

    test_results = {}
    for name, fdirs in tests.items():
        if not fdirs or any(d is None for d in fdirs):
            print(f"Skipping {name} because directory missing.")
            continue

        test_dataset = RealSyntheticDataset(real_dir=real_dir, fake_dirs=fdirs, split="test_set",
                                            balance_fake_to_real=False, seed=args.seed, image_size=args.image_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

        loss, acc = evaluate(model, test_loader, criterion, device, test_name=name, save_dir=logs_dir)
        test_results[name] = {"loss": float(loss), "acc": float(acc)}

    # Save summary CSV in logs/joint
    summary_df = pd.DataFrame(test_results).T
    summary_path = os.path.join(logs_dir, "joint_test_summary.csv")
    summary_df.to_csv(summary_path, index=True)
    print(f"Saved test summary to {summary_path}")
    print(summary_df)

    return test_results


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--use_pretrained', action='store_true',
                        help="If set, attempt to load model from PRETRAINED_MODELS using --backbone key")
    parser.add_argument('--backbone', type=str, default='stylegan1',
                        choices=list(PRETRAINED_MODELS.keys()) if isinstance(PRETRAINED_MODELS, dict) else ['stylegan1','stylegan2','stylegan_xl','sdv1_4'])
    parser.add_argument('--num_train_samples', type=int, default=10000,
                        help="If set, subsample training images to this many (optional).")
    args = parser.parse_args()

    fine_tune(args)
