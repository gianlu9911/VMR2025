import time
import argparse
import warnings
import os
warnings.filterwarnings("ignore")
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# === Project imports - adjust these to your repo layout ===
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model

# ---------------------------------------------
# Feature extraction helper
# ---------------------------------------------
def extract_and_save_features(backbone, dataloader, feature_path, device, split='train_set'):
    feats, labels = [], []
    print(f"Saving features to {feature_path}...")

    start_time = time.time()
    backbone.eval()
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc=f"Extracting {os.path.basename(feature_path)}"):
            imgs = imgs.to(device)
            out = backbone(imgs)  # assuming backbone returns feature vector
            feats.append(out.cpu())
            labels.append(lbls)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    total_time = time.time() - start_time

    torch.save({"features": feats, "labels": labels}, feature_path)
    print(f"Features saved to {feature_path} (time: {total_time/60:.2f} min)")
    return feats, labels, total_time

# ---------------------------------------------
# Plotting utility: anchors + real + fake features
# ---------------------------------------------
def plot_features_with_anchors(real_feats, fake_feats, anchors, method="pca", save_path=None, subsample=5000, random_state=42):
    """
    real_feats, fake_feats, anchors: torch.Tensor or np.ndarray with shape [N, D]
    method: "pca" or "tsne"
    subsample: maximum number of total points (real+fake) to plot (anchors always included)
    """
    # Convert to numpy
    if isinstance(real_feats, torch.Tensor): real_np = real_feats.cpu().numpy()
    else: real_np = np.array(real_feats)
    if isinstance(fake_feats, torch.Tensor): fake_np = fake_feats.cpu().numpy()
    else: fake_np = np.array(fake_feats)
    if isinstance(anchors, torch.Tensor): anchor_np = anchors.cpu().numpy()
    else: anchor_np = np.array(anchors)

    # Subsample real+fake if necessary
    n_real, n_fake, n_anchor = len(real_np), len(fake_np), len(anchor_np)
    total_points = n_real + n_fake
    if total_points > max(0, subsample):
        # proportionally sample from real and fake
        prop_real = n_real / (n_real + n_fake)
        keep_real = int(round(subsample * prop_real))
        keep_fake = subsample - keep_real
        rng = np.random.default_rng(seed=random_state)
        real_idx = rng.choice(n_real, size=max(1, keep_real), replace=False)
        fake_idx = rng.choice(n_fake, size=max(1, keep_fake), replace=False)
        real_np = real_np[real_idx]
        fake_np = fake_np[fake_idx]
        print(f"[plot] Subsampled to {len(real_np)} real and {len(fake_np)} fake points (anchors: {n_anchor})")
    else:
        print(f"[plot] Using all {n_real} real and {n_fake} fake points (anchors: {n_anchor})")

    # Build matrix and labels
    X = np.vstack([real_np, fake_np, anchor_np])
    y = np.array([0]*len(real_np) + [1]*len(fake_np) + [2]*len(anchor_np))

    # Reduce dimensionality
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        X2 = reducer.fit_transform(X)
    elif method == "tsne":
        # TSNE can be slow for large N
        print("[plot] Running t-SNE (can be slow) ...")
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, init="pca", random_state=random_state)
        X2 = reducer.fit_transform(X)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    # Plotting
    plt.figure(figsize=(8, 6))
    # real
    idx_real = y == 0
    plt.scatter(X2[idx_real, 0], X2[idx_real, 1], marker='o', s=8, alpha=1, label='Real (eval)')
    # fake
    idx_fake = y == 1
    plt.scatter(X2[idx_fake, 0], X2[idx_fake, 1], marker='o', s=8, alpha=1, label='Fake (eval)')
    # anchors (make them visually prominent)
    idx_anchor = y == 2
    plt.scatter(X2[idx_anchor, 0], X2[idx_anchor, 1], alpha = 0.5, marker='*', s=120, edgecolor='k', linewidth=0.6, label='Anchors (train real)')

    plt.legend()
    plt.title(f"Feature visualization ({method.upper()})")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[plot] Saved feature plot to {save_path}")
    else:
        plt.show()

# ---------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_acc, num_samples = 0.0, 0.0, 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
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