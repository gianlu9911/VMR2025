#!/usr/bin/env python3
import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
# Device selection helper (robust)
# ---------------------------------------------
def get_device(arg_device=None):
    if torch.cuda.is_available():
        if arg_device is None:
            return torch.device("cuda")
        try:
            idx = int(arg_device)
            return torch.device(f"cuda:{idx}")
        except:
            return torch.device(arg_device)
    else:
        return torch.device("cpu")

# ---------------------------------------------
# Balanced Batch Sampler (robust)
# ---------------------------------------------
class BalancedBatchSampler(Sampler):
    def __init__(self, labels: torch.Tensor, batch_size: int):
        # normalize labels to cpu long tensor
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().long()
        else:
            labels = torch.tensor(labels, dtype=torch.long)

        self.labels = labels
        self.batch_size = int(batch_size)
        self.classes = sorted(list(torch.unique(self.labels).cpu().tolist()))
        self.num_classes = len(self.classes)
        assert self.batch_size % self.num_classes == 0, "batch_size must be multiple of num_classes"

        # class -> indices (cpu)
        self.class_indices = {
            int(cls): torch.where(self.labels == cls)[0].cpu()
            for cls in self.classes
        }

        self.per_class = self.batch_size // self.num_classes
        # number of full balanced batches based on smallest class
        self.min_class_len = min(len(idxs) for idxs in self.class_indices.values())
        self.num_batches = self.min_class_len // self.per_class

    def __iter__(self):
        # build shuffled iterators
        class_iters = {}
        for cls, idxs in self.class_indices.items():
            perm = idxs[torch.randperm(len(idxs))]
            class_iters[cls] = iter(perm.tolist())

        used_counts = {cls: 0 for cls in self.class_indices}
        for _ in range(self.num_batches):
            batch = []
            for cls in self.classes:
                it = class_iters[cls]
                selected = []
                for _ in range(self.per_class):
                    try:
                        selected_idx = next(it)
                    except StopIteration:
                        # reshuffle and continue
                        idxs = self.class_indices[cls]
                        perm = idxs[torch.randperm(len(idxs))]
                        class_iters[cls] = iter(perm.tolist())
                        it = class_iters[cls]
                        selected_idx = next(it)
                    selected.append(int(selected_idx))
                    used_counts[cls] += 1
                batch.extend(selected)
            yield batch

        print(f"[BalancedBatchSampler] Batches: {self.num_batches}, samples used per class: {used_counts}, total_used: {sum(used_counts.values())}")

    def __len__(self):
        return self.num_batches

# ---------------------------------------------
# Relative Representation and classifier
# ---------------------------------------------
class RelativeRepresentation(nn.Module):
    def __init__(self, anchors, eps=1e-8):
        super().__init__()
        anchors = anchors.float()
        norms = anchors.norm(dim=1, keepdim=True).clamp_min(eps)
        anchors = anchors / norms
        self.register_buffer("anchors", anchors)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        return torch.matmul(x, self.anchors.T)


class RelClassifier(nn.Module):
    def __init__(self, rel_module, in_dim, num_classes=2):
        super().__init__()
        self.rel_module = rel_module
        self.classifier = nn.Linear(in_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        rel_x = self.rel_module(x)  # raw feats -> relative feats
        return self.classifier(rel_x)

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

def evaluate(model, dataloader, criterion, device, rel_module=None, test_name="test_set", save_dir="./logs"):
    model.eval()
    val_loss, val_acc, num_samples = 0.0, 0.0, 0
    all_preds, all_labels, all_feats = [], [], []

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).float().mean().item()

            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size
            val_acc += acc * batch_size
            num_samples += batch_size

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            if rel_module is not None:
                rel_feat = rel_module(features).cpu()  # [batch, num_anchors]
                all_feats.append(rel_feat)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    if rel_module is not None and len(all_feats) > 0:
        all_feats = torch.cat(all_feats)  # [N, num_anchors]
        mean_sim = all_feats.mean(dim=1).numpy()
        std_sim  = all_feats.std(dim=1).numpy()
        max_sim  = all_feats.max(dim=1)[0].numpy()
        min_sim  = all_feats.min(dim=1)[0].numpy()

        df_stats = pd.DataFrame({
            "label": all_labels,
            "pred": all_preds,
            "mean_sim": mean_sim,
            "std_sim": std_sim,
            "max_sim": max_sim,
            "min_sim": min_sim
        })

        csv_path = os.path.join(save_dir, f"relative_stats_{test_name}.csv")
        df_stats.to_csv(csv_path, index=False)
        print(f"Relative feature statistics saved to {csv_path}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, alpha=0.85)
    plt.colorbar()

    # class names
    tick_labels = ["Real", "Fake"] if cm.shape == (2,2) else list(range(cm.shape[0]))
    plt.xticks(np.arange(len(tick_labels)), tick_labels, rotation=45)
    plt.yticks(np.arange(len(tick_labels)), tick_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {test_name}")

    # annotate each cell with counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            plt.text(j, i, str(count),
                    ha="center", va="center",
                    color="white" if count > thresh else "black",
                    fontsize=10, fontweight="bold")

    plt.tight_layout()
    cm_path = os.path.join(save_dir, f"confusion_matrix_{test_name}.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")


    avg_loss = val_loss / num_samples
    avg_acc = val_acc / num_samples
    print(f"[{test_name}] - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    return avg_loss, avg_acc
