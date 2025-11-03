#!/usr/bin/env python3
import os
import time
import argparse
import warnings


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

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# -------------------------
# RelClassifier con embedding layer
# -------------------------
class RelClassifierWithEmbedding(nn.Module):
    def __init__(self, rel_module, num_anchors, embedding_dim=256, hidden_dim=512, num_classes=2, dropout=0.2):
        """
        rel_module: modulo che mappa input -> relative representation (assunto già fornito)
        num_anchors: numero di anchors (può servire per test/debug)
        embedding_dim: dimensionalità del nuovo embedding layer
        hidden_dim: dimensione layer intermedio (opzionale)
        """
        super().__init__()
        self.rel = rel_module
        # simple MLP embedding head
        # assumo che rel_module restituisca un vettore 1D (es. dim D)
        rel_out_dim = num_anchors  # se il rel_module ha attributo out_dim
        # fallback: infer later dinamicamente con Linear after seeing input (but keep simple)
        if rel_out_dim is None:
            rel_out_dim = 512  # adattalo se sai dim esatta

        self.embedding_net = nn.Sequential(
            nn.Linear(rel_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # normalizzazione embedding utile per contrastive/distillation
        self.norm_embeddings = True

    def forward(self, x, return_embedding=False):
        """
        x: input features expected by rel_module
        return_embedding: se True ritorna anche embedding (prima della classificazione)
        """
        # pass through relative representation
        rel_feat = self.rel(x)  # assume torch tensor [B, rel_out_dim]
        emb = self.embedding_net(rel_feat)
        if self.norm_embeddings:
            emb = F.normalize(emb, dim=1)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits

# -------------------------
# NT-Xent (InfoNCE) loss per coppie (emb_new, emb_old)
# -------------------------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z_i, z_j, mask_positive=None):
        """
        z_i, z_j: [B, D] embeddings (positive pairs are (z_i[k], z_j[k]))
        mask_positive: optional boolean mask [B] to select which pairs count (e.g., only real samples)
        Returns scalar loss (mean over selected positives)
        """
        assert z_i.size() == z_j.size()
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # [2B,2B]
        sim = sim / self.temperature

        B = z_i.size(0)
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels, labels], dim=0)

        # mask to remove self-similarity
        diag_mask = torch.eye(2*B, device=z.device).bool()
        sim_masked = sim.masked_fill(diag_mask, -9e15)

        # positives: for index k (0..B-1), positive is k+B; for k+B positive is k
        positives = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z.device)

        # optionally select subset of positives by mask_positive
        if mask_positive is not None:
            # mask_positive shape [B], expand to [2B] by concatenation
            mp = torch.cat([mask_positive, mask_positive], dim=0)
        else:
            mp = torch.ones(2*B, dtype=torch.bool, device=z.device)

        # compute log-softmax over rows
        logprob = F.log_softmax(sim_masked, dim=1)
        loss_vec = -logprob[torch.arange(2*B, device=z.device), positives]  # negative log prob of positive
        # apply mask and average
        if mp.sum() == 0:
            return torch.tensor(0., device=z.device, requires_grad=True)
        loss = (loss_vec[mp]).mean()
        return loss

# -------------------------
# Compactness loss (intra-class variance)
# -------------------------
def intra_class_compactness_loss(embeddings, labels):
    """
    embeddings: [B, D] normalized or not
    labels: [B] int64
    Simple version: for each class compute mean embedding and penalize squared distance of samples to class mean.
    """
    device = embeddings.device
    loss = torch.tensor(0., device=device)
    classes = torch.unique(labels)
    total = 0
    for c in classes:
        mask = (labels == c)
        if mask.sum() < 2:
            continue
        emb_c = embeddings[mask]
        mean_c = emb_c.mean(dim=0, keepdim=True)
        loss += ((emb_c - mean_c).pow(2).sum(dim=1)).mean()
        total += 1
    if total == 0:
        return torch.tensor(0., device=device)
    return loss / total

# -------------------------
# Utility: freeze a model (for old model)
# -------------------------
def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model

def train_one_epoch_with_distill(classifier: nn.Module,
                                 dataloader,
                                 criterion_ce,
                                 optimizer,
                                 device,
                                 old_model=None,
                                 nt_xent_loss=None,
                                 lambda_contrast=1.0,
                                 lambda_compact=0.1,
                                 only_on_real_for_contrast=True,
                                 rel_module=None):
    """
    classifier: il nuovo modello (RelClassifierWithEmbedding)
    old_model: modello vecchio congelato (stesso architettura embedding)
    nt_xent_loss: istanza di NTXentLoss
    lambda_contrast, lambda_compact: pesi dei termini addizionali
    only_on_real_for_contrast: se True, calcola contrastive solo su batch samples con label==0 (real)
    rel_module: opzionale, aiuta se serve mappare input -> rel rep (ma la tua classifier già lo contiene)
    """
    classifier.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for batch in dataloader:
        feats, labels = batch
        feats = feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, emb = classifier(feats, return_embedding=True)  # nuovo -> logits + embedding

        loss_ce = criterion_ce(logits, labels)

        # contrastive/distillation loss (se abbiamo old_model)
        loss_contrast = torch.tensor(0., device=device)
        if old_model is not None and nt_xent_loss is not None:
            with torch.no_grad():
                _, emb_old = old_model(feats, return_embedding=True)
            if only_on_real_for_contrast:
                real_mask = (labels == 0)
                if real_mask.sum() > 0:
                    emb_new_sel = emb[real_mask]
                    emb_old_sel = emb_old[real_mask]
                    mp = torch.ones(emb_new_sel.size(0), dtype=torch.bool, device=device)
                    loss_contrast = nt_xent_loss(emb_new_sel, emb_old_sel, mask_positive=mp)
            else:
                loss_contrast = nt_xent_loss(emb, emb_old, mask_positive=None)

        # compactness
        loss_compact = intra_class_compactness_loss(emb, labels)

        loss = loss_ce + lambda_contrast * loss_contrast + lambda_compact * loss_compact
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        running_acc += (preds == labels).sum().item()
        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_acc / total
    return epoch_loss, epoch_acc


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

from sklearn.metrics import confusion_matrix, classification_report

import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

def evaluate_mc(model, dataloader, criterion, device, rel_module=None, 
                test_name="test_set", save_dir="./logs", num_total_classes=7):
    """
    Robust evaluation for both binary and multiclass models.
    - Works even if model was trained on a subset of all labels (e.g. {0,2}).
    - Saves confusion matrix, per-class report, and optional relative feature stats.
    """

    model.eval()
    val_loss, val_acc, num_samples = 0.0, 0.0, 0
    all_preds, all_labels, all_feats = [], [], []

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc=f"Evaluating {test_name}"):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            
            # Detect number of classes in model
            num_model_classes = outputs.shape[1]

            # If model trained on only 2 classes (binary real/fake)
            # remap unseen labels (1,3,4,5,6) to "fake" (1)
            if num_model_classes == 2:
                label_mapping = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
                labels = torch.tensor([label_mapping[int(l)] for l in labels.cpu()]).to(device)

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
                rel_feat = rel_module(features).cpu()
                all_feats.append(rel_feat)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Save relative feature statistics (if any)
    if rel_module is not None and len(all_feats) > 0:
        all_feats = torch.cat(all_feats)  # [N, num_anchors]
        mean_sim = all_feats.mean(dim=1).cpu().numpy()
        std_sim  = all_feats.std(dim=1).cpu().numpy()
        max_sim  = torch.max(all_feats, dim=1).values.cpu().numpy()
        min_sim  = torch.min(all_feats, dim=1).values.cpu().numpy()

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

    # ---- Confusion Matrix ----
    # For visualization: show all 7 classes (0–6)
    cm_labels = np.arange(num_total_classes)
    cm = confusion_matrix(all_labels, all_preds, labels=cm_labels)

    plt.figure(figsize=(7,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, alpha=0.85)
    plt.colorbar()
    plt.xticks(np.arange(num_total_classes), [f"{i}" for i in cm_labels], rotation=45)
    plt.yticks(np.arange(num_total_classes), [f"{i}" for i in cm_labels])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {test_name}")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            plt.text(j, i, str(count),
                     ha="center", va="center",
                     color="white" if count > thresh else "black",
                     fontsize=9, fontweight="bold")

    plt.tight_layout()
    cm_path = os.path.join(save_dir, f"confusion_matrix_{test_name}.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # ---- Classification Report ----
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
    print(report)
    with open(os.path.join(save_dir, f"classification_report_{test_name}.txt"), "w") as f:
        f.write(report)

    # ---- Averages ----
    avg_loss = val_loss / num_samples
    avg_acc = val_acc / num_samples
    print(f"[{test_name}] - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    return avg_loss, avg_acc




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

import torchvision.transforms as T

data_transforms = {
    'image': T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}