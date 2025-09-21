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

# PCA / t-SNE removed — we only plot raw first two feature dimensions
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
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
# (kept for backward compatibility, but we support 'none' to skip DR)
# ---------------------------------------------
def plot_features_with_anchors(real_feats, fake_feats, anchors, method="none", save_path=None, subsample=5000, random_state=42):
    """
    Simplified plotting utility — PCA / t-SNE removed by user request.
    This function plots the first two feature dimensions (no dimensionality reduction).
    real_feats, fake_feats, anchors: torch.Tensor or np.ndarray with shape [N, D]
    method: kept for API compatibility, must be 'none'
    """
    if method != "none":
        print(f"[plot] Dimensionality reduction disabled; ignoring method={method} and using first two feature dims.")

    # Convert to numpy
    if isinstance(real_feats, torch.Tensor): real_np = real_feats.cpu().numpy()
    else: real_np = np.array(real_feats)
    if isinstance(fake_feats, torch.Tensor): fake_np = fake_feats.cpu().numpy()
    else: fake_np = np.array(fake_feats)
    if isinstance(anchors, torch.Tensor): anchor_np = anchors.cpu().numpy()
    else: anchor_np = np.array(anchors)

    # Check dims and use first two coordinates
    def take_first2(arr):
        if arr.size == 0:
            return np.zeros((0,2))
        if arr.shape[1] < 2:
            raise ValueError("Features have less than 2 dimensions; cannot plot first two dims")
        return arr[:, :2]

    real_2 = take_first2(real_np) if real_np.size else np.zeros((0,2))
    fake_2 = take_first2(fake_np) if fake_np.size else np.zeros((0,2))
    anchor_2 = take_first2(anchor_np) if anchor_np.size else np.zeros((0,2))

    plt.figure(figsize=(8, 6))
    if real_2.size:
        plt.scatter(real_2[:, 0], real_2[:, 1], marker='o', s=8, alpha=0.5, label='Real (eval)')
    if fake_2.size:
        plt.scatter(fake_2[:, 0], fake_2[:, 1], marker='o', s=8, alpha=0.5, label='Fake (eval)')
    if anchor_2.size:
        plt.scatter(anchor_2[:, 0], anchor_2[:, 1], marker='*', s=120, edgecolor='k', linewidth=0.6, label='Anchors (train real)')

    plt.legend()
    plt.title(f"Feature visualization (raw first 2 dims)")
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
    # Simple confusion matrix plot (binary labels assumed)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_labels = ["Real", "Fake"] if cm.shape == (2,2) else list(range(cm.shape[0]))
    plt.xticks(np.arange(len(tick_labels)), tick_labels, rotation=45)
    plt.yticks(np.arange(len(tick_labels)), tick_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {test_name}")
    cm_path = os.path.join(save_dir, f"confusion_matrix_{test_name}.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    avg_loss = val_loss / num_samples
    avg_acc = val_acc / num_samples
    print(f"[{test_name}] - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    return avg_loss, avg_acc

# ---------------------------------------------
# Fine-tuning pipeline (main)
# ---------------------------------------------
def fine_tune(args, backbone_name=None, fine_tuning_on=None):
    device = get_device(args.device)
    print(f"Using device: {device}")

    backbone_name = backbone_name or args.backbone
    fine_tuning_on = fine_tuning_on or args.fine_tuning_on

    feature_dir = f"./feature_{backbone_name}"
    checkpoint_dir = "./checkpoint"
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Backbone
    backbone = load_pretrained_model(PRETRAINED_MODELS[backbone_name])
    backbone.resnet.fc = nn.Identity()
    backbone.to(device)
    backbone.eval()

    # Dataset directories
    real_dir = IMAGE_DIR['real']
    fake_dir = IMAGE_DIR[fine_tuning_on]

    dataset = RealSynthethicDataloader(real_dir, fake_dir)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    # Extract / load features
    full_train_feat_file = os.path.join(feature_dir, f"real_vs_{fine_tuning_on}_features.pt")
    if args.force_recompute_features or not os.path.exists(full_train_feat_file):
        print("Extracting full training features...")
        feats_full, labels_full, feat_time_full = extract_and_save_features(backbone, train_loader,
                                                                            full_train_feat_file, device)
    else:
        data = torch.load(full_train_feat_file)
        feats_full, labels_full = data["features"], data["labels"]
        feat_time_full = 0.0
        print("Loaded cached full training features")

    # Subsample training samples if requested
    num_train_samples = getattr(args, "num_train_samples", None)
    if num_train_samples is not None and num_train_samples < len(feats_full):
        indices = torch.randperm(len(feats_full))[:num_train_samples]
        feats = feats_full[indices]
        labels = labels_full[indices]
    else:
        feats = feats_full
        labels = labels_full

    print(f"Using {len(feats)} training samples (real: {(labels==0).sum().item()}, fake: {(labels==1).sum().item()})")

    # Anchors (take from real training features) -> FORCE using only 2 anchors
    real_mask = labels == 0
    real_feats = feats[real_mask]
    if args.num_anchors is not None and len(real_feats) > args.num_anchors:
        perm = torch.randperm(len(real_feats))[:args.num_anchors]
        anchors = real_feats[perm]
    else:
        anchors = real_feats

    # force to exactly 2 anchors (if available)
    if anchors.size(0) > 2:
        anchors = anchors[:2]
    print(f"Using {anchors.size(0)} anchors for relative representation (forced to 2)")
    rel_module = RelativeRepresentation(anchors.to(device))

    # Dataset + Sampler for training classifier
    feat_dataset = TensorDataset(feats, labels)
    sampler = BalancedBatchSampler(labels, batch_size=args.batch_size)
    feat_loader = DataLoader(feat_dataset, batch_sampler=sampler)

    # Classifier
    classifier = RelClassifier(rel_module, anchors.size(0), num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    # Training loop
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(classifier, feat_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.2f} minutes")

    checkpoint_path = os.path.join(checkpoint_dir,
                                   f'finetuned_rel_{backbone_name}_on_{fine_tuning_on}_samples{len(feats)}.pth')
    torch.save({'state_dict': classifier.state_dict()}, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Prepare test datasets
    dataloaders_test = {
        "real_vs_stylegan1": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan1'], split='test_set'),
        "real_vs_stylegan2": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan2'], split='test_set'),
        "real_vs_styleganxl": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan_xl'], split='test_set'),
        "real_vs_sdv1_4": RealSynthethicDataloader(real_dir, IMAGE_DIR['sdv1_4'], split='test_set')
    }

    # ensure raw-save folder exists
    raw_save_dir = "./log_2dim"
    os.makedirs(raw_save_dir, exist_ok=True)

    test_results = {}
    for name, dataset in dataloaders_test.items():
        feat_file_test = os.path.join(feature_dir, f"test_{name}_features.pt")
        print(f"Preparing test features for {name} -> {feat_file_test}")

        if args.force_recompute_features or not os.path.exists(feat_file_test):
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)
            feats_test, labels_test, feat_time = extract_and_save_features(backbone, loader,
                                                                          feat_file_test, device, split='test_set')
        else:
            data = torch.load(feat_file_test)
            feats_test, labels_test = data["features"], data["labels"]
            feat_time = 0.0
            print("Loaded cached test features")

        # anchors is available (torch tensor on device) -> move to cpu for saving
        anchors_cpu = anchors.cpu()
        real_mask_eval = (labels_test == 0)
        fake_mask_eval = (labels_test == 1)
        real_feats_eval = feats_test[real_mask_eval]
        fake_feats_eval = feats_test[fake_mask_eval]

        # SAVE RAW features and anchors WITHOUT dimensionality reduction into ./log_2dim
        # Save both torch .pt and numpy .npy for convenience
        base_name = name.replace('/', '_')
        anchor_pt = os.path.join(raw_save_dir, f"{base_name}_anchors.pt")
        real_pt = os.path.join(raw_save_dir, f"{base_name}_real_feats.pt")
        fake_pt = os.path.join(raw_save_dir, f"{base_name}_fake_feats.pt")

        torch.save(anchors_cpu, anchor_pt)
        torch.save(real_feats_eval.cpu(), real_pt)
        torch.save(fake_feats_eval.cpu(), fake_pt)

        # also save numpy for quick loading in other tools
        np.save(os.path.join(raw_save_dir, f"{base_name}_anchors.npy"), anchors_cpu.numpy())
        np.save(os.path.join(raw_save_dir, f"{base_name}_real_feats.npy"), real_feats_eval.cpu().numpy())
        np.save(os.path.join(raw_save_dir, f"{base_name}_fake_feats.npy"), fake_feats_eval.cpu().numpy())

        print(f"Saved raw anchors and features (no DR) to {raw_save_dir} for {name}")
        print(f"Anchors shape: {anchors_cpu.shape}, Real eval: {real_feats_eval.shape}, Fake eval: {fake_feats_eval.shape}")

        # Always produce a simple 2D scatter plot using the first two feature dimensions (no DR)
        try:
            def safe_plot_2d(X_real, X_fake, X_anchor, out_path):
                # X_* are torch or numpy arrays
                if isinstance(X_real, torch.Tensor): X_real_np = X_real.cpu().numpy()
                else: X_real_np = np.array(X_real)
                if isinstance(X_fake, torch.Tensor): X_fake_np = X_fake.cpu().numpy()
                else: X_fake_np = np.array(X_fake)
                if isinstance(X_anchor, torch.Tensor): X_anchor_np = X_anchor.cpu().numpy()
                else: X_anchor_np = np.array(X_anchor)

                # Check dims
                D = None
                for arr in (X_real_np, X_fake_np, X_anchor_np):
                    if arr.size:
                        D = arr.shape[1]
                        break
                if D is None:
                    print("[plot2d] No features available; skipping 2D plot.")
                    return False
                if D < 2:
                    print(f"[plot2d] Features have dimensionality {D} < 2; skipping 2D plot.")
                    return False

                # Take first two dims
                XR = X_real_np[:, :2] if X_real_np.size else np.zeros((0,2))
                XF = X_fake_np[:, :2] if X_fake_np.size else np.zeros((0,2))
                XA = X_anchor_np[:, :2] if X_anchor_np.size else np.zeros((0,2))

                plt.figure(figsize=(7,6))
                if XR.size:
                    plt.scatter(XR[:,0], XR[:,1], marker='o', s=8, alpha=0.5, label='Real (eval)')
                if XF.size:
                    plt.scatter(XF[:,0], XF[:,1], marker='o', s=8, alpha=0.5, label='Fake (eval)')
                if XA.size:
                    plt.scatter(XA[:,0], XA[:,1], marker='*', s=120, edgecolor='k', linewidth=0.6, label='Anchors (train real)')
                plt.legend()
                plt.title(f"Feature 2D plot (first 2 dims) - {base_name}")
                plt.tight_layout()
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                plt.savefig(out_path, dpi=200)
                plt.close()
                print(f"[plot2d] Saved 2D feature plot to {out_path}")
                return True

            plot2d_path = os.path.join(raw_save_dir, f"{base_name}_feature_2d.png")
            safe_plot_2d(real_feats_eval, fake_feats_eval, anchors_cpu, plot2d_path)
        except Exception as e:
            print(f"[plot2d] Failed to produce 2D plot: {e}")

        # If user still wants a DR-based visual, call plotting with chosen method
        if args.plot_method != 'none':
            plot_save_path = os.path.join("./logs", f"feature_plot_{name}_{args.plot_method}.png")
            os.makedirs("./logs", exist_ok=True)
            plot_features_with_anchors(real_feats_eval, fake_feats_eval, anchors_cpu,
                                       method=args.plot_method, save_path=plot_save_path,
                                       subsample=args.plot_subsample)

        # Evaluate classifier on test features
        test_dataset = TensorDataset(feats_test, labels_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        loss, acc = evaluate(classifier, test_loader, criterion, device,
                             rel_module=rel_module, test_name=name, save_dir="./logs")
        test_results[name] = {"loss": loss, "acc": acc, "feat_time": feat_time}

    return test_results

# ---------------------------------------------
# Main CLI
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help="Number of training samples to use. If None, use all available samples.")
    parser.add_argument('--fine_tuning_on', type=str, default='stylegan2',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4'])
    parser.add_argument('--backbone', type=str, default='stylegan1',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4'])
    parser.add_argument('--num_anchors', type=int, default=2,
                        help="Maximum number of real features to use as anchors (forced to 2 in this script)")
    parser.add_argument('--plot_method', type=str, default='none', choices=['none'],
                        help="Dimensionality: only 'none' is supported (plot raw first two dims)")
    parser.add_argument('--plot_subsample', type=int, default=5000,
                        help="Max number of eval points (real+fake) to plot (anchors always included)")
    parser.add_argument('--force_recompute_features', action='store_true',
                        help="Force recomputation of saved features")
    args = parser.parse_args()

    results = fine_tune(args)
    print("All test results:")
    for k, v in results.items():
        print(f" - {k}: loss={v['loss']:.4f}, acc={v['acc']:.4f}, feat_time={v['feat_time']:.2f}s")
