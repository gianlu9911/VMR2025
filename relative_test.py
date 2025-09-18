# finetune_visualize.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import pandas as pd

# Project imports (assumes these exist in your project)
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model

# -----------------------------
# Balanced Batch Sampler (robust)
# -----------------------------
class BalancedBatchSampler(Sampler):
    """
    Yields balanced batches with equal number of samples per class.
    If classes differ in size, this sampler will oversample the smaller classes
    (reshuffling them when exhausted) so each epoch yields `num_batches`.
    """
    def __init__(self, labels: torch.Tensor, batch_size: int, oversample: bool = True):
        self.labels = labels.cpu()
        self.batch_size = int(batch_size)
        self.num_classes = int(torch.unique(self.labels).numel())
        assert self.batch_size % self.num_classes == 0, "batch_size must be multiple of num_classes"
        self.per_class = self.batch_size // self.num_classes
        self.class_indices = {int(cls.item()): torch.where(self.labels == cls)[0].tolist()
                              for cls in torch.unique(self.labels)}
        lengths = [len(idxs) for idxs in self.class_indices.values()]
        if oversample:
            self.num_batches = int(max(lengths) // self.per_class)
            if max(lengths) % self.per_class != 0:
                self.num_batches += 1
        else:
            self.num_batches = int(min(lengths) // self.per_class)

    def __iter__(self):
        # Start with a randomized list per class
        permuted = {cls: np.random.permutation(idxs).tolist() for cls, idxs in self.class_indices.items()}
        cursors = {cls: 0 for cls in permuted}
        for _ in range(self.num_batches):
            batch = []
            for cls, idx_list in permuted.items():
                for _ in range(self.per_class):
                    pos = cursors[cls]
                    if pos >= len(idx_list):
                        # reshuffle to oversample
                        idx_list = permuted[cls] = np.random.permutation(self.class_indices[cls]).tolist()
                        cursors[cls] = 0
                        pos = 0
                    batch.append(int(idx_list[pos]))
                    cursors[cls] += 1
            yield batch

    def __len__(self):
        return int(self.num_batches)


# -----------------------------
# Safer backbone head removal
# -----------------------------
def remove_backbone_head(backbone):
    """
    Try common attribute names to remove classification head.
    Returns True if changed, False otherwise.
    """
    try:
        # common resnet-like architectures from some stylegan wrappers
        if hasattr(backbone, "resnet") and hasattr(backbone.resnet, "fc"):
            backbone.resnet.fc = nn.Identity()
            return True
        if hasattr(backbone, "fc"):
            backbone.fc = nn.Identity()
            return True
        if hasattr(backbone, "classifier"):
            backbone.classifier = nn.Identity()
            return True
    except Exception as e:
        print("Warning while attempting to remove backbone head:", e)
    return False


# ==========================
# Relative Representation
# ==========================
class RelativeRepresentation(nn.Module):
    def __init__(self, anchors: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        # anchors: [M, D] where D is backbone embedding dim
        norms = anchors.norm(dim=1, keepdim=True).clamp_min(eps)
        anchors = anchors / norms
        self.register_buffer("anchors", anchors)  # [M, D]

    def forward(self, x: torch.Tensor):
        # x: [N, D]
        x = F.normalize(x, p=2, dim=1)
        # return similarity matrix: [N, M] (cosine similarities to anchors)
        return torch.matmul(x, self.anchors.T)


class RelClassifier(nn.Module):
    def __init__(self, rel_module: RelativeRepresentation, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.rel_module = rel_module
        self.classifier = nn.Linear(in_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        rel_x = self.rel_module(x)  # raw feats -> relative feats [batch, M]
        return self.classifier(rel_x)


# -----------------------------
# Feature Extraction
# -----------------------------
def extract_and_save_features(backbone, dataloader, feature_path, device, split='train_set'):
    feats, labels = [], []
    print(f"Saving features to {feature_path}...")

    start_time = time.time()
    backbone.eval()
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc=f"Extracting {os.path.basename(feature_path)}"):
            imgs = imgs.to(device)
            out = backbone(imgs)
            feats.append(out.detach().cpu())
            labels.append(lbls.detach().cpu())

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    total_time = time.time() - start_time

    torch.save({"features": feats, "labels": labels}, feature_path)
    print(f"Features saved to {feature_path} (time: {total_time/60:.2f} min)")
    return feats, labels, total_time


# -----------------------------
# Training and Evaluation
# -----------------------------
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


def evaluate(model, dataloader, criterion, device, rel_module=None, test_name="test_set", save_dir="./logs", class_names=None):
    """
    Evaluate model, compute confusion matrix, relative feature statistics, and save CSV + CM.
    """
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

    # Relative stats
    if rel_module is not None and len(all_feats) > 0:
        all_feats = torch.cat(all_feats)
        mean_sim = all_feats.mean(dim=1).numpy()
        std_sim = all_feats.std(dim=1).numpy()
        max_sim = all_feats.max(dim=1)[0].numpy()
        min_sim = all_feats.min(dim=1)[0].numpy()

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
    elif rel_module is not None:
        print("No relative features were collected; skipping stats CSV.")

    # Confusion matrix for N classes
    cm = confusion_matrix(all_labels, all_preds)
    num_classes = cm.shape[0]
    if class_names is None:
        # default for binary: Real/Fake, else numeric labels
        class_names = ["Real", "Fake"] if num_classes == 2 else [str(i) for i in range(num_classes)]

    plt.figure(figsize=(4 + num_classes, 3 + num_classes * 0.25))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {test_name}")
    cm_path = os.path.join(save_dir, f"confusion_matrix_{test_name}.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    avg_loss = val_loss / num_samples
    avg_acc = val_acc / num_samples
    print(f"[{test_name}] - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    return avg_loss, avg_acc


# -----------------------------
# Visualization utilities
# -----------------------------
def visualize_features_and_anchors(sim_features: torch.Tensor,
                                   sim_anchors: torch.Tensor,
                                   labels: torch.Tensor,
                                   save_dir: str = "./viz",
                                   prefix: str = "exp",
                                   tsne_perplexity: int = 30,
                                   save_pngs: bool = True):
    """
    Create:
      1) t-SNE of similarity vectors (samples + anchors)
      2) Heatmap of sample x anchor similarities (samples sorted by label)
      3) Per-anchor histogram comparing real vs fake similarities (top K anchors optional)

    Args:
        sim_features: [N, M] torch tensor (similarities of samples to anchors)
        sim_anchors: [M, M] torch tensor (anchors similarities to anchors)
        labels: [N] torch tensor (0/1 or multi)
    Returns:
        dict with paths to saved images
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    # Ensure CPU numpy
    sim_features = sim_features.cpu().numpy()
    sim_anchors = sim_anchors.cpu().numpy()
    labels_np = labels.cpu().numpy()
    N, M = sim_features.shape

    # -------------------------
    # 1) t-SNE scatter
    # -------------------------
    combined = np.vstack([sim_features, sim_anchors])  # [N+M, M]
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, init='random', random_state=42)
    combined_2d = tsne.fit_transform(combined)

    plt.figure(figsize=(9, 7))
    # samples
    scatter = plt.scatter(combined_2d[:N, 0], combined_2d[:N, 1],
                          c=labels_np, cmap="coolwarm", alpha=0.6, s=25)
    # anchors
    anchor_idx = np.arange(N, N + M)
    plt.scatter(combined_2d[anchor_idx, 0], combined_2d[anchor_idx, 1],
                c="gold", marker="*", edgecolors="black", s=180, label="anchors")

    # legend for the classes
    try:
        handles, _ = scatter.legend_elements()
        class_labels = ["Real", "Fake"] if len(np.unique(labels_np)) == 2 else [str(x) for x in np.unique(labels_np)]
        plt.legend(loc="best")
    except Exception:
        pass

    plt.title(f"t-SNE (similarity vectors) - {prefix}")
    tsne_path = os.path.join(save_dir, f"{prefix}_tsne.png")
    if save_pngs:
        plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
        plt.close()
    results['tsne'] = tsne_path

    # -------------------------
    # 2) Heatmap sample x anchor (sorted by label)
    # -------------------------
    order = np.argsort(labels_np)
    sorted_feats = sim_features[order]
    sorted_labels = labels_np[order]

    plt.figure(figsize=(12, 6))
    sns.heatmap(sorted_feats, cmap="viridis", cbar=True, xticklabels=False, yticklabels=False)
    plt.title(f"Sample x Anchor similarities (sorted by label) - {prefix}")
    heatmap_path = os.path.join(save_dir, f"{prefix}_heatmap.png")
    if save_pngs:
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
    results['heatmap'] = heatmap_path

    # -------------------------
    # 3) Per-anchor histograms (real vs fake)
    #    We'll plot top-K anchors by variance across samples (most informative)
    # -------------------------
    K = min(12, M)  # number of anchors to visualize
    anchor_variances = sim_features.var(axis=0)
    topk_idx = np.argsort(-anchor_variances)[:K]

    fig, axs = plt.subplots(nrows=(K + 3) // 4, ncols=4, figsize=(16, 3 * ((K + 3) // 4)))
    axs = axs.flatten()
    for i, aid in enumerate(topk_idx):
        real_vals = sim_features[labels_np == 0, aid]
        fake_vals = sim_features[labels_np == 1, aid] if np.any(labels_np == 1) else np.array([])
        ax = axs[i]
        ax.hist(real_vals, bins=30, alpha=0.6, label="Real", density=True)
        if fake_vals.size:
            ax.hist(fake_vals, bins=30, alpha=0.6, label="Fake", density=True)
        ax.set_title(f"Anchor {aid} (var={anchor_variances[aid]:.3f})")
        ax.legend(loc="upper right")
    # hide empty axes
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    per_anchor_path = os.path.join(save_dir, f"{prefix}_per_anchor_hist.png")
    if save_pngs:
        plt.tight_layout()
        plt.savefig(per_anchor_path, dpi=150, bbox_inches='tight')
        plt.close()
    results['per_anchor_hist'] = per_anchor_path

    print(f"[Saved visualizations] tsne: {results['tsne']}, heatmap: {results['heatmap']}, per-anchor: {results['per_anchor_hist']}")
    return results


# -----------------------------
# Fine-Tuning Pipeline
# -----------------------------
def fine_tune(args, backbone_name=None, fine_tuning_on=None):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    backbone_name = backbone_name or args.backbone
    fine_tuning_on = fine_tuning_on or args.fine_tuning_on

    feature_dir = f"./feature_{backbone_name}"
    checkpoint_dir = "./checkpoint"
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---------------- Backbone ----------------
    backbone = load_pretrained_model(PRETRAINED_MODELS[backbone_name])
    removed = remove_backbone_head(backbone)
    if not removed:
        print("Warning: couldn't safely remove backbone head; you may need to set it manually.")
    backbone.to(device)
    backbone.eval()

    # ---------------- Dataset ----------------
    real_dir = IMAGE_DIR['real']
    fake_dir = IMAGE_DIR[fine_tuning_on]

    dataset = RealSynthethicDataloader(real_dir, fake_dir)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    # ---------------- Extract or load full train features ----------------
    full_train_feat_file = os.path.join(feature_dir, f"real_vs_{fine_tuning_on}_features.pt")
    if not os.path.exists(full_train_feat_file):
        print("Extracting full training features...")
        feats_full, labels_full, feat_time_full = extract_and_save_features(backbone, train_loader,
                                                                            full_train_feat_file, device)
    else:
        data = torch.load(full_train_feat_file)
        feats_full, labels_full = data["features"], data["labels"]
        feat_time_full = 0.0
        print("Loaded cached full training features")

    # ---------------- Subsample if requested ----------------
    num_train_samples = getattr(args, "num_train_samples", None)
    if num_train_samples is not None and num_train_samples < len(feats_full):
        indices = torch.randperm(len(feats_full))[:num_train_samples]
        feats = feats_full[indices]
        labels = labels_full[indices]
    else:
        feats = feats_full
        labels = labels_full

    print(f"Using {len(feats)} training samples")
    print(f"with real: {int((labels == 0).sum())}, fake: {int((labels == 1).sum())}")

    # ---------------- Anchors ----------------
    real_mask = labels == 0
    real_feats = feats[real_mask]
    if args.num_anchors is not None and len(real_feats) > args.num_anchors:
        perm = torch.randperm(len(real_feats))[:args.num_anchors]
        anchors = real_feats[perm]
    else:
        anchors = real_feats
    print(f"Using {anchors.size(0)} anchors for relative representation")
    rel_module = RelativeRepresentation(anchors.to(device))

    # ---------------- Dataset + Sampler ----------------
    feat_dataset = TensorDataset(feats, labels)
    sampler = BalancedBatchSampler(labels, batch_size=args.batch_size, oversample=True)
    feat_loader = DataLoader(feat_dataset, batch_sampler=sampler, pin_memory=True)

    # ---------------- Classifier ----------------
    classifier = RelClassifier(rel_module, anchors.size(0), num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    # ---------------- Training Loop ----------------
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(classifier, feat_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.2f} minutes")

    # ---------------- Save Checkpoint ----------------
    checkpoint_path = os.path.join(checkpoint_dir,
                                   f'finetuned_rel_{backbone_name}_on_{fine_tuning_on}_samples{len(feats)}.pth')
    torch.save({'state_dict': classifier.state_dict()}, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # ---------------- Evaluate + Visualize Test Sets ----------------
    dataloaders_test = {
        "real_vs_stylegan1": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan1'], split='test_set'),
        "real_vs_stylegan2": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan2'], split='test_set'),
        "real_vs_styleganxl": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan_xl'], split='test_set'),
        "real_vs_sdv1_4": RealSynthethicDataloader(real_dir, IMAGE_DIR['sdv1_4'], split='test_set')
    }

    test_results = {}
    for name, dataset in dataloaders_test.items():
        feat_file_test = os.path.join(feature_dir, f"test_{name}_features.pt")
        print(f"Saving test features to {feat_file_test}...")

        if os.path.exists(feat_file_test):
            data = torch.load(feat_file_test)
            feats_test, labels_test = data["features"], data["labels"]
            feat_time = 0.0
        else:
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            feats_test, labels_test, feat_time = extract_and_save_features(backbone, loader,
                                                                          feat_file_test, device, split='test_set')

        test_dataset = TensorDataset(feats_test, labels_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        loss, acc = evaluate(classifier, test_loader, criterion, device,
                             rel_module=rel_module, test_name=name, save_dir="./logs")
        test_results[name] = {"loss": loss, "acc": acc, "feat_time": feat_time}

        # ---------------- Visualization: compute similarity vectors and plot
        # compute sim vectors for test samples and anchors (anchors -> anchors similarity)
        with torch.no_grad():
            sim_test = rel_module(feats_test.to(device)).cpu()  # [N_test, M]
            sim_anchors = rel_module(anchors.to(device)).cpu()  # [M, M]
        viz_dir = "./viz"
        _ = visualize_features_and_anchors(sim_features=sim_test, sim_anchors=sim_anchors,
                                          labels=labels_test, save_dir=viz_dir, prefix=f"{name}")

    return test_results


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0', help="torch device index (logical after CUDA_VISIBLE_DEVICES).")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help="Number of training samples to use. If None, uses all.")
    parser.add_argument('--fine_tuning_on', type=str, default='stylegan2',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4'])
    parser.add_argument('--backbone', type=str, default='stylegan1',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4'])
    parser.add_argument('--num_anchors', type=int, default=1000,
                        help="Maximum number of real features to use as anchors")
    args = parser.parse_args()
    fine_tune(args)
