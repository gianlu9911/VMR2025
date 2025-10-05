#!/usr/bin/env python3
import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from src.utils import extract_and_save_features, train_one_epoch, plot_features_with_anchors

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

# ---------------------------------------------
# Fine-tuning pipeline (main)
# ---------------------------------------------
def fine_tune(backbone_name, fine_tuning_on, seed, batch_size, num_workers, num_points, num_anchors, saved_accuracy_path,
    checkpoint_path=None, force_recompute_features=False):
    device = get_device(args.device)
    print(f"Using device: {device}")

    backbone_name = backbone_name or args.backbone
    fine_tuning_on = fine_tuning_on or args.fine_tuning_on

    feature_dir = f"./feature_{backbone_name}"
    checkpoint_dir = checkpoint_path or f"./checkpoints_rel_{backbone_name}_on_{fine_tuning_on}"
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
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
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)

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
    num_train_samples = num_points
    if num_train_samples is not None and num_train_samples < len(feats_full):
        indices = torch.randperm(len(feats_full))[:num_train_samples]
        feats = feats_full[indices]
        labels = labels_full[indices]
    else:
        feats = feats_full
        labels = labels_full

    print(f"Using {len(feats)} training samples (real: {(labels==0).sum().item()}, fake: {(labels==1).sum().item()})")

    # Anchors (take from real training features only, allow sampling WITH replacement
    real_mask = labels == 0
    real_feats = feats[real_mask]
    if real_feats.size(0) == 0:
        raise RuntimeError("No real training features available to form anchors. Check your dataset / subsampling (num_train_samples).")

    if num_anchors is not None:
        # create a deterministic generator seeded by args.seed
        rng = torch.Generator().manual_seed(seed)
        num_requested = int(num_anchors)
        if num_requested > len(real_feats):
            print(f"[warning] Requested {num_requested} anchors but only {len(real_feats)} unique real samples available. Sampling WITH replacement from real set.")
            # sample with replacement
            idx = torch.randint(low=0, high=len(real_feats), size=(num_requested,), generator=rng)
        else:
            # sample without replacement
            idx = torch.randperm(len(real_feats), generator=rng)[:num_requested]
        anchors = real_feats[idx]
    else:
        anchors = real_feats

    if anchors.size(0) == 0:
        raise RuntimeError("Anchors is empty after selection. Aborting.")

    print(f"Using {anchors.size(0)} anchors for relative representation")

    rel_module = RelativeRepresentation(anchors.to(device))

    # Dataset + Sampler for training classifier
    feat_dataset = TensorDataset(feats, labels)
    sampler = BalancedBatchSampler(labels, batch_size=batch_size)
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
        "real_vs_sdv1_4": RealSynthethicDataloader(real_dir, IMAGE_DIR['sdv1_4'], split='test_set'),
        "real_vs_stylegan3": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan3'], split='test_set'),
        "real_vs_styleganxl": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan_xl'], split='test_set'),
    }

    test_results = {}
    for name, dataset in dataloaders_test.items():
        feat_file_test = os.path.join(feature_dir, f"test_{name}_features.pt")
        print(f"Preparing test features for {name} -> {feat_file_test}")

        if args.force_recompute_features or not os.path.exists(feat_file_test):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers)
            feats_test, labels_test, feat_time = extract_and_save_features(backbone, loader,
                                                                          feat_file_test, device, split='test_set')
        else:
            data = torch.load(feat_file_test)
            feats_test, labels_test = data["features"], data["labels"]
            feat_time = 0.0
            print("Loaded cached test features")

        # Plot anchors + eval real + eval fake
        # anchors is available (torch tensor on device) -> move to cpu for plotting
        anchors_cpu = anchors.cpu()
        real_mask_eval = (labels_test == 0)
        fake_mask_eval = (labels_test == 1)
        real_feats_eval = feats_test[real_mask_eval]
        fake_feats_eval = feats_test[fake_mask_eval]

        #plot_save_path = os.path.join("./logs", f"feature_plot_{name}_{args.plot_method}.png")
        os.makedirs("./logs/confusion_matrices", exist_ok=True)
        #plot_features_with_anchors(real_feats_eval, fake_feats_eval, anchors_cpu,method=args.plot_method, save_path=plot_save_path,subsample=args.plot_subsample)

        # Evaluate classifier on test features
        test_dataset = TensorDataset(feats_test, labels_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        loss, acc = evaluate(classifier, test_loader, criterion, device,
                             rel_module=rel_module, test_name=name, save_dir="./logs/confusion_matrices")
        test_results[name] = {"loss": loss, "acc": acc, "feat_time": feat_time}
        

    csv_path = saved_accuracy_path 
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Verifica se il file esiste e se è vuoto
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)

        # Scrivi l'intestazione solo se il file è vuoto
        if write_header:
            writer.writerow(["fine_tuning_on"] + list(test_results.keys()))

        # Scrivi la riga con le accuratezze
        writer.writerow([args.fine_tuning_on] + [test_results[name]["acc"] for name in test_results])


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
    parser.add_argument('--num_train_samples', type=int, default=100,
                        help="Number of training samples to use. If None, use all available samples.")
    parser.add_argument('--fine_tuning_on', type=str, default='stylegan1',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4'])
    parser.add_argument('--backbone', type=str, default='stylegan1',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4'])
    parser.add_argument('--num_anchors', type=int, default=100,
                        help="Exact number of real features to use as anchors; if greater than available reals, sampling with replacement is used.")
    #parser.add_argument('--plot_method', type=str, default='pca', choices=['pca', 'tsne'],help="Dimensionality reduction method for plotting (pca or tsne)")
    #parser.add_argument('--plot_subsample', type=int, default=100,help="Max number of eval points (real+fake) to plot (anchors always included)")
    parser.add_argument('--force_recompute_features', default=False, action='store_true',
                        help="Force recomputation of saved features")
    args = parser.parse_args()

    results = fine_tune(backbone_name=args.backbone, fine_tuning_on=args.fine_tuning_on, seed= args.seed, saved_accuracy_path="./logs/test_accuracies.csv",
        batch_size=args.batch_size, num_workers=args.num_workers, num_points=args.num_train_samples, num_anchors=args.num_anchors, 
        checkpoint_path=None, force_recompute_features=args.force_recompute_features)
    print("All test results:")
    for k, v in results.items():
        print(f" - {k}: loss={v['loss']:.4f}, acc={v['acc']:.4f}, feat_time={v['feat_time']:.2f}s")
