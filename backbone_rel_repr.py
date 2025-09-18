import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler
from tqdm import tqdm
import argparse
import time
import numpy as np
import csv
import warnings
warnings.filterwarnings("ignore")
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model


# ==========================
# Relative Representation
# ==========================
class RelativeRepresentation(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        anchors = anchors / anchors.norm(dim=1, keepdim=True)
        self.register_buffer("anchors", anchors)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
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


# -----------------------------
# Balanced Batch Sampler
# -----------------------------
class BalancedBatchSampler(Sampler):
    def __init__(self, labels: torch.Tensor, batch_size: int):
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = torch.unique(self.labels).numel()
        assert batch_size % self.num_classes == 0, "batch_size must be multiple of num_classes"

        self.class_indices = {cls.item(): torch.where(self.labels == cls)[0]
                              for cls in torch.unique(self.labels)}
        self.min_class_len = min(len(idxs) for idxs in self.class_indices.values())
        self.num_batches = self.min_class_len // (batch_size // self.num_classes)


    def __iter__(self):
        real = 0
        fake = 0
        per_class = self.batch_size // self.num_classes
        class_iters = {cls: iter(idxs[torch.randperm(len(idxs))])
                       for cls, idxs in self.class_indices.items()}

        for _ in range(self.num_batches):
            batch = []
            for cls, it in class_iters.items():
                selected = []
                for _ in range(per_class):
                    try:
                        selected.append(next(it).item())
                        if cls == 0:
                            real += 1
                        else:
                            fake += 1
                    except StopIteration:
                        return
                batch.extend(selected)
            yield batch
        print(f"Total real samples used: {real}, fake samples used: {fake}")

    def __len__(self):
        return self.num_batches


# -----------------------------
# Feature Extraction
# -----------------------------
def extract_and_save_features(backbone, dataloader, feature_path, device, split='train_set'):
    feats, labels = [], []
    print(f"Saving features to {feature_path}...")

    start_time = time.time()
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc=f"Extracting {os.path.basename(feature_path)}"):
            imgs = imgs.to(device)
            out = backbone(imgs)
            feats.append(out.cpu())
            labels.append(lbls)

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


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import torch

import pandas as pd

def evaluate(model, dataloader, criterion, device, rel_module=None, backbone_name="stylegan1", 
             test_name="test_set", save_dir="./logs"):
    """
    Evaluate model, compute confusion matrix, relative feature statistics, and save CSV.

    Args:
        model: Classifier model
        dataloader: DataLoader for features
        criterion: Loss function
        device: torch device
        rel_module: RelativeRepresentation module (optional, for cosine vectors)
        test_name: Name of the dataset (used for saving CM and CSV)
        save_dir: Directory to save confusion matrix and CSV
    """
    import os
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

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

    # Aggregate
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    if rel_module is not None:
        all_feats = torch.cat(all_feats)  # [num_samples, num_anchors]

        # Calcola statistiche per feature
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

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real","Fake"], yticklabels=["Real","Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {test_name}")
    cm_path = os.path.join(save_dir, f"confusion_matrix_{test_name}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    avg_loss = val_loss / num_samples
    avg_acc = val_acc / num_samples
    print(f"[{test_name}] - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    return avg_loss, avg_acc




# -----------------------------
# Fine-Tuning Pipeline
# -----------------------------
def fine_tune(args, backbone_name=None, fine_tuning_on=None):
    # ---------------- Device ----------------
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    backbone_name = backbone_name or args.backbone
    fine_tuning_on = fine_tuning_on or args.fine_tuning_on

    # ---------------- Directories ----------------
    feature_dir = f"./feature_{backbone_name}"
    checkpoint_dir = "./checkpoint"
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---------------- Seeds ----------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---------------- Backbone ----------------
    backbone = load_pretrained_model(PRETRAINED_MODELS[backbone_name])
    backbone.resnet.fc = nn.Identity()
    backbone.to(device)
    backbone.eval()

    # ---------------- Training dataset ----------------
    real_dir = IMAGE_DIR['real']
    #real_dir = os.path.join(real_dir, 'train_set')
    fake_dir = IMAGE_DIR[fine_tuning_on]
    #fake_dir = os.path.join(fake_dir, 'train_set')

    #print(f"Number of real images: {len(os.listdir(real_dir))}, fake images: {len(os.listdir(fake_dir))}")

    dataset = RealSynthethicDataloader(real_dir, fake_dir)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    # ---------------- Extract features ----------------
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

    # ---------------- Subsample by number of samples ----------------
    num_train_samples = getattr(args, "num_train_samples", None)
    if num_train_samples is not None and num_train_samples < len(feats_full):
        indices = torch.randperm(len(feats_full))[:num_train_samples]
        feats = feats_full[indices]
        labels = labels_full[indices]
    else:
        feats = feats_full
        labels = labels_full

    print(f"Using {len(feats)} training samples")
    print(f"with real: {int((labels==0).sum())}, fake: {int((labels==1).sum())}")

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
    sampler = BalancedBatchSampler(labels, batch_size=args.batch_size)
    feat_loader = DataLoader(feat_dataset, batch_sampler=sampler)

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

    # ---------------- Evaluate Test Sets ----------------
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
                                num_workers=args.num_workers)
            feats_test, labels_test, feat_time = extract_and_save_features(backbone, loader,
                                                                          feat_file_test, device, split='test_set')

        test_dataset = TensorDataset(feats_test, labels_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        loss, acc = evaluate(classifier, test_loader, criterion, device,
                             rel_module=rel_module, backbone_name=backbone_name, test_name=name)
        test_results[name] = {"loss": loss, "acc": acc, "feat_time": feat_time}

    return test_results



# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help="Numero di campioni di addestramento da utilizzare. Se None, usa tutti i campioni disponibili.")
    parser.add_argument('--fine_tuning_on', type=str, default='stylegan2',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4'])
    parser.add_argument('--backbone', type=str, default='stylegan1',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4'])
    parser.add_argument('--num_anchors', type=int, default=1000,
                        help="Numero massimo di feature reali da usare come ancore")
    args = parser.parse_args()
    fine_tune(args)
