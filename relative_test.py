import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse
import time
import numpy as np

from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model


# -----------------------------
# Feature Extraction
# -----------------------------
def extract_and_save_features(backbone, dataloader, feature_dir, device):
    os.makedirs(feature_dir, exist_ok=True)

    feats, labels = [], []
    start_time = time.time()
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(device)
            out = backbone(imgs)
            feats.append(out.cpu())
            labels.append(lbls)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    total_time = time.time() - start_time
    print(f"Feature extraction completed in {total_time/60:.2f} minutes.")

    torch.save({"features": feats, "labels": labels}, os.path.join(feature_dir, "train_features.pt"))
    print(f"Features saved to {feature_dir}/train_features.pt")


# -----------------------------
# Utilities
# -----------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, print_freq=10):
    model.train()
    running_loss, running_acc, num_samples = 0.0, 0.0, 0

    for i, (features, labels) in enumerate(dataloader):
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

        #if i % print_freq == 0:
            #print(f"[Batch {i}] loss={running_loss/num_samples:.4f}, acc={running_acc/num_samples:.4f}")

    return running_loss / num_samples, running_acc / num_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss, val_acc, num_samples = 0.0, 0.0, 0
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
    return val_loss / num_samples, val_acc / num_samples


# -----------------------------
# Fine-tuning pipeline
# -----------------------------
def fine_tune(args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    feature_dir = "./feature_stylegan1"
    os.makedirs(feature_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load pretrained backbone
    backbone = load_pretrained_model(PRETRAINED_MODELS['stylegan1'])
    backbone.resnet.fc = nn.Identity()  # remove last classifier
    backbone.to(device)
    backbone.eval()

    real_dir = IMAGE_DIR['real']
    fake_st2_dir = IMAGE_DIR['stylegan2']
    dataset = RealSynthethicDataloader(real_dir, fake_st2_dir)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Extract or load features
    feat_file = os.path.join(feature_dir, "train_features.pt")
    if not os.path.exists(feat_file):
        print("Extracting features from backbone...")
        extract_and_save_features(backbone, train_loader, feature_dir, device)
    else:
        print("Found cached features, skipping extraction.")

    data = torch.load(feat_file)
    feats, labels = data["features"], data["labels"]
    feat_dataset = TensorDataset(feats, labels)
    feat_loader = DataLoader(feat_dataset, batch_size=args.batch_size, shuffle=True)

    # Define linear classifier
    classifier = nn.Linear(feats.size(1), 2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    # Training loop
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(classifier, feat_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes.")
    save_path = f'./checkpoint/finetuned_stylegan1_on_stylegan2_relative.pth'
    torch.save({'state_dict': classifier.state_dict()}, save_path)
    print(f"Model saved to {save_path}")


    # ----------------------
    # Evaluate on all test sets
    # ----------------------
    dataloaders_test = {
        "real vs stylegan1": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan1'], split='test_set'),
        "real vs stylegan2": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan2'], split='test_set'),
        "real vs styleganxl": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan_xl'], split='test_set'),
        "real vs sdv1_4": RealSynthethicDataloader(real_dir, IMAGE_DIR['sdv1_4'], split='test_set')
    }

    print("\nEvaluating on all test sets:")
    for name, dataset in dataloaders_test.items():
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # Extract features
        feats_test, labels_test = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                out = backbone(imgs)
                feats_test.append(out.cpu())
                labels_test.append(lbls)

        feats_test = torch.cat(feats_test, dim=0)
        labels_test = torch.cat(labels_test, dim=0)

        test_dataset = TensorDataset(feats_test, labels_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        test_loss, test_acc = evaluate(classifier, test_loader, criterion, device)
        print(f"[{name}] - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    fine_tune(args)
