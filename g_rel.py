#!/usr/bin/env python3
import os
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# === Project imports - adjust these to your repo layout ===
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model
from src.utils import get_device, BalancedBatchSampler, RelativeRepresentation, RelClassifier, extract_and_save_features, evaluate, train_one_epoch, plot_features_with_anchors

# ---------------------------------------------
# Fine-tuning pipeline (main)
# ---------------------------------------------

def fine_tune(
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = '0',
    epochs: int = 300,
    lr: float = 1e-3,
    seed: int = 42,
    num_train_samples = 100,
    fine_tuning_on: str = 'stylegan2',
    backbone: str = 'stylegan1',
    num_anchors: int = 5000,
    plot_method: str = 'pca',
    plot_subsample: int = 5000,
    force_recompute_features: bool = False,
    eval_csv_path: str = None,
):
    """Fine-tune relative-representation classifier.

    This function no longer depends on an argparse.Namespace. All configuration
    is provided via keyword arguments so it can be called directly from code.

    If `eval_csv_path` is provided (or left as None), evaluation results will be
    appended to ./logs/eval_results.csv with the header:

    fine_tuning_on,real_vs_stylegan1,real_vs_stylegan2,real_vs_sdv1_4,real_vs_stylegan3,real_vs_styleganxl

    Each call to fine_tune appends a single row where the first column is the
    `fine_tuning_on` dataset name and the remaining columns are the test
    accuracies (formatted with 5 decimal places). Missing test entries are
    left empty.

    Returns:
        dict: test_results mapping dataset name -> {loss, acc, feat_time}
    """
    device = get_device(device)
    print(f"Using device: {device}")

    feature_dir = f"./feature_{backbone}"
    checkpoint_dir = "./checkpoint"
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Backbone
    backbone_net = load_pretrained_model(PRETRAINED_MODELS[backbone])
    backbone_net.resnet.fc = nn.Identity()
    backbone_net.to(device)
    backbone_net.eval()

    # Dataset directories
    real_dir = IMAGE_DIR['real']
    fake_dir = IMAGE_DIR[fine_tuning_on]

    dataset = RealSynthethicDataloader(real_dir, fake_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)

    # Extract / load features
    full_train_feat_file = os.path.join(feature_dir, f"real_vs_{fine_tuning_on}_features.pt")
    if force_recompute_features or not os.path.exists(full_train_feat_file):
        print("Extracting full training features...")
        feats_full, labels_full, feat_time_full = extract_and_save_features(backbone_net, train_loader,
                                                                            full_train_feat_file, device)
    else:
        data = torch.load(full_train_feat_file)
        feats_full, labels_full = data["features"], data["labels"]
        feat_time_full = 0.0
        print("Loaded cached full training features")

    # Subsample training samples if requested
    if num_train_samples is None:
        num_train_samples_val = None
    else:
        num_train_samples_val = int(num_train_samples)

    if num_train_samples_val is not None and num_train_samples_val < len(feats_full):
        indices = torch.randperm(len(feats_full))[:num_train_samples_val]
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
        # create a deterministic generator seeded by seed
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
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(classifier, feat_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.2f} minutes")

    checkpoint_path = os.path.join(checkpoint_dir,
                                   f'finetuned_rel_{backbone}_on_{fine_tuning_on}_samples{len(feats)}.pth')
    torch.save({'state_dict': classifier.state_dict()}, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Prepare test datasets
    dataloaders_test = {
        "real_vs_stylegan1": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan1'], split='test_set'),
        "real_vs_stylegan2": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan2'], split='test_set'),
                "real_vs_sdv1_4": RealSynthethicDataloader(real_dir, IMAGE_DIR['sdv1_4'], split='test_set')
,        "real_vs_stylegan3": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan3'], split='test_set'),
        "real_vs_styleganxl": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan_xl'], split='test_set'),
    }

    test_results = {}
    for name, dataset in dataloaders_test.items():
        feat_file_test = os.path.join(feature_dir, f"test_{name}_features.pt")
        print(f"Preparing test features for {name} -> {feat_file_test}")

        if force_recompute_features or not os.path.exists(feat_file_test):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers)
            feats_test, labels_test, feat_time = extract_and_save_features(backbone_net, loader,
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

        #plot_save_path = os.path.join("./logs", f"feature_plot_{name}_{plot_method}.png")
        #os.makedirs("./logs", exist_ok=True)
        #plot_features_with_anchors(real_feats_eval, fake_feats_eval, anchors_cpu,
                                   #method=plot_method, save_path=plot_save_path,
                                   #subsample=plot_subsample)

        # Evaluate classifier on test features
        test_dataset = TensorDataset(feats_test, labels_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        loss, acc = evaluate(classifier, test_loader, criterion, device,
                             rel_module=rel_module, test_name=name, save_dir="./logs")
        test_results[name] = {"loss": loss, "acc": acc, "feat_time": feat_time}

    # --- Append evaluation results to CSV ---
    # CSV columns/order requested by user:
    csv_columns = [
        'fine_tuning_on',
        'real_vs_stylegan1',
        'real_vs_stylegan2',
        'real_vs_sdv1_4',
        'real_vs_stylegan3',
        'real_vs_styleganxl',
    ]

    # Default path if not provided
    if eval_csv_path is None:
        eval_csv_path = os.path.join('logs', 'test_accuracies.csv')
    os.makedirs(os.path.dirname(eval_csv_path), exist_ok=True)

    with open(eval_csv_path, 'a') as f:
        if os.path.getsize(eval_csv_path) == 0:
            # write header if file is empty
            f.write(','.join(csv_columns) + '\n')

        row = [fine_tuning_on]
        for col in csv_columns[1:]:
            if col in test_results:
                row.append(f"{test_results[col]['acc']:.5f}")
            else:
                row.append('')
        f.write(','.join(row) + '\n')
    return test_results


# ---------------------------------------------
# Main CLI (kept for convenience)
# ---------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_samples', type=int, default=100,
                        help="Number of training samples to use. If None, use all available samples.")
    parser.add_argument('--fine_tuning_on', type=str, default='stylegan2',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4'])
    parser.add_argument('--backbone', type=str, default='stylegan1',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4'])
    parser.add_argument('--num_anchors', type=int, default=5000,
                        help="Exact number of real features to use as anchors; if greater than available reals, sampling with replacement is used.")
    parser.add_argument('--plot_method', type=str, default='pca', choices=['pca', 'tsne'],
                        help="Dimensionality reduction method for plotting (pca or tsne)")
    parser.add_argument('--plot_subsample', type=int, default=5000,
                        help="Max number of eval points (real+fake) to plot (anchors always included)")
    parser.add_argument('--force_recompute_features', action='store_true',
                        help="Force recomputation of saved features")
    parser.add_argument('--eval_csv_path', type=str, default=None,
                        help='Optional path to evaluation CSV (default: ./logs/eval_results.csv)')
    args = parser.parse_args()

    # pass explicit keyword args to the function (no argparse.Namespace usage inside fine_tune)
    results = fine_tune(**vars(args))
    print("All test results:")
    for k, v in results.items():
        print(f" - {k}: loss={v['loss']:.4f}, acc={v['acc']:.4f}, feat_time={v['feat_time']:.2f}s")
