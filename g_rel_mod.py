#!/usr/bin/env python3
import os
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from src.utils import  BalancedBatchSampler, RelativeRepresentation, extract_and_save_features, evaluate, plot_features_with_anchors
from src.utils import train_one_epoch_with_distill as train_one_epoch  # assuming same signature
from src.utils import RelClassifierWithEmbedding
from src.utils import intra_class_compactness_loss
from src.utils import NTXentLoss
from src.utils import freeze_model

# ---------------------------------------------
# Fine-tuning pipeline (main)
# ---------------------------------------------
# --- nuova fine_tune (sostituisce la versione precedente) ---
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
    # checkpoint handling
    load_checkpoint: bool = False,
    checkpoint_file: str = "checkpoint_mod/checkpoint_HELLO_mod.pth",
    # NEW: distillation & compactness hyperparams
    lambda_contrast: float = 0.5,
    lambda_compact: float = 0.01,
    use_distillation: bool = False,
    only_on_real_for_contrast: bool = True,
    embedding_dim: int = 256,
    hidden_dim: int = 512,
):
    """
    Fine-tune with optional contrastive distillation and intra-class compactness loss.

    IMPORTANT: requires in src.utils:
      - RelClassifierWithEmbedding
      - NTXentLoss
      - freeze_model
      - train_one_epoch_with_distill

    Returns:
        test_results (dict), anchors (torch.Tensor)
    """
    # imports specific to extended training (assumes you added them in src.utils)
    from src.utils import RelClassifierWithEmbedding, NTXentLoss, freeze_model, train_one_epoch_with_distill

    # --- device handling
    if isinstance(device, str) and device.lower() == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{device}")
        else:
            device = torch.device('cpu')
    print(f"Using device: {device}")

    feature_dir = f"./feature_{backbone}_mod"
    checkpoint_dir = "./checkpoint_mod"
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file_dir = os.path.dirname(checkpoint_file) if os.path.dirname(checkpoint_file) != '' else checkpoint_dir
    os.makedirs(checkpoint_file_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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

    assert len(feats_full) == len(labels_full), "Mismatch between features and labels lengths."

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

    # Anchors (take from real training features only)
    real_mask = labels == 0
    real_feats = feats[real_mask]
    if real_feats.size(0) == 0:
        raise RuntimeError("No real training features available to form anchors. Check your dataset / subsampling (num_train_samples).")

    if num_anchors is not None:
        rng = torch.Generator().manual_seed(seed)
        num_requested = int(num_anchors)
        if num_requested > len(real_feats):
            print(f"[warning] Requested {num_requested} anchors but only {len(real_feats)} unique real samples available. Sampling WITH replacement from real set.")
            idx = torch.randint(low=0, high=len(real_feats), size=(num_requested,), generator=rng)
        else:
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

    # Classifier: embedding head + classifier
    classifier = RelClassifierWithEmbedding(rel_module, anchors.size(0),
                                            embedding_dim=embedding_dim,
                                            hidden_dim=hidden_dim,
                                            num_classes=2).to(device)

    # Optionally initialize classifier weights from checkpoint (existing behavior)
    if load_checkpoint:
        if os.path.exists(checkpoint_file):
            try:
                print(f"Initializing classifier from checkpoint {checkpoint_file} ...")
                checkpoint = torch.load(checkpoint_file, map_location=device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    classifier.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    classifier.load_state_dict(checkpoint, strict=False)
                print("Checkpoint loaded into classifier (init).")
            except Exception as e:
                print(f"Failed to init classifier from checkpoint: {e}. Continuing.")

    # Optionally prepare old model for distillation (frozen)
    old_model = None
    if use_distillation and os.path.exists(checkpoint_file):
        try:
            old_model = RelClassifierWithEmbedding(rel_module, anchors.size(0),
                                                  embedding_dim=embedding_dim,
                                                  hidden_dim=hidden_dim,
                                                  num_classes=2).to(device)
            ckpt = torch.load(checkpoint_file, map_location=device)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                old_model.load_state_dict(ckpt['state_dict'], strict=False)
            else:
                old_model.load_state_dict(ckpt, strict=False)
            print(f"Old model loaded from checkpoint {checkpoint_file} for distillation.")
            old_model = freeze_model(old_model)
            print("Old model loaded and frozen for distillation.")
        except Exception as e:
            print(f"Could not load old checkpoint for distillation: {e}")
            old_model = None
            # continue without distillation

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # Prepare NT-Xent instance
    nt_xent = NTXentLoss(temperature=0.1)

    # Training loop (uses the specialized train_one_epoch_with_distill)
    start_time = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_with_distill(
            classifier, feat_loader, criterion, optimizer, device,
            old_model=old_model,
            nt_xent_loss=nt_xent,
            lambda_contrast=lambda_contrast,
            lambda_compact=lambda_compact,
            only_on_real_for_contrast=only_on_real_for_contrast
        )
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.2f} minutes")

    # Save checkpoint to the specified checkpoint_file
    checkpoint_path = checkpoint_file
    os.makedirs(os.path.dirname(checkpoint_path) or checkpoint_dir, exist_ok=True)
    torch.save({'state_dict': classifier.state_dict()}, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Prepare test datasets (same as before)
    dataloaders_test = {
        "real_vs_stylegan1": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan1'], split='test_set'),
        "real_vs_stylegan2": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan2'], split='test_set'),
        "real_vs_sdv1_4": RealSynthethicDataloader(real_dir, IMAGE_DIR['sdv1_4'], split='test_set'),
        "real_vs_stylegan3": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan3'], split='test_set'),
        "real_vs_styleganxl": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan_xl'], split='test_set'),
        "real_vs_sdv2_1": RealSynthethicDataloader(real_dir, IMAGE_DIR['sdv2_1'], split='test_set'),
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

        anchors_cpu = anchors.cpu()
        real_mask_eval = (labels_test == 0)
        fake_mask_eval = (labels_test == 1)
        real_feats_eval = feats_test[real_mask_eval]
        fake_feats_eval = feats_test[fake_mask_eval]

        test_dataset = TensorDataset(feats_test, labels_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        loss, acc = evaluate(classifier, test_loader, criterion, device,
                             rel_module=rel_module, test_name=name, save_dir="./logs")
        test_results[name] = {"loss": loss, "acc": acc, "feat_time": feat_time}

    # Append evaluation results to CSV
    csv_columns = [
        'fine_tuning_on',
        'real_vs_stylegan1',
        'real_vs_stylegan2',
        'real_vs_sdv1_4',
        'real_vs_stylegan3',
        'real_vs_styleganxl',
        'real_vs_sdv2_1',
        'lambda_contrast',
        'lambda_compact'
    ]

    if eval_csv_path is None:
        eval_csv_path = os.path.join('logs_mod', 'test_accuracies.csv')
    os.makedirs(os.path.dirname(eval_csv_path), exist_ok=True)

    file_new = not os.path.exists(eval_csv_path) or os.path.getsize(eval_csv_path) == 0
    with open(eval_csv_path, 'a') as f:
        if file_new:
            f.write(','.join(csv_columns) + '\n')
        row = [fine_tuning_on]
        for col in csv_columns[1:-2]:
            if col in test_results:
                row.append(f"{test_results[col]['acc']:.5f}")
            else:
                row.append('')
        row.append(f"{lambda_contrast:.5f}")
        row.append(f"{lambda_compact:.5f}")
        f.write(','.join(row) + '\n')

    return test_results, anchors

# --- nuovo main per grid-search su lambda_contrast / lambda_compact ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # re-usa gli stessi argomenti essenziali
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_samples', type=int, default=None)
    parser.add_argument('--fine_tuning_on', type=str, default='stylegan2',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4', 'stylegan3','sdv2_1'])
    parser.add_argument('--backbone', type=str, default='stylegan1',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4', 'stylegan3','sdv2_1'])
    parser.add_argument('--force_recompute_features', action='store_true')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint_mod/checkpoint_HELLO_mod.pth')
    parser.add_argument('--load_checkpoint', action='store_true',
                        help="If set, attempt to initialize classifier from checkpoint (keeps old behaviour).")
    parser.add_argument('--use_distillation', action='store_true',
                        help="If set, tries to load checkpoint and use as frozen teacher for contrastive distillation.")
    parser.add_argument('--eval_csv_path', type=str, default='logs_mod/lambda_search_results.csv')
    args = parser.parse_args()

    # grid di esempio (modifica come preferisci)
    lambda_contrast_values = [0.0, 0.1, 0.5]
    lambda_compact_values = [0.0, 0.001, 0.01]

    summary = {}
    os.makedirs(os.path.dirname(args.eval_csv_path) or './logs_mod', exist_ok=True)

    for lc in lambda_contrast_values:
        for lcomp in lambda_compact_values:
            print("="*80)
            print(f"Running with lambda_contrast={lc}, lambda_compact={lcomp}")
            test_results, anchors = fine_tune(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
                num_train_samples=args.num_train_samples,
                fine_tuning_on=args.fine_tuning_on,
                backbone=args.backbone,
                force_recompute_features=args.force_recompute_features,
                load_checkpoint=args.load_checkpoint,
                checkpoint_file=args.checkpoint_file,
                lambda_contrast=lc,
                lambda_compact=lcomp,
                use_distillation=args.use_distillation,
                eval_csv_path=args.eval_csv_path
            )
            # salvataggio riassunto in memoria
            summary[(lc, lcomp)] = test_results

    print("\n=== GRID SEARCH SUMMARY ===")
    for (lc, lcomp), res in summary.items():
        # prendo l'accuracy su real_vs_stylegan2 come esempio (se esiste)
        acc_main = res.get("real_vs_stylegan2", {}).get("acc", None)
        print(f"lc={lc}, lcomp={lcomp} -> real_vs_stylegan2 acc: {acc_main}")
