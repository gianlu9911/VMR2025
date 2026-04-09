#!/usr/bin/env python3
import os
import time
import warnings

warnings.filterwarnings("ignore")
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import re 

# === Project imports - adjust these to your repo layout ===
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model
from src.utils import BalancedBatchSampler, RelativeRepresentation, RelClassifier, extract_and_save_features, evaluate
from src.g_utils import save_features_only

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
    load_checkpoint: bool = False,
    checkpoint_file: str = "checkpoint/checkpoint_HELLO.pth",
    save_feats: bool = False,
    save_feats_prefix: str = None,
    order: str = '[stylegan1, stylegan2, sdv1_4, stylegan3, stylegan_xl, sdv2_1]',
    run_id: str = "Run_0"  # <-- AGGIUNTO PARAMETRO PER IDENTIFICARE L'ORDINE
):
    device = torch.device("cuda:"+device if torch.cuda.is_available() else "cpu")
    print(f"\n--- [{run_id}] Inizio training su: {fine_tuning_on} ---")
    print(f"Using device: {device}")

    feature_dir = f"./feature_{backbone}"
    checkpoint_dir = "./checkpoint"
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_file_dir = os.path.dirname(checkpoint_file) if os.path.dirname(checkpoint_file) != '' else checkpoint_dir
    os.makedirs(checkpoint_file_dir, exist_ok=True)

    logits_eval_root = os.path.join("logits_eval")
    os.makedirs(logits_eval_root, exist_ok=True)

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
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Extract / load features
    full_train_feat_file = os.path.join(feature_dir, f"real_vs_{fine_tuning_on}_features.pt")
    if force_recompute_features or not os.path.exists(full_train_feat_file):
        print("Extracting full training features...")
        feats_full, labels_full, feat_time_full = extract_and_save_features(
            backbone_net, train_loader, full_train_feat_file, device)
    else:
        data = torch.load(full_train_feat_file)
        feats_full, labels_full = data["features"], data["labels"]
        feat_time_full = 0.0
        print("Loaded cached full training features")

    # Subsample training samples
    num_train_samples_val = int(num_train_samples) if num_train_samples is not None else None

    if num_train_samples_val is not None and num_train_samples_val < len(feats_full):
        indices = torch.randperm(len(feats_full))[:num_train_samples_val]
        feats = feats_full[indices]
        labels = labels_full[indices]
    else:
        feats = feats_full
        labels = labels_full

    print(f"Using {len(feats)} training samples (real: {(labels==0).sum().item()}, fake: {(labels==1).sum().item()})")

    # Anchors
    real_mask = labels == 0
    real_feats = feats[real_mask]
    if real_feats.size(0) == 0:
        raise RuntimeError("No real training features available to form anchors.")

    if num_anchors is not None:
        rng = torch.Generator().manual_seed(seed)
        num_requested = int(num_anchors)
        if num_requested > len(real_feats):
            print(f"[warning] Requested {num_requested} anchors but only {len(real_feats)} unique real samples available. Sampling WITH replacement.")
            idx = torch.randint(low=0, high=len(real_feats), size=(num_requested,), generator=rng)
        else:
            idx = torch.randperm(len(real_feats), generator=rng)[:num_requested]
        anchors = real_feats[idx]
    else:
        anchors = real_feats

    print(f"Using {anchors.size(0)} anchors for relative representation")
    rel_module = RelativeRepresentation(anchors.to(device))

    # Dataset + Sampler
    feat_dataset = TensorDataset(feats, labels)
    sampler = BalancedBatchSampler(labels, batch_size=batch_size)
    feat_loader = DataLoader(feat_dataset, batch_sampler=sampler)

    # Classifier
    classifier = RelClassifier(rel_module, anchors.size(0), num_classes=2).to(device)

    # Load checkpoint
    if load_checkpoint:
        if os.path.exists(checkpoint_file):
            try:
                print(f"Loading checkpoint from {checkpoint_file} ...")
                checkpoint = torch.load(checkpoint_file, map_location=device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    classifier.load_state_dict(checkpoint['state_dict'])
                else:
                    classifier.load_state_dict(checkpoint)
                print("Checkpoint loaded into classifier.")
            except Exception as e:
                print(f"Failed to load checkpoint from {checkpoint_file}: {e}. Continuing without loading.")
        else:
            print(f"Checkpoint file {checkpoint_file} not found. Continuing without loading.")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    
    # NEW: AMP Scaler per il Training
    scaler = torch.cuda.amp.GradScaler()

    # --- TRAINING LOOP (CON AMP INTEGRATO) ---
    start_time = time.time()
    for epoch in range(epochs):
        classifier.train()
        epoch_loss = 0.0
        correct, total = 0, 0
        
        for b_feats, b_labels in feat_loader:
            b_feats, b_labels = b_feats.to(device), b_labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = classifier(b_feats)
                loss = criterion(outputs, b_labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += b_labels.size(0)
            correct += (preds == b_labels).sum().item()
            
        train_loss = epoch_loss / len(feat_loader)
        train_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.2f} minutes")

    # Save checkpoint
    torch.save({'state_dict': classifier.state_dict()}, checkpoint_file)
    print(f"Model saved to {checkpoint_file}")

    # --- PREPARAZIONE TEST DATASETS DINAMICA ---
    parsed_order = re.sub(r'[\[\]\s]', '', order).split(',')
    dataloaders_test = {}
    for t_name in parsed_order:
        if t_name in IMAGE_DIR:
            dataloaders_test[t_name] = RealSynthethicDataloader(real_dir, IMAGE_DIR[t_name], split='test_set')
        else:
            print(f"[Warning] Dataset {t_name} non trovato in IMAGE_DIR. Verrà ignorato nei test.")

    test_results = {}
    
    # --- EVALUATION LOOP (CON AMP E METRICHE SCIKIT-LEARN) ---
    classifier.eval()
    for name, ds in dataloaders_test.items():
        feat_file_test = os.path.join(feature_dir, f"test_{name}_features.pt")
        print(f"Evaluating on {name}...")

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        if force_recompute_features or not os.path.exists(feat_file_test):
            feats_test, labels_test, feat_time = extract_and_save_features(
                backbone_net, loader, feat_file_test, device, split='test_set')
        else:
            data = torch.load(feat_file_test)
            feats_test, labels_test = data["features"], data["labels"]
            feat_time = 0.0

        test_dataset = TensorDataset(feats_test, labels_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        eval_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []
        
        with torch.no_grad():
            for b_f, b_l in test_loader:
                b_f, b_l = b_f.to(device), b_l.to(device)
                
                with torch.cuda.amp.autocast():
                    outs = classifier(b_f)
                    loss = criterion(outs, b_l)
                    # Estraiamo le probabilità per l'AUC
                    probs = torch.softmax(outs, dim=1)[:, 1] 
                
                _, preds = torch.max(outs, 1)
                
                eval_loss += loss.item()
                all_labels.extend(b_l.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        # Calcolo Metriche
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0 # Sicurezza in caso di classi mancanti

        test_results[name] = {
            "loss": eval_loss / len(test_loader), 
            "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc, 
            "feat_time": feat_time, "preds": all_preds, "labels": all_labels
        }
        
        print(f"  -> Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Recall: {rec:.4f} | Prec: {prec:.4f}")

    # --- SALVATAGGIO CSV CON IDENTIFICATIVO RUN ---
    if eval_csv_path is None:
        eval_csv_path = os.path.join('logs_original_dataset', 'detailed_metrics_results.csv')
    os.makedirs(os.path.dirname(eval_csv_path), exist_ok=True)

    header_needed = not os.path.exists(eval_csv_path) or os.path.getsize(eval_csv_path) == 0
    with open(eval_csv_path, 'a') as f:
        if header_needed:
            f.write("Run_ID,Train_Task,Eval_Task,Accuracy,Precision,Recall,F1_Score,AUC\n")
            
        for eval_task, metrics in test_results.items():
            row = [
                run_id, fine_tuning_on, eval_task,
                f"{metrics['acc']:.5f}", f"{metrics['prec']:.5f}",
                f"{metrics['rec']:.5f}", f"{metrics['f1']:.5f}", f"{metrics['auc']:.5f}"
            ]
            f.write(','.join(row) + '\n')

    classifier.to(device)
    anchors = anchors.to(device)
    anchros_logits = classifier(anchors)
    anchros_logits = anchros_logits.detach().cpu().numpy()
    
    return test_results, anchors, anchros_logits


# ---------------------------------------------
# Main CLI
# ---------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_samples', type=int, default=None)
    parser.add_argument('--backbone', type=str, default='stylegan1')
    parser.add_argument('--num_anchors', type=int, default=5000)
    parser.add_argument('--plot_method', type=str, default='pca')
    parser.add_argument('--plot_subsample', type=int, default=5000)
    parser.add_argument('--force_recompute_features', action='store_true')
    parser.add_argument('--eval_csv_path', type=str, default=None)
    parser.add_argument('--save_feats', action='store_true')
    parser.add_argument('--save_feats_prefix', type=str, default='saved_numpy_features/step_prova')
    
    # LISTA DEI TASK default
    parser.add_argument('--order', default='[stylegan1, stylegan2, sdv1_4, stylegan3, stylegan_xl, sdv2_1]')

    # Liste fisse da iterare
    orders = [
        '[stylegan1, stylegan2, sdv1_4, stylegan3, stylegan_xl, sdv2_1]',
        '[stylegan1, stylegan2, stylegan3, stylegan_xl, sdv1_4, sdv2_1]',
        '[sdv1_4, sdv2_1, stylegan1, stylegan2, stylegan3, stylegan_xl]',
        '[stylegan2, sdv1_4, stylegan_xl, stylegan3, sdv2_1, stylegan1]',
    ]
    
    args = parser.parse_args()

    for idx, o in enumerate(orders):
        run_name = f"Run_{idx + 1}"
        print(f"\n\n========================================================")
        print(f"=== {run_name} | Testing order: {o} ===")
        print(f"========================================================")
        
        args.order = o
        
        # Parso l'ordine eliminando spazi e parentesi
        task_list = re.sub(r'[\[\]\s]', '', args.order).split(',')
        
        current_ckpt = ""
        
        for i, current_task in enumerate(task_list):
            # IL NOME DEL CHECKPOINT ORA INCLUDE LA RUN_ID
            current_ckpt = os.path.join("checkpoint", f"cl_{run_name}_step_{i}_{current_task}.pth")
            
            # Copiamo tutti i parametri del parser in un dizionario
            func_args = vars(args).copy()
            
            # Forziamo i parametri dinamici
            func_args['fine_tuning_on'] = current_task
            func_args['checkpoint_file'] = current_ckpt
            func_args['load_checkpoint'] = (i > 0) # Carica pesi solo dal secondo task in poi
            func_args['run_id'] = run_name         # Passiamo l'ID
            
            results, anchors, _ = fine_tune(**func_args)