#!/usr/bin/env python3
import os
import time
import warnings
import argparse
import re
import copy
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50

# === Project imports ===
from src.g_dataloader import RealSynthethicDataloader

# ==============================================================================
# 1. CONFIGURAZIONE DATASET
# ==============================================================================

IMAGE_DIR = {
    # REAL
    'real_progan256' : '/seidenas/datasets/DoGANs/new/Pristine/GGAN256/celeba256/',
    'real_progan1024' : '/seidenas/datasets/DoGANs/new/Pristine/GGAN1024/HQ-IMG/',
    'real_stargan' : '/seidenas/datasets/DoGANs/new/Pristine/starGAN/celeba256',
    'real_glow' : '/seidenas/datasets/DoGANs/new/Pristine/GGAN256/celeba256/', 
    
    # FAKE
    'fake_progan256' : '/seidenas/datasets/DoGANs/new/Generated/GGAN256/celeba256',
    'fake_progan1024' : '/seidenas/datasets/DoGANs/new/Generated/GGAN1024/celebhq',
    'fake_stargan' : '/seidenas/datasets/DoGANs/new/Generated/starGAN/Smiling',
    'fake_glow' : '/seidenas/datasets/DoGANs/new/Generated/glow/Male',
}

TASK_REAL_MAPPING = {
    'fake_progan256':  'real_progan256',
    'fake_progan1024': 'real_progan1024',
    'fake_stargan':    'real_stargan',
    'fake_glow':       'real_glow',
}

# ---------------------------------------------
# TRAINING FUNCTION (iCaRL / LwF - No Buffer)
# ---------------------------------------------
def train_icarl_epoch(model, old_model, train_loader, optimizer, scaler, device_obj, lambda_dist=1.0):
    model.train()
    if old_model:
        old_model.eval()
        
    criterion_cls = nn.CrossEntropyLoss()
    running_loss, running_dist = 0.0, 0.0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device_obj), labels.to(device_obj)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            # 1. Forward modello corrente
            logits = model(imgs)
            loss_cls = criterion_cls(logits, labels)
            
            # 2. Distillation Loss (iCaRL style: BCE sui logits vecchi)
            loss_dist = torch.tensor(0.).to(device_obj)
            if old_model is not None:
                with torch.no_grad():
                    old_logits = old_model(imgs)
                
                # In iCaRL, la distillazione si fa forzando i nuovi logits a produrre 
                # le stesse probabilità (sigmoidi) del modello precedente.
                old_targets = torch.sigmoid(old_logits)
                loss_dist = F.binary_cross_entropy_with_logits(logits, old_targets)
            
            # 3. Loss Totale
            loss = loss_cls + (lambda_dist * loss_dist)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss_cls.item()
        running_dist += loss_dist.item() if old_model else 0.0
        
    return running_loss / len(train_loader), running_dist / len(train_loader)

# ---------------------------------------------
# PIPELINE CONTINUAL LEARNING iCaRL
# ---------------------------------------------
def fine_tune_cl_icarl(
    prev_model_path: str,
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = '0',
    epochs: int = 10,
    lr: float = 1e-4,
    seed: int = 42,
    fine_tuning_on: str = 'fake_progan256',
    checkpoint_file: str = "checkpoint/icarl_step.pth",
    order_list: list = None,
    csv_log_path: str = "results_icarl_table.csv",
    real_dataset_key: str = None, 
    lambda_dist: float = 1.0, # Peso della distillazione (1.0 è standard per iCaRL)
    **kwargs
):
    device_obj = torch.device("cuda:"+device if torch.cuda.is_available() else "cpu")
    print(f"\n" + "="*50)
    print(f"--- CL Step: iCaRL (No Buffer) on {fine_tuning_on} ---")
    print("="*50)
    torch.manual_seed(seed); np.random.seed(seed)
    
    # 1. INIT MODELLI (Standard ResNet50)
    model = resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    if os.path.exists(prev_model_path):
        print(f"Loading previous weights from: {prev_model_path}")
        ckpt = torch.load(prev_model_path, map_location=device_obj)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True) 
    else:
        print("Starting from scratch (ImageNet initialization)")
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device_obj)
    
    # Assicuriamo che tutto sia addestrabile (Full Backbone FT)
    for param in model.parameters(): param.requires_grad = True
    
    # Creiamo il Modello Vecchio (Frozen) per la Distillazione
    old_model = None
    if os.path.exists(prev_model_path):
        old_model = copy.deepcopy(model)
        old_model.eval()
        for param in old_model.parameters(): param.requires_grad = False
        print("Old Model loaded and frozen for Logit Distillation.")

    # 2. DATALOADER TRAINING
    real_dir = IMAGE_DIR[real_dataset_key]
    fake_dir = IMAGE_DIR[fine_tuning_on]
    print(f"Training Data:\n - Real: {real_dir}\n - Fake: {fake_dir}")
    
    dataset = RealSynthethicDataloader(real_dir, fake_dir, split='', num_training_samples=None) 
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,  pin_memory=False)
    
    # 3. TRAINING LOOP
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"Starting Training for {epochs} epochs (Lambda Dist = {lambda_dist})...")
    for epoch in range(epochs):
        cls_loss, dist_loss = train_icarl_epoch(model, old_model, train_loader, optimizer, scaler, device_obj, lambda_dist)
        print(f"   Epoch {epoch+1:02d}/{epochs} | Cls Loss: {cls_loss:.4f} | Dist Loss: {dist_loss:.4f}")

    # 4. VALUTAZIONE MULTI-TASK
    model.eval()
    csv_row = {'Train_Step': fine_tuning_on}
    res_dir = os.path.join("results_icarl", f"train_on_{fine_tuning_on}")
    os.makedirs(res_dir, exist_ok=True)
    
    print(f"\n--- Evaluation Table Generation ---")
    for t_task in order_list:
        t_real_key = TASK_REAL_MAPPING.get(t_task, 'real_progan256')
        t_real_dir = IMAGE_DIR[t_real_key]
        t_fake_dir = IMAGE_DIR[t_task]
        
        try:
            t_ds = RealSynthethicDataloader(t_real_dir, t_fake_dir, split='')
            split_idx = int(0.8 * len(t_ds))
            _, t_test_ds = torch.utils.data.random_split(t_ds, [split_idx, len(t_ds)-split_idx])
            
            t_loader = DataLoader(t_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
            
            all_preds, all_labels, all_logits = [], [], []
            
            with torch.no_grad():
                for imgs, labels in t_loader:
                    imgs = imgs.to(device_obj)
                    with torch.cuda.amp.autocast():
                        logits = model(imgs) 
                    _, predicted = torch.max(logits.data, 1)
                    
                    all_logits.append(logits.cpu().numpy())
                    all_preds.append(predicted.cpu().numpy())
                    all_labels.append(labels.numpy())
            
            all_logits = np.concatenate(all_logits)
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            
            acc = (all_preds == all_labels).mean()
            csv_row[t_task] = acc
            
            # Salvataggio Artifacts
            np.save(os.path.join(res_dir, f"logits_eval_{t_task}.npy"), all_logits)
            cm = confusion_matrix(all_labels, all_preds)
            np.savetxt(os.path.join(res_dir, f"cm_{t_task}.txt"), cm, fmt='%d')
            
            mark = "(*)" if t_task == fine_tuning_on else ""
            print(f"Task {t_task:20} | Acc: {acc:.4f} {mark}")
            
        except Exception as e:
            print(f"Errore durante la valutazione di {t_task}: {e}")
            csv_row[t_task] = 0.0

    # 5. SALVATAGGIO CSV E MODELLO
    df_row = pd.DataFrame([csv_row])
    header = not os.path.exists(csv_log_path)
    df_row.to_csv(csv_log_path, mode='a', header=header, index=False)
    
    torch.save({'state_dict': model.state_dict()}, checkpoint_file)
    print(f"Model saved to {checkpoint_file}")
    
    return csv_row

# ---------------------------------------------
# Main
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='0') # Modificato per allinearsi all'os.environ
    parser.add_argument('--epochs', type=int, default=10) 
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda_dist', type=float, default=1.0, help="Peso della Distillazione iCaRL")
    parser.add_argument('--order', default='[fake_progan256, fake_progan1024, fake_stargan, fake_glow]', help="Lista dei task")
    parser.add_argument('--csv_log', type=str, default='results_icarl_table.csv')

    args = parser.parse_args()
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    order_list = re.sub(r'[\[\]\s]', '', args.order).split(',')

    print("\n=== Starting iCaRL Continual Learning (No Buffer) ===")
    if os.path.exists(args.csv_log): os.remove(args.csv_log)
    os.makedirs("checkpoint", exist_ok=True)
    
    prev_ckpt = "checkpoint/dummy_non_existent.pth" 
    
    for i, task in enumerate(order_list):
        current_ckpt = os.path.join("checkpoint", f"icarl_step_{i}_{task}.pth")
        real_key = TASK_REAL_MAPPING.get(task, 'real_progan256')
        
        fine_tune_cl_icarl(
            prev_model_path=prev_ckpt,
            checkpoint_file=current_ckpt,
            fine_tuning_on=task,
            real_dataset_key=real_key,
            order_list=order_list,
            csv_log_path=args.csv_log,
            lambda_dist=args.lambda_dist,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed
        )
        
        prev_ckpt = current_ckpt