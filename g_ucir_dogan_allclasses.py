#!/usr/bin/env python3
import os
import time
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
import argparse
import re
import copy
import shutil
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms

# === Project imports ===
from src.net import ResNet50BC 
from config import PRETRAINED_MODELS

# ==============================================================================
# 0. UCIR: COSINE CLASSIFIER
# ==============================================================================
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.sigma.data.fill_(10.0)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        out = self.sigma * out
        return out

# ==============================================================================
# 1. CONFIGURAZIONE DATASET
# ==============================================================================

IMAGE_DIR = {
    # REAL
    'real_cycle_gan' : '/seidenas/datasets/DoGANs/new/Pristine/CycleGAN/horse2zebra', 
    'real_progan256' : '/seidenas/datasets/DoGANs/new/Pristine/GGAN256/lsun_tower',
    'real_progan1024': '/seidenas/datasets/DoGANs/new/Pristine/GGAN1024/HQ-IMG',    # <--- ECCOLA!
    'real_glow'      : '/seidenas/datasets/DoGANs/new/Pristine/starGAN/celeba256',            
    'real_stargan'   : '/seidenas/datasets/DoGANs/new/Pristine/starGAN/celeba256',   
    
    # FAKE
    'fake_cycle_gan' : '/seidenas/datasets/DoGANs/new/Generated/CycleGAN/horse2zebra',
    'fake_progan256' : '/seidenas/datasets/DoGANs/new/Generated/GGAN256/lsun_tower',
    'fake_progan1024': '/seidenas/datasets/DoGANs/new/Generated/GGAN1024/celebhq', # (I fake invece li trovava, quindi lascialo così o verifica)
    'fake_glow'      : '/seidenas/datasets/DoGANs/new/Generated/glow/Male',
    'fake_stargan'   : '/seidenas/datasets/DoGANs/new/Generated/starGAN/Brown_Hair',
}

TASK_REAL_MAPPING = {
    'fake_cycle_gan':  'real_cycle_gan',
    'fake_progan256':  'real_progan256',
    'fake_progan1024': 'real_progan1024',
    'fake_glow':       'real_glow',
    'fake_stargan':    'real_stargan',
}

# ==============================================================================
# 2. GESTIONE RIGOROSA DEI DATI (No Data Leakage)
# ==============================================================================

def build_task_datasets(order_list, image_dir, task_mapping, seed=42):
    """
    Crea gli split Train/Test una volta per tutte, associando i Reali corretti
    ed evitando rimescolamenti dinamici che causano Data Leakage.
    """
    print("\n" + "="*50)
    print("[DEBUG] PREPARAZIONE STRICT TRAIN/TEST SPLIT (1:1 Mapping)")
    print("="*50)
    
    task_data = {}
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    
    for task in order_list:
        real_key = task_mapping.get(task)
        if not real_key or real_key not in image_dir or task not in image_dir:
            print(f"[WARNING] Path mancanti per {task}. Salto.")
            continue
            
        real_dir = image_dir[real_key]
        fake_dir = image_dir[task]
        
        # Estrazione path fisici
        real_paths = [os.path.join(root, f) for root, _, files in os.walk(real_dir) for f in files if f.lower().endswith(valid_ext)]
        fake_paths = [os.path.join(root, f) for root, _, files in os.walk(fake_dir) for f in files if f.lower().endswith(valid_ext)]
        
        if len(real_paths) == 0 or len(fake_paths) == 0:
            print(f"[WARNING] Dati insufficienti per {task} (Real: {len(real_paths)}, Fake: {len(fake_paths)})")
            continue
            
        # Bilanciamento 1:1 Fissi
        # Bilanciamento 1:1 Fissi
        random.seed(seed)
        min_len = min(len(real_paths), len(fake_paths))
        real_paths = random.sample(real_paths, min_len)
        fake_paths = random.sample(fake_paths, min_len)
        
        all_paths = real_paths + fake_paths
        
        # --- CREAZIONE LABELS SICURA (Senza asterischi) ---
        all_labels = []
        
        # Aggiungo 'min_len' zeri per le immagini Real
        for i in range(min_len):
            all_labels.append(0)
            
        # Aggiungo 'min_len' uni per le immagini Fake
        for i in range(min_len):
            all_labels.append(1)
        # --------------------------------------------------
        
        # Train/Test Split Sicuro (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            all_paths, all_labels, test_size=0.2, random_state=seed, stratify=all_labels
        )
        
        task_data[task] = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test
        }
        print(f"Task {task:20} -> {len(X_train)} Train imgs | {len(X_test)} Test imgs")
        
    return task_data

class StrictPathDataset(Dataset):
    """
    Dataloader puro: non prende decisioni, non rimescola. Fa solo quello che gli viene detto.
    """
    def __init__(self, paths, labels, is_train=True):
        self.paths = paths
        self.labels = labels
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224))
        return self.transform(img), label

# ==============================================================================
# 3. UCIR FINE-TUNING LOOP
# ==============================================================================

def train_ucir_network(
    old_model_path, current_task, task_data, order_list, device_obj,
    batch_size=32, num_workers=8, epochs=10, lr=1e-4, seed=42, 
    lambda_lf=1.0, checkpoint_file="checkpoint.pth", csv_log_path="results.csv"
):
    print("\n" + "="*50)
    print(f">>> UCIR TRAINING SU: {current_task} (No Memory)")
    print("="*50)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = ResNet50BC()
    num_ftrs = model.resnet.fc.in_features
    model.resnet.fc = CosineLinear(num_ftrs, 2)
    
    current_features = {}
    old_features = {}
    def get_curr_features(m, i, o): current_features['feat'] = o.view(o.size(0), -1)
    def get_old_features(m, i, o): old_features['feat'] = o.view(o.size(0), -1)
    
    model.resnet.avgpool.register_forward_hook(get_curr_features)
    
    old_model = None
    if old_model_path and os.path.exists(old_model_path):
        print(f"Caricamento pesi correnti da: {old_model_path}")
        checkpoint = torch.load(old_model_path, map_location=device_obj)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        
        print("Congelamento 'Old Model' per Less-Forget Constraint...")
        old_model = ResNet50BC()
        old_model.resnet.fc = CosineLinear(num_ftrs, 2)
        old_model.load_state_dict(state_dict, strict=False)
        old_model = old_model.to(device_obj)
        old_model.eval() 
        for param in old_model.parameters():
            param.requires_grad = False
            
        old_model.resnet.avgpool.register_forward_hook(get_old_features)
    else:
        print("Nessun modello precedente. Partenza da zero (Task 1).")
        
    model = model.to(device_obj)
    for param in model.parameters(): 
        param.requires_grad = True

    # Caricamento ESATTO del Train Set
    train_ds = StrictPathDataset(task_data[current_task]['X_train'], task_data[current_task]['y_train'], is_train=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Training Loop
    print(f"Training UCIR per {epochs} epoche...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct, total = 0, 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device_obj), labels.to(device_obj)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss_cls = criterion_cls(outputs, labels)
                
                loss = loss_cls
                if old_model is not None:
                    with torch.no_grad():
                        _ = old_model(imgs)
                    
                    feat_curr = current_features['feat']
                    feat_old = old_features['feat'].detach()
                    
                    loss_lf = (1.0 - F.cosine_similarity(feat_curr, feat_old, dim=1)).mean()
                    loss = loss_cls + (lambda_lf * loss_lf)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")

    torch.save({'state_dict': model.state_dict()}, checkpoint_file)

    # Valutazione
    model.eval()
    csv_row = {'Train_Step': current_task}
    res_dir = os.path.join("results", f"train_on_{current_task}")
    os.makedirs(res_dir, exist_ok=True)

    print(f"\n--- Valutazione Multi-Task ---")
    for t_task in order_list:
        if t_task not in task_data: continue
            
        # Caricamento ESATTO del Test Set inviolato
        t_test_ds = StrictPathDataset(task_data[t_task]['X_test'], task_data[t_task]['y_test'], is_train=False)
        t_loader = DataLoader(t_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        t_correct, t_total = 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, labels in t_loader:
                imgs, labels = imgs.to(device_obj), labels.to(device_obj)
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                
                t_total += labels.size(0)
                t_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = t_correct / t_total if t_total > 0 else 0
        csv_row[t_task] = acc
        
        cm = confusion_matrix(all_labels, all_preds)
        np.savetxt(os.path.join(res_dir, f"cm_{t_task}.txt"), cm, fmt='%d')
        
        mark = "(*)" if t_task == current_task else ""
        print(f"Task {t_task:20} | Test Acc: {acc:.4f} {mark}")

    df_row = pd.DataFrame([csv_row])
    header = not os.path.exists(csv_log_path)
    df_row.to_csv(csv_log_path, mode='a', header=header, index=False)

    return checkpoint_file

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--lambda_lf', type=float, default=1.0)
    parser.add_argument('--order', default='[fake_progan256, fake_cycle_gan, fake_progan1024, fake_glow, fake_stargan]', help="Lista dei task")
    parser.add_argument('--csv_log', type=str, default='ucir_nomemory_results_table.csv')

    args = parser.parse_args()
    device_obj = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
    
    order_list = re.sub(r'[\[\]\s]', '', args.order).split(',')

    # 1. COSTRUZIONE DEI DATASET (Split inviolabile)
    task_data = build_task_datasets(
        order_list=order_list, 
        image_dir=IMAGE_DIR, 
        task_mapping=TASK_REAL_MAPPING, 
        seed=args.seed
    )

    # 2. Inizializzazione UCIR
    os.makedirs("checkpoint", exist_ok=True)
    if os.path.exists(args.csv_log): os.remove(args.csv_log)
    
    current_model_path = "" 
    
    for i, task in enumerate(order_list):
        if task not in task_data: continue
            
        current_ckpt = os.path.join("checkpoint", f"ucir_nomem_step_{i}_{task}.pth")
        
        current_model_path = train_ucir_network(
            old_model_path=current_model_path,
            current_task=task,
            task_data=task_data, # Passiamo tutto il dizionario pre-splittato
            order_list=order_list,
            device_obj=device_obj,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            lambda_lf=args.lambda_lf, 
            checkpoint_file=current_ckpt,
            csv_log_path=args.csv_log
        )