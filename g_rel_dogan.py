#!/usr/bin/env python3
import os
import time
import warnings
import argparse
import re
import copy
import shutil
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# === Project imports ===
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model
from src.utils import RelativeRepresentation, RelClassifier, extract_and_save_features
from src.g_utils import evaluate3

# ==============================================================================
# 1. CONFIGURAZIONE DATASET
# ==============================================================================

IMAGE_DIR = {
    # REAL
    'real_progan256' : '/seidenas/datasets/DoGANs/new/Pristine/GGAN256/celeba256/',
    'real_progan1024' : '/seidenas/datasets/DoGANs/new/Pristine/GGAN1024/celebhq/',
    'real_stargan' : '/seidenas/datasets/DoGANs/new/Pristine/starGAN/Smiling',
    'real_glow' : '/seidenas/datasets/DoGANs/new/Pristine/glow/Male',
    
    # FAKE
    'fake_progan256' : '/seidenas/datasets/DoGANs/new/Generated/GGAN256/celeba256',
    'fake_progan1024' : '/seidenas/datasets/DoGANs/new/Generated/GGAN1024/celebhq',
    'fake_stargan' : '/seidenas/datasets/DoGANs/new/Generated/starGAN/Smiling',
    'fake_glow' : '/seidenas/datasets/DoGANs/new/Generated/glow/Male',
}

# MAPPING: Quale dataset Real usare per confrontare ogni Fake?
TASK_REAL_MAPPING = {
    'fake_progan256':  'real_progan256',
    'fake_progan1024': 'real_progan1024',
    'fake_stargan':    'real_stargan',
    'fake_glow':       'real_glow',
}

# ==============================================================================
# UTILS
# ==============================================================================

def resolve_data_path(base_path_from_config):
    """Gestisce la presenza/assenza di sottocartelle 'train_set' o 'train'."""
    if not base_path_from_config or not os.path.exists(base_path_from_config):
        return None
    
    # Cerca file immagine direttamente nella root
    if len(glob.glob(os.path.join(base_path_from_config, "*.png"))) > 0: return base_path_from_config
    if len(glob.glob(os.path.join(base_path_from_config, "*.jpg"))) > 0: return base_path_from_config
        
    # Altrimenti cerca sottocartelle comuni
    for sub in ['train_set', 'train', 'val_set', 'val', '']:
        p = os.path.join(base_path_from_config, sub)
        if os.path.exists(p):
            n_imgs = len(glob.glob(os.path.join(p, "*.png"))) + len(glob.glob(os.path.join(p, "*.jpg")))
            if n_imgs > 0:
                return p
    return base_path_from_config

# ---------------------------------------------
# A. Funzione per il Fine-Tuning della Backbone (Fase 1)
# ---------------------------------------------
def finetune_backbone_routine(args, device_obj):
    print("\n" + "="*50)
    print(">>> PHASE 1: FULL BACKBONE FINE-TUNING ON PROGAN (AMP ENABLED)")
    print("="*50)
    
    # Carichiamo una ResNet50 generica (ImageNet)
    model = load_pretrained_model('resnet50') 
    
    num_ftrs = model.resnet.fc.in_features
    model.resnet.fc = nn.Linear(num_ftrs, 2) 
    model = model.to(device_obj)
    
    # Scongeliamo tutto per il fine-tuning completo
    for param in model.parameters():
        param.requires_grad = True

    # Usiamo esplicitamente i path per ProGAN 256
    real_dir = resolve_data_path(IMAGE_DIR['real_progan256'])
    fake_dir = resolve_data_path(IMAGE_DIR['fake_progan256'])
    
    print(f"Loading Data for Backbone FT:\n - Real: {real_dir}\n - Fake: {fake_dir}")
    if not real_dir or not fake_dir:
        raise ValueError("Cannot find paths for backbone fine-tuning!")

    dataset = RealSynthethicDataloader(real_dir, fake_dir, split='')
    
    # Split 80% Train, 20% Val
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Learning rate basso
    scaler = torch.cuda.amp.GradScaler() 
    
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    ft_epochs = 10 # Aumentato a 10 per dare tempo di convergere
    
    print(f"Starting Fine-Tuning for {ft_epochs} epochs...")

    for epoch in range(ft_epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device_obj), labels.to(device_obj)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calcolo metriche training
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100 * train_correct / train_total

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device_obj), labels.to(device_obj)
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    v_loss = criterion(outputs, labels) # Calcolo Loss Validation
                
                val_loss += v_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / val_total
        
        # Logging completo
        print(f"Epoch {epoch+1:02d}/{ft_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}%")
        
        # Salviamo il modello se migliora la validation accuracy
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_weights = copy.deepcopy(model.state_dict())

    print(f"Fine-tuning complete. Best Validation Acc: {best_val_acc:.2f}%")
    
    model.load_state_dict(best_weights)
    save_path = "checkpoint/resnet50_progan_finetuned.pth"
    torch.save({'state_dict': model.state_dict()}, save_path)
    print(f"Saved compatible checkpoint to: {save_path}")
    return save_path

# ---------------------------------------------
# B. Pre-computazione (Aggiornata con Real dinamici)
# ---------------------------------------------
def precompute_all_features(backbone_path, order_list, device_obj, batch_size=64, num_workers=8):
    print("\n" + "="*50)
    print(">>> PHASE 2: PRE-COMPUTING FEATURES")
    print("="*50)
    
    # Caricamento custom backbone
    from src.net import ResNet50BC
    backbone_net = ResNet50BC()
    checkpoint = torch.load(backbone_path, map_location=device_obj)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    backbone_net.load_state_dict(state_dict, strict=False) 
    
    backbone_net.resnet.fc = nn.Identity()
    backbone_net.to(device_obj).eval().half()
    
    feat_folder = "./feature_resnet50_progan"
    os.makedirs(feat_folder, exist_ok=True)
    
    for task in order_list:
        feature_file = os.path.join(feat_folder, f"feats_{task}.pt")
        if os.path.exists(feature_file):
            print(f"Features for {task} exist. Skipping.")
            continue
            
        print(f"Extracting features for {task}...")
        
        # LOGICA DINAMICA PER I REAL
        # Se il task non è nella mappa, fallback a progan256
        real_key = TASK_REAL_MAPPING.get(task, 'real_progan256') 
        
        if real_key not in IMAGE_DIR:
             print(f"WARNING: Key {real_key} not in IMAGE_DIR. Fallback.")
             real_key = 'real_progan256'

        real_dir = resolve_data_path(IMAGE_DIR[real_key])
        fake_dir = resolve_data_path(IMAGE_DIR[task])
        
        print(f"   Real: {real_dir}")
        print(f"   Fake: {fake_dir}")

        if not real_dir or not fake_dir:
            print(f"SKIP {task}: Paths invalid.")
            continue

        try:
            dataset = RealSynthethicDataloader(real_dir, fake_dir, split='')
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            with torch.cuda.amp.autocast():
                extract_and_save_features(backbone_net, loader, feature_file, device_obj)
        except Exception as e:
            print(f"ERROR extracting {task}: {e}")

# ---------------------------------------------
# C. Pipeline Continual Learning (CSV Tabellare)
# ---------------------------------------------
def fine_tune_cl(
    model_path: str,
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = '0',
    epochs: int = 300,
    lr: float = 1e-3,
    seed: int = 42,
    fine_tuning_on: str = 'stylegan2',
    num_anchors: int = 5000,
    checkpoint_file: str = "checkpoint/checkpoint_HELLO.pth",
    load_checkpoint: bool = False,
    order_list: list = None,
    csv_log_path: str = "results_table.csv",
    real_dataset_key: str = None, 
    **kwargs
):
    device_obj = torch.device("cuda:"+device if torch.cuda.is_available() else "cpu")
    print(f"--- CL Step: Training on {fine_tuning_on} ---")
    torch.manual_seed(seed); np.random.seed(seed)
    
    feat_folder = "./feature_resnet50_progan"
    train_feat_file = os.path.join(feat_folder, f"feats_{fine_tuning_on}.pt")
    
    if not os.path.exists(train_feat_file):
        raise FileNotFoundError(f"Features for {fine_tuning_on} missing!")

    data = torch.load(train_feat_file)
    feats_all, labels_all = data["features"], data["labels"]

    # Split
    f_train, f_test, l_train, l_test = train_test_split(
        feats_all.cpu().numpy(), labels_all.cpu().numpy(), 
        test_size=0.2, random_state=seed, stratify=labels_all.cpu().numpy()
    )
    f_train, l_train = torch.from_numpy(f_train), torch.from_numpy(l_train)
    f_test, l_test = torch.from_numpy(f_test), torch.from_numpy(l_test)
    
    # 2. Setup Classificatore e Ancore
    real_mask = l_train == 0
    anchors = f_train[real_mask][:num_anchors]
    
    rel_module = RelativeRepresentation(anchors.to(device_obj).half())
    classifier = RelClassifier(rel_module, anchors.size(0), num_classes=2).to(device_obj)
    
    if load_checkpoint and os.path.exists(checkpoint_file):
        print(f"Loading CL checkpoint: {checkpoint_file}")
        classifier.load_state_dict(torch.load(checkpoint_file)['state_dict'])

    # 3. Training Loop
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device_obj)
    scaler = torch.cuda.amp.GradScaler()

    train_loader = DataLoader(TensorDataset(f_train, l_train), batch_size=batch_size, shuffle=True)
    
    classifier.train()
    print(f"Training RelClassifier for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_f, batch_l in train_loader:
            batch_f, batch_l = batch_f.to(device_obj), batch_l.to(device_obj)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = classifier(batch_f)
                loss = criterion(outputs, batch_l)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")

    # 4. Valutazione Multi-Task e Creazione Riga Tabella
    classifier.eval()
    all_results = {}
    
    # Riga CSV per questo step
    csv_row = {'Train_Step': fine_tuning_on}
    
    # Cartella per confusion matrices
    res_dir = os.path.join("results", f"train_on_{fine_tuning_on}")
    os.makedirs(res_dir, exist_ok=True)

    print(f"\n--- Evaluation Table Generation ---")
    for t_task in order_list:
        t_feat_file = os.path.join(feat_folder, f"feats_{t_task}.pt")
        if not os.path.exists(t_feat_file): 
            csv_row[t_task] = 0.0 
            continue
            
        t_data = torch.load(t_feat_file)
        ft_all, lt_all = t_data["features"], t_data["labels"]
        
        _, ft_test, _, lt_test = train_test_split(
            ft_all.cpu().numpy(), lt_all.cpu().numpy(), 
            test_size=0.2, random_state=seed, stratify=lt_all.cpu().numpy()
        )
        
        t_loader = DataLoader(TensorDataset(torch.from_numpy(ft_test), torch.from_numpy(lt_test)), batch_size=batch_size)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # Nota: qui ho rimesso evaluate3 ma puoi usare anche il loop manuale se preferisci i logits
                # Per brevità riuso la tua importazione, ma se vuoi i logits usa la versione precedente
                _, acc, preds, _ = evaluate3(classifier, t_loader, criterion, device_obj)
        
        all_results[t_task] = acc
        csv_row[t_task] = acc
        
        # Confusion Matrix
        if isinstance(preds, torch.Tensor): y_pred = preds.cpu().numpy()
        else: y_pred = preds
        cm = confusion_matrix(lt_test, y_pred)
        np.savetxt(os.path.join(res_dir, f"cm_{t_task}.txt"), cm, fmt='%d')
        
        mark = "(*)" if t_task == fine_tuning_on else ""
        print(f"Task {t_task:20} | Acc: {acc:.4f} {mark}")

    # 5. Salvataggio Riga nel CSV
    df_row = pd.DataFrame([csv_row])
    header = not os.path.exists(csv_log_path)
    df_row.to_csv(csv_log_path, mode='a', header=header, index=False)
    print(f"TableRow appended to {csv_log_path}")

    torch.save({'state_dict': classifier.state_dict()}, checkpoint_file)
    return all_results, anchors, None

# ---------------------------------------------
# Main
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10) # 10 epoche per il classifier
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_anchors', type=int, default=5000)
    parser.add_argument('--backbone', type=str, default='stylegan1') # Placeholder
    parser.add_argument('--do_backbone_finetuning', action='store_true')
    
    parser.add_argument('--order', default='[fake_progan256, fake_progan1024, fake_stargan, fake_glow]', 
                        help="Lista dei task per il Continual Learning")
    
    parser.add_argument('--csv_log', type=str, default='cl_results_table.csv')

    args = parser.parse_args()
    device_obj = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
    order_list = re.sub(r'[\[\]\s]', '', args.order).split(',')

    # 1. Backbone
    backbone_model_path = "checkpoint/resnet50_progan_finetuned.pth"
    if args.do_backbone_finetuning or not os.path.exists(backbone_model_path):
        backbone_model_path = finetune_backbone_routine(args, device_obj)
    elif not os.path.exists(backbone_model_path):
        print(f"ERRORE: Backbone {backbone_model_path} non trovata.")
        exit(1)
    
    print(f"\n>>> Using Backbone: {backbone_model_path}")

    # 2. Pre-compute Features
    precompute_all_features(backbone_model_path, order_list, device_obj, 
                            batch_size=args.batch_size, num_workers=args.num_workers)

    # 3. CL Loop
    print(f"\n=== Starting CL on: {order_list} ===")
    
    if os.path.exists(args.csv_log): 
        os.remove(args.csv_log)
        print("Old CSV removed. Starting fresh table.")
    
    prev_ckpt = None
    for i, task in enumerate(order_list):
        print(f"\n>>> CL STEP {i+1}: {task}")
        current_ckpt = os.path.join("checkpoint", f"cl_step_{i}_{task}.pth")
        
        real_key = TASK_REAL_MAPPING.get(task, 'real_progan256')
        
        cl_args = vars(args)
        cl_args['fine_tuning_on'] = task
        cl_args['checkpoint_file'] = current_ckpt
        cl_args['load_checkpoint'] = (i > 0)
        
        if i > 0:
            cl_args['checkpoint_file'] = current_ckpt 
            shutil.copy(prev_ckpt, current_ckpt)
        
        results, _, _ = fine_tune_cl(
            model_path=backbone_model_path, 
            order_list=order_list,
            real_dataset_key=real_key,
            csv_log_path=args.csv_log,
            **cl_args
        )
        prev_ckpt = current_ckpt