#!/usr/bin/env python3
import os
import time
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
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms

# === Project imports ===
from src.net import load_pretrained_model
from src.utils import RelativeRepresentation, RelClassifier, extract_and_save_features
from src.g_utils import evaluate3

# ==============================================================================
# 1. CONFIGURAZIONE DATASET
# ==============================================================================

IMAGE_DIR = {
    # REAL (Specifici per ogni task)
    'real_cycle_gan' : '/seidenas/datasets/DoGANs/new/Pristine/CycleGAN/horse2zebra', 
    'real_progan256' : '/seidenas/datasets/DoGANs/new/Pristine/GGAN256/lsun_tower',
    'real_progan1024': '/seidenas/datasets/DoGANs/new/Pristine/GGAN1024/HQ-IMG',
    #'real_dc_gan'      : '/seidenas/datasets/DoGANs/new/Pristine/dcgan/celeba64',
    'real_stargan'   : '/seidenas/datasets/DoGANs/new/Pristine/starGAN/celeba256',
    
    # FAKE
    'fake_cycle_gan' : '/seidenas/datasets/DoGANs/new/Generated/CycleGAN/horse2zebra',
    'fake_progan256' : '/seidenas/datasets/DoGANs/new/Generated/GGAN256/lsun_tower',
    'fake_progan1024': '/seidenas/datasets/DoGANs/new/Generated/GGAN1024/celebhq',
    #'fake_dc_gan'      : '/seidenas/datasets/DoGANs/new/Generated/dcgan/celeba64',
    'fake_stargan'   : '/seidenas/datasets/DoGANs/new/Generated/starGAN/Brown_Hair',
}

TASK_REAL_MAPPING = {
    'fake_cycle_gan':  'real_cycle_gan',
    'fake_progan256':  'real_progan256',
    'fake_progan1024': 'real_progan1024',
    #'fake_dc_gan':     'real_dc_gan',
    'fake_stargan':    'real_stargan',
}

def verify_datasets_integrity(order_list, image_dir, task_mapping):
    """
    Esegue un controllo preventivo (Fail-Fast) su tutti i dataset richiesti.
    Verifica che le cartelle esistano e contengano file immagine validi.
    """
    print("\n" + "="*50)
    print("[VERIFICA] Controllo integrità preventivo dei dataset...")
    print("="*50)
    
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    
    for task in order_list:
        # 1. Verifica FAKE
        if task not in image_dir:
            raise ValueError(f"\n[ERRORE FATALE] Il task fake '{task}' non è definito nel dizionario IMAGE_DIR!")
        
        fake_path = image_dir[task]
        if not os.path.exists(fake_path):
            raise ValueError(f"\n[ERRORE FATALE] La cartella FAKE non esiste fisicamente: {fake_path}")
            
        fake_imgs = [f for root, _, files in os.walk(fake_path) for f in files if f.lower().endswith(valid_ext)]
        if len(fake_imgs) == 0:
            raise ValueError(f"\n[ERRORE FATALE] La cartella FAKE è VUOTA o non contiene immagini valide: {fake_path}")
        print(f" [OK] {task:15} -> Trovate {len(fake_imgs)} immagini fake.")

        # 2. Verifica REAL mappato
        real_key = task_mapping.get(task)
        if not real_key:
            raise ValueError(f"\n[ERRORE FATALE] Nessun mapping reale trovato per il task fake '{task}' in TASK_REAL_MAPPING!")
        if real_key not in image_dir:
            raise ValueError(f"\n[ERRORE FATALE] La chiave reale '{real_key}' non è definita in IMAGE_DIR!")
            
        real_path = image_dir[real_key]
        if not os.path.exists(real_path):
            raise ValueError(f"\n[ERRORE FATALE] La cartella REAL non esiste fisicamente: {real_path}")
            
        real_imgs = [f for root, _, files in os.walk(real_path) for f in files if f.lower().endswith(valid_ext)]
        if len(real_imgs) == 0:
            raise ValueError(f"\n[ERRORE FATALE] La cartella REAL è VUOTA o non contiene immagini valide: {real_path}")
        print(f" [OK] {real_key:15} -> Trovate {len(real_imgs)} immagini reali.")

    print("\n[VERIFICA COMPLETATA] Tutti i path sono corretti e i dataset contengono immagini!\n")

# ==============================================================================
# 2. GESTIONE REAL "UNIVERSAL" E CUSTOM DATASET
# ==============================================================================

def get_task_specific_balanced_reals(order_list, image_dir, task_mapping, total_samples=50000, seed=42):
    """
    Crea un pool bilanciato di immagini Real prelevando SOLO dalle cartelle 
    associate ai task presenti in order_list.
    """
    print("\n[INFO] Creazione del pool 'Universal Real' dai task specificati...")
    np.random.seed(seed)
    random.seed(seed)
    
    # 1. Identifichiamo i path Real necessari in base ai task
    real_dirs = set()
    for task in order_list:
        real_key = task_mapping.get(task)
        if real_key and real_key in image_dir:
            real_dirs.add(image_dir[real_key])
            
    # 2. Raccogliamo le immagini
    dir_to_imgs = {}
    valid_ext = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
    for d in real_dirs:
        imgs = []
        for ext in valid_ext:
            imgs.extend(glob.glob(os.path.join(d, '**', ext), recursive=True))
        if imgs:
            imgs.sort() # Sorting per determinismo
            random.shuffle(imgs)
            dir_to_imgs[d] = imgs
            print(f" - Trovate {len(imgs)} immagini in: {d}")
            
    dirs = list(dir_to_imgs.keys())
    if not dirs:
        raise ValueError("Nessuna immagine Real trovata per i task specificati!")
        
    # 3. Estrazione Round-Robin per un bilanciamento perfetto
    selected_paths = []
    idx_map = {d: 0 for d in dirs}
    active_dirs = list(dirs)
    
    while len(selected_paths) < total_samples and active_dirs:
        for d in list(active_dirs):
            if len(selected_paths) >= total_samples:
                break
            if idx_map[d] < len(dir_to_imgs[d]):
                selected_paths.append(dir_to_imgs[d][idx_map[d]])
                idx_map[d] += 1
            else:
                active_dirs.remove(d) # Cartella esaurita
                
    print(f"[INFO] Pool completato: {len(selected_paths)} immagini Real estratte da {len(dirs)} domini.")
    return selected_paths

# ==============================================================================
# 2. GESTIONE REAL "UNIVERSAL" E CUSTOM DATASET (VERBOSE & ROBUST)
# ==============================================================================

def get_task_specific_balanced_reals(order_list, image_dir, task_mapping, total_samples=50000, seed=42):
    print("\n" + "="*50)
    print("[DEBUG] CREAZIONE POOL 'UNIVERSAL REAL'")
    print("="*50)
    np.random.seed(seed)
    random.seed(seed)
    
    dir_to_imgs = {}
    # Estensioni valide espanse per sicurezza
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    
    for task in order_list:
        real_key = task_mapping.get(task)
        print(f"\nAnalizzo Task: [{task}] -> Mappato su Real: [{real_key}]")
        
        if not real_key or real_key not in image_dir:
            print(f"  -> [ERRORE] Chiave {real_key} mancante nel dizionario IMAGE_DIR!")
            continue
            
        d = image_dir[real_key]
        print(f"  -> Cerco nel path: {d}")
        
        if not os.path.exists(d):
            print(f"  -> [ERRORE FATALE] La cartella NON ESISTE fisicamente sul server!")
            continue
            
        # Ricerca a prova di bomba in tutte le sottocartelle
        imgs = []
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.lower().endswith(valid_ext):
                    imgs.append(os.path.join(root, file))
                    
        if imgs:
            imgs.sort() # Sorting per determinismo
            random.shuffle(imgs)
            dir_to_imgs[d] = imgs
            print(f"  -> [OK] Trovate {len(imgs)} immagini valide.")
        else:
            print(f"  -> [ERRORE] Cartella trovata, ma 0 immagini valide al suo interno. Formato errato?")
            
    dirs = list(dir_to_imgs.keys())
    if not dirs:
        raise ValueError("\nERRORE CRITICO: Nessun dominio Real valido trovato. Interruzione.")
        
    print(f"\n[INFO] Trovati {len(dirs)} domini Real validi. Inizio estrazione Round-Robin per {total_samples} campioni...")
    
    selected_paths = []
    idx_map = {d: 0 for d in dirs}
    active_dirs = list(dirs)
    
    while len(selected_paths) < total_samples and active_dirs:
        for d in list(active_dirs):
            if len(selected_paths) >= total_samples:
                break
            if idx_map[d] < len(dir_to_imgs[d]):
                selected_paths.append(dir_to_imgs[d][idx_map[d]])
                idx_map[d] += 1
            else:
                active_dirs.remove(d) # Cartella esaurita
                
    print(f"[INFO] Pool completato: {len(selected_paths)} immagini estratte da {len(dirs)} domini.")
    
    # Stampa la distribuzione finale
    print("[INFO] Distribuzione finale del Pool Real:")
    for d in dirs:
        print(f" - Da {d.split('/')[-1]}: {idx_map[d]} immagini")
        
    return selected_paths

class UnifiedRealFakeDataset(Dataset):
    def __init__(self, real_paths_list, fake_dir):
        self.real_paths = real_paths_list
        self.fake_paths = []
        
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
        
        if not os.path.exists(fake_dir):
            raise ValueError(f"ERRORE FATALE DATALOADER: Il path FAKE {fake_dir} non esiste!")
            
        for root, _, files in os.walk(fake_dir):
            for file in files:
                if file.lower().endswith(valid_ext):
                    self.fake_paths.append(os.path.join(root, file))
                    
        print(f"\n[DEBUG DATALOADER] Costruzione Dataset:")
        print(f" - REAL passati al loader: {len(self.real_paths)}")
        print(f" - FAKE trovati in {fake_dir}: {len(self.fake_paths)}")
        
        if len(self.fake_paths) == 0:
            raise ValueError(f"ERRORE FATALE: Nessuna immagine Fake in {fake_dir}")
                          
        self.all_paths = self.real_paths + self.fake_paths
        self.labels = [0]*len(self.real_paths) + [1]*len(self.fake_paths)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        path = self.all_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            img = Image.new('RGB', (224, 224))
            
        img_tensor = self.transform(img)
        return img_tensor, label
from config import PRETRAINED_MODELS
# ---------------------------------------------
# A. Funzione per il Fine-Tuning della Backbone
# ---------------------------------------------
def finetune_backbone_routine(args, global_real_paths, order_list, device_obj):
    first_task_fake = order_list[0]
    print("\n" + "="*50)
    print(f">>> PHASE 1: FULL BACKBONE FINE-TUNING (Universal Reals vs {first_task_fake})")
    print("="*50)
    
    model = load_pretrained_model(PRETRAINED_MODELS['stylegan1']) 
    num_ftrs = model.resnet.fc.in_features
    model.resnet.fc = nn.Linear(num_ftrs, 2) 
    model = model.to(device_obj)
    for param in model.parameters(): param.requires_grad = True

    fake_dir = IMAGE_DIR[first_task_fake]
    dataset = UnifiedRealFakeDataset(global_real_paths, fake_dir)
    print(f"Dataset Size for Backbone FT: {len(dataset)} images (Real + Fake)")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
    scaler = torch.cuda.amp.GradScaler() 
    
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(args.epochs_ft):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device_obj), labels.to(device_obj)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device_obj), labels.to(device_obj)
                with torch.cuda.amp.autocast(): outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{args.epochs_ft} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    print(f"Fine-tuning complete. Best Acc: {best_acc:.2f}%")
    model.load_state_dict(best_weights)
    save_path = f"checkpoint/resnet50_finetuned_on_{first_task_fake}.pth"
    torch.save({'state_dict': model.state_dict()}, save_path)
    return save_path

# ---------------------------------------------
# B. Pre-computazione Features 
# ---------------------------------------------
def precompute_all_features(backbone_path, global_real_paths, order_list, device_obj, batch_size=64, num_workers=8):
    print("\n" + "="*50)
    print(">>> PHASE 2: PRE-COMPUTING FEATURES (Universal Reals)")
    print("="*50)
    
    from src.net import ResNet50BC
    backbone_net = ResNet50BC()
    checkpoint = torch.load(backbone_path, map_location=device_obj)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    backbone_net.load_state_dict(state_dict, strict=False) 
    backbone_net.resnet.fc = nn.Identity()
    backbone_net.to(device_obj).eval().half()
    
    feat_folder = "./feature_resnet50_universal"
    os.makedirs(feat_folder, exist_ok=True)
    
    for task in order_list:
        feature_file = os.path.join(feat_folder, f"feats_{task}.pt")
        if os.path.exists(feature_file):
            print(f"Features for {task} exist. Skipping.")
            continue
            
        print(f"Extracting features for {task}...")
        fake_dir = IMAGE_DIR[task]
        
        dataset = UnifiedRealFakeDataset(global_real_paths, fake_dir)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        
        try:
            with torch.cuda.amp.autocast():
                extract_and_save_features(backbone_net, loader, feature_file, device_obj)
        except Exception as e:
            print(f"ERROR extracting {task}: {e}")

# ---------------------------------------------
# C. Pipeline Continual Learning
# ---------------------------------------------
def fine_tune_cl(
    model_path: str, batch_size: int = 32, num_workers: int = 8, device: str = '0',
    epochs: int = 300, lr: float = 1e-3, seed: int = 42,
    fine_tuning_on: str = 'stylegan2', num_anchors: int = 5000,
    checkpoint_file: str = "checkpoint/checkpoint_HELLO.pth",
    load_checkpoint: bool = False, order_list: list = None,
    csv_log_path: str = "results_table_dogan_all_classes.csv", **kwargs
):
    device_obj = torch.device("cuda:"+device if torch.cuda.is_available() else "cpu")
    print(f"--- CL Step: Training on {fine_tuning_on} ---")
    torch.manual_seed(seed); np.random.seed(seed)
    
    feat_folder = "./feature_resnet50_universal"
    train_feat_file = os.path.join(feat_folder, f"feats_{fine_tuning_on}.pt")
    
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
        classifier.load_state_dict(torch.load(checkpoint_file)['state_dict'])

    # 3. Training Loop
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device_obj)
    scaler = torch.cuda.amp.GradScaler()

    train_loader = DataLoader(TensorDataset(f_train, l_train), batch_size=batch_size, shuffle=True)
    
    classifier.train()
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

    # 4. Valutazione Multi-Task
    classifier.eval()
    all_results = {}
    csv_row = {'Train_Step': fine_tuning_on}
    res_dir = os.path.join("results_dogan_all_classes", f"train_on_{fine_tuning_on}")
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
                _, acc, preds, _ = evaluate3(classifier, t_loader, criterion, device_obj)
        
        all_results[t_task] = acc
        csv_row[t_task] = acc
        
        if isinstance(preds, torch.Tensor): y_pred = preds.cpu().numpy()
        else: y_pred = preds
        cm = confusion_matrix(lt_test, y_pred)
        np.savetxt(os.path.join(res_dir, f"cm_{t_task}.txt"), cm, fmt='%d')
        print(f"Task {t_task:20} | Acc: {acc:.4f}")

    # 5. Salvataggio Riga nel CSV
    df_row = pd.DataFrame([csv_row])
    header = not os.path.exists(csv_log_path)
    df_row.to_csv(csv_log_path, mode='a', header=header, index=False)
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
    parser.add_argument('--epochs', type=int, default=10) # CL classifier
    parser.add_argument('--epochs_ft', type=int, default=10) # Backbone FT
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_anchors', type=int, default=5000)
    parser.add_argument('--num_real_samples', type=int, default=50000)
    
    parser.add_argument('--do_backbone_finetuning', action='store_true')
    parser.add_argument('--order', default='[fake_progan256, fake_cycle_gan, fake_progan1024, fake_stargan]', help="Lista dei task")
    parser.add_argument('--csv_log', type=str, default='cl_results_table.csv')

    args = parser.parse_args()
    order_list = re.sub(r'[\[\]\s]', '', args.order).split(',')
    verify_datasets_integrity(order_list, IMAGE_DIR, TASK_REAL_MAPPING)
    device_obj = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
    
    order_list = re.sub(r'[\[\]\s]', '', args.order).split(',')

    # 0. GESTIONE GLOBAL REAL PATHS (Costruiti solo sui task scelti)
    global_real_paths = get_task_specific_balanced_reals(
        order_list=order_list, 
        image_dir=IMAGE_DIR, 
        task_mapping=TASK_REAL_MAPPING, 
        total_samples=args.num_real_samples, 
        seed=args.seed
    )

    # 1. Backbone
    os.makedirs("checkpoint", exist_ok=True)
    first_task = order_list[0]
    backbone_model_path = f"checkpoint/resnet50_finetuned_on_{first_task}.pth"
    
    if args.do_backbone_finetuning or not os.path.exists(backbone_model_path):
        backbone_model_path = finetune_backbone_routine(args, global_real_paths, order_list, device_obj)
    
    # 2. Pre-compute Features
    precompute_all_features(backbone_model_path, global_real_paths, order_list, device_obj, 
                            batch_size=args.batch_size, num_workers=args.num_workers)

    # 3. CL Loop
    print(f"\n=== Starting Relative Representation CL ===")
    if os.path.exists(args.csv_log): os.remove(args.csv_log)
    
    prev_ckpt = None
    for i, task in enumerate(order_list):
        print(f"\n>>> CL STEP {i+1}: {task}")
        current_ckpt = os.path.join("checkpoint", f"cl_step_{i}_{task}.pth")
        
        cl_args = vars(args)
        cl_args['fine_tuning_on'] = task
        cl_args['checkpoint_file'] = current_ckpt
        cl_args['load_checkpoint'] = (i > 0)
        
        if i > 0: shutil.copy(prev_ckpt, current_ckpt)
        
        fine_tune_cl(model_path=backbone_model_path, order_list=order_list, csv_log_path=args.csv_log, **cl_args)
        prev_ckpt = current_ckpt