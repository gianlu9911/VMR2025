#!/usr/bin/env python3
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
# 1. CONFIGURAZIONE DATASET
# ==============================================================================

IMAGE_DIR = {
    # REAL
    'real_cycle_gan' : '/seidenas/datasets/DoGANs/new/Pristine/cycleGAN/horse2zebra', 
    'real_progan256' : '/seidenas/datasets/DoGANs/new/Pristine/GGAN256/lsun_tower',
    'real_progan1024': '/seidenas/datasets/DoGANs/new/Pristine/GGAN1024/celebhq',
    'real_glow'      : '/seidenas/datasets/DoGANs/new/Pristine/glow/Male',
    'real_stargan'   : '/seidenas/datasets/DoGANs/new/Pristine/starGAN/Brown_Hair',
    
    # FAKE
    'fake_cycle_gan' : '/seidenas/datasets/DoGANs/new/Generated/CycleGAN/horse2zebra',
    'fake_progan256' : '/seidenas/datasets/DoGANs/new/Generated/GGAN256/lsun_tower',
    'fake_progan1024': '/seidenas/datasets/DoGANs/new/Generated/GGAN1024/celebhq',
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
# 2. UTILS DATI E DATASET (Adattati per iCaRL Buffer)
# ==============================================================================

def get_task_specific_balanced_reals(order_list, image_dir, task_mapping, total_samples=50000, seed=42):
    print("\n" + "="*50)
    print("[DEBUG] CREAZIONE POOL 'UNIVERSAL REAL'")
    print("="*50)
    np.random.seed(seed)
    random.seed(seed)
    
    dir_to_imgs = {}
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    
    for task in order_list:
        real_key = task_mapping.get(task)
        if not real_key or real_key not in image_dir: continue
        d = image_dir[real_key]
        if not os.path.exists(d): continue
            
        imgs = []
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.lower().endswith(valid_ext):
                    imgs.append(os.path.join(root, file))
                    
        if imgs:
            imgs.sort()
            random.shuffle(imgs)
            dir_to_imgs[d] = imgs
            
    dirs = list(dir_to_imgs.keys())
    if not dirs:
        raise ValueError("\nERRORE CRITICO: Nessun dominio Real valido trovato.")
        
    print(f"\n[INFO] Trovati {len(dirs)} domini Real validi. Estrazione di {total_samples} campioni...")
    
    selected_paths = []
    idx_map = {d: 0 for d in dirs}
    active_dirs = list(dirs)
    
    while len(selected_paths) < total_samples and active_dirs:
        for d in list(active_dirs):
            if len(selected_paths) >= total_samples: break
            if idx_map[d] < len(dir_to_imgs[d]):
                selected_paths.append(dir_to_imgs[d][idx_map[d]])
                idx_map[d] += 1
            else:
                active_dirs.remove(d)
                
    return selected_paths

def get_task_fake_paths(fake_dir):
    """Estrae tutte le immagini fake da una cartella."""
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    fake_paths = []
    if not os.path.exists(fake_dir):
        raise ValueError(f"ERRORE FATALE: Il path FAKE {fake_dir} non esiste!")
        
    for root, _, files in os.walk(fake_dir):
        for file in files:
            if file.lower().endswith(valid_ext):
                fake_paths.append(os.path.join(root, file))
    if len(fake_paths) == 0:
        raise ValueError(f"ERRORE FATALE: Nessuna immagine Fake in {fake_dir}")
    return fake_paths

class UnifiedRealFakeDataset(Dataset):
    def __init__(self, real_paths_list, fake_paths_list, balance=True):
        self.real_paths = copy.deepcopy(real_paths_list)
        self.fake_paths = copy.deepcopy(fake_paths_list)
        
        if balance:
            min_len = min(len(self.real_paths), len(self.fake_paths))
            self.real_paths = random.sample(self.real_paths, min_len)
            self.fake_paths = random.sample(self.fake_paths, min_len)
                          
        self.all_paths = self.real_paths + self.fake_paths
        self.labels = [0]*len(self.real_paths) + [1]*len(self.fake_paths)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
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
        except:
            img = Image.new('RGB', (224, 224))
        return self.transform(img), label

# ==============================================================================
# 3. iCaRL FINE-TUNING LOOP (Knowledge Distillation + Buffer)
# ==============================================================================

def train_icarl_network(
    old_model_path, current_task, current_fake_paths, buffer_fake_paths, global_real_paths, order_list, device_obj,
    batch_size=32, num_workers=8, epochs=10, lr=1e-4, seed=42, 
    kd_temp=2.0, kd_alpha=0.5, checkpoint_file="checkpoint.pth", csv_log_path="results.csv"
):
    print("\n" + "="*50)
    print(f">>> iCaRL TRAINING SU: {current_task} (Buffer Fakes: {len(buffer_fake_paths)})")
    print("="*50)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1. Inizializzazione Rete Corrente
    model = ResNet50BC()
    num_ftrs = model.resnet.fc.in_features
    model.resnet.fc = nn.Linear(num_ftrs, 2)
    
    old_model = None
    if old_model_path and os.path.exists(old_model_path):
        print(f"Caricamento pesi correnti (Continual Learning) da: {old_model_path}")
        checkpoint = torch.load(old_model_path, map_location=device_obj)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        
        # In iCaRL, creiamo una copia congelata del modello vecchio per la distillazione
        print("Congelamento 'Old Model' per Knowledge Distillation...")
        old_model = ResNet50BC()
        old_model.resnet.fc = nn.Linear(num_ftrs, 2)
        old_model.load_state_dict(state_dict, strict=False)
        old_model = old_model.to(device_obj)
        old_model.eval() # Sempre in eval
        for param in old_model.parameters():
            param.requires_grad = False
    else:
        print("Nessun modello precedente. Partenza da ImageNet (Task 1). Nessuna distillazione.")
        
    model = model.to(device_obj)
    for param in model.parameters(): 
        param.requires_grad = True

    # 2. Setup Dataloader (Fakes correnti + Fakes dal Buffer)
    combined_fake_paths = current_fake_paths + buffer_fake_paths
    dataset = UnifiedRealFakeDataset(global_real_paths, combined_fake_paths, balance=True)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, _ = torch.utils.data.random_split(dataset, [train_size, test_size], generator=gen)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    scaler = torch.cuda.amp.GradScaler()

    # 3. Training Loop con Distillation Loss
    print(f"Training iCaRL per {epochs} epoche...")
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
                # Applica Knowledge Distillation se abbiamo il modello vecchio
                if old_model is not None:
                    with torch.no_grad():
                        old_outputs = old_model(imgs)
                    
                    # Logits soft per la Distillazione
                    log_scores_new = F.log_softmax(outputs / kd_temp, dim=1)
                    scores_old = F.softmax(old_outputs / kd_temp, dim=1)
                    
                    loss_kd = criterion_kd(log_scores_new, scores_old) * (kd_temp ** 2)
                    loss = loss_cls + (kd_alpha * loss_kd)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")

    # Salvataggio
    torch.save({'state_dict': model.state_dict()}, checkpoint_file)

    # 4. Valutazione su TUTTI i task
    model.eval()
    csv_row = {'Train_Step': current_task}
    res_dir = os.path.join("results", f"train_on_{current_task}")
    os.makedirs(res_dir, exist_ok=True)

    print(f"\n--- Valutazione Multi-Task ---")
    for t_task in order_list:
        t_fake_paths = get_task_fake_paths(IMAGE_DIR[t_task])
        
        try:
            t_dataset = UnifiedRealFakeDataset(global_real_paths, t_fake_paths, balance=True)
            t_train_size = int(0.8 * len(t_dataset))
            t_test_size = len(t_dataset) - t_train_size
            
            gen_test = torch.Generator().manual_seed(seed)
            _, t_test_ds = torch.utils.data.random_split(t_dataset, [t_train_size, t_test_size], generator=gen_test)
            
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
            np.savetxt(os.path.join(res_dir, f"cm_{t_task}.txt"), cm, fmt='%d') # <- SALVATAGGIO CONFUSION MATRIX!
            
            mark = "(*)" if t_task == current_task else ""
            print(f"Task {t_task:20} | Test Acc: {acc:.4f} {mark}")
            
        except Exception as e:
            print(f"Errore in valutazione su {t_task}: {e}")
            csv_row[t_task] = 0.0

    # 5. Salvataggio su CSV
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
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_real_samples', type=int, default=50000)
    
    # Parametri specifici iCaRL
    parser.add_argument('--memory_size', type=int, default=0, help="Quante immagini FAKE tenere totali nel buffer")
    parser.add_argument('--kd_temp', type=float, default=2.0, help="Temperatura per la Knowledge Distillation")
    parser.add_argument('--kd_alpha', type=float, default=0.5, help="Peso della Knowledge Distillation rispetto alla CrossEntropy")
    
    parser.add_argument('--order', default='[fake_cycle_gan,fake_progan256,fake_progan1024, fake_glow, fake_stargan]', help="Lista dei task")
    parser.add_argument('--csv_log', type=str, default='icarl_results_table_dogan.csv')

    args = parser.parse_args()
    device_obj = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
    
    order_list = re.sub(r'[\[\]\s]', '', args.order).split(',')

    # 1. Costruzione Universal Reals
    global_real_paths = get_task_specific_balanced_reals(
        order_list=order_list, 
        image_dir=IMAGE_DIR, 
        task_mapping=TASK_REAL_MAPPING, 
        total_samples=args.num_real_samples, 
        seed=args.seed
    )

    # 2. Inizializzazione iCaRL
    os.makedirs("checkpoint", exist_ok=True)
    if os.path.exists(args.csv_log): os.remove(args.csv_log)
    
    current_model_path = "" 
    buffer_fake_paths = [] # La nostra memoria Episodica iCaRL
    memory_per_task = args.memory_size // len(order_list) # Distribuzione equa della memoria
    
    for i, task in enumerate(order_list):
        current_ckpt = os.path.join("checkpoint", f"icarl_step_{i}_{task}.pth")
        
        # Carica Fakes correnti
        current_fake_paths = get_task_fake_paths(IMAGE_DIR[task])
        
        current_model_path = train_icarl_network(
            old_model_path=current_model_path,
            current_task=task,
            current_fake_paths=current_fake_paths,
            buffer_fake_paths=buffer_fake_paths, # Passiamo il buffer
            global_real_paths=global_real_paths,
            order_list=order_list,
            device_obj=device_obj,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            kd_temp=args.kd_temp,
            kd_alpha=args.kd_alpha,
            checkpoint_file=current_ckpt,
            csv_log_path=args.csv_log
        )
        
        # AGGIORNAMENTO BUFFER (Exemplars Management)
        # Preleviamo 'memory_per_task' immagini dal task appena finito e le salviamo per i prossimi
        random.seed(args.seed + i)
        num_to_add = min(memory_per_task, len(current_fake_paths))
        sampled_fakes = random.sample(current_fake_paths, num_to_add)
        buffer_fake_paths.extend(sampled_fakes)
        print(f"[iCaRL] Aggiunte {num_to_add} immagini al buffer. Nuova dimensione memoria: {len(buffer_fake_paths)}")