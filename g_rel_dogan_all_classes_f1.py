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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from PIL import Image

warnings.filterwarnings("ignore")

import torch
import collections
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms

# --- Avalanche Imports ---
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.supervised import Naive

# === Project imports ===
from src.net import load_pretrained_model
from src.utils import RelativeRepresentation, RelClassifier, extract_and_save_features
from config import PRETRAINED_MODELS, IMAGE_DIR
# ==============================================================================
# 1. CONFIGURAZIONE DATASET
# ==============================================================================

IMAGE_DIR = {
    'real_cycle_gan' : '/seidenas/datasets/DoGANs/new/Pristine/CycleGAN/horse2zebra', 
    'real_progan256' : '/seidenas/datasets/DoGANs/new/Pristine/GGAN256/lsun_tower',
    'real_progan1024': '/seidenas/datasets/DoGANs/new/Pristine/GGAN1024/HQ-IMG',
    #'real_glow'      : '/seidenas/datasets/DoGANs/new/Pristine/glow/Male',
    'real_stargan'   : '/seidenas/datasets/DoGANs/new/Pristine/starGAN/celeba256',
    
    'fake_cycle_gan' : '/seidenas/datasets/DoGANs/new/Generated/CycleGAN/horse2zebra',
    'fake_progan256' : '/seidenas/datasets/DoGANs/new/Generated/GGAN256/lsun_tower',
    'fake_progan1024': '/seidenas/datasets/DoGANs/new/Generated/GGAN1024/celebhq',
    #'fake_glow'      : '/seidenas/datasets/DoGANs/new/Generated/glow/Male',
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
# 2. VERIFICA E GESTIONE DATASET
# ==============================================================================

def verify_datasets_integrity(order_list, image_dir, task_mapping):
    print("\n" + "="*50)
    print("[VERIFICA] Controllo integrità preventivo dei dataset...")
    print("="*50)
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    
    for task in order_list:
        if task not in image_dir:
            raise ValueError(f"\n[ERRORE] Il task fake '{task}' non è in IMAGE_DIR!")
        
        fake_path = image_dir[task]
        if not os.path.exists(fake_path):
            raise ValueError(f"\n[ERRORE] La cartella FAKE non esiste: {fake_path}")
            
        fake_imgs = [f for root, _, files in os.walk(fake_path) for f in files if f.lower().endswith(valid_ext)]
        if len(fake_imgs) == 0:
            raise ValueError(f"\n[ERRORE] La cartella FAKE è VUOTA: {fake_path}")
        print(f" [OK] {task:15} -> {len(fake_imgs)} img fake.")

        real_key = task_mapping.get(task)
        if not real_key or real_key not in image_dir:
            raise ValueError(f"\n[ERRORE] Nessun mapping reale per '{task}'!")
            
        real_path = image_dir[real_key]
        if not os.path.exists(real_path):
            raise ValueError(f"\n[ERRORE] La cartella REAL non esiste: {real_path}")
            
        real_imgs = [f for root, _, files in os.walk(real_path) for f in files if f.lower().endswith(valid_ext)]
        if len(real_imgs) == 0:
            raise ValueError(f"\n[ERRORE] La cartella REAL è VUOTA: {real_path}")
        print(f" [OK] {real_key:15} -> {len(real_imgs)} img reali.")

    print("\n[VERIFICA COMPLETATA] Tutti i path sono validi!\n")

def get_task_specific_balanced_reals(order_list, image_dir, task_mapping, total_samples=50000, seed=42):
    np.random.seed(seed); random.seed(seed)
    dir_to_imgs = {}
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    
    for task in order_list:
        d = image_dir[task_mapping.get(task)]
        imgs = [os.path.join(root, file) for root, _, files in os.walk(d) for file in files if file.lower().endswith(valid_ext)]
        imgs.sort(); random.shuffle(imgs)
        dir_to_imgs[d] = imgs
            
    dirs = list(dir_to_imgs.keys())
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

class UnifiedRealFakeDataset(Dataset):
    def __init__(self, real_paths_list, fake_dir):
        self.real_paths = real_paths_list
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
        self.fake_paths = [os.path.join(root, file) for root, _, files in os.walk(fake_dir) for file in files if file.lower().endswith(valid_ext)]
        self.all_paths = self.real_paths + self.fake_paths
        self.labels = [0]*len(self.real_paths) + [1]*len(self.fake_paths)
        print(f"\n[DEBUG DATASET] Cartella: {os.path.basename(fake_dir)}")
        print(f"   -> Reali (Classe 0): {len(self.real_paths)}")
        print(f"   -> Fake  (Classe 1): {len(self.fake_paths)}")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def __len__(self): return len(self.all_paths)
    def __getitem__(self, idx):
        try: img = Image.open(self.all_paths[idx]).convert('RGB')
        except: img = Image.new('RGB', (224, 224))
        return self.transform(img), self.labels[idx]

# ==============================================================================
# 3. BACKBONE E FEATURE EXTRACTION
# ==============================================================================
def finetune_backbone_routine(args, global_real_paths, order_list, device_obj):
    first_task_fake = order_list[0]
    print(f"\n>>> PHASE 1: BACKBONE FINE-TUNING (Reals vs {first_task_fake})")
    
    model = load_pretrained_model(PRETRAINED_MODELS['stylegan1']) 
    model.resnet.fc = nn.Linear(model.resnet.fc.in_features, 2) 
    model = model.to(device_obj)
    
    dataset = UnifiedRealFakeDataset(global_real_paths, IMAGE_DIR[first_task_fake])
    train_ds, val_ds = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
    best_acc = 0.0; best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(args.epochs_ft):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device_obj), labels.to(device_obj)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                _, predicted = torch.max(model(imgs.to(device_obj)), 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.to(device_obj)).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1} | Val Acc: {val_acc:.2f}%")
        if val_acc > best_acc: best_acc = val_acc; best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    save_path = f"checkpoint/resnet50_finetuned_{first_task_fake}.pth"
    torch.save({'state_dict': model.state_dict()}, save_path)
    return save_path

def precompute_all_features(backbone_path, global_real_paths, order_list, device_obj, batch_size=64, num_workers=8):
    print("\n>>> PHASE 2: PRE-COMPUTING FEATURES")
    from src.net import ResNet50BC
    backbone_net = ResNet50BC()
    ckpt = torch.load(backbone_path, map_location=device_obj)
    backbone_net.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt, strict=False) 
    backbone_net.resnet.fc = nn.Identity()
    backbone_net.to(device_obj).eval()
    
    os.makedirs("./feature_resnet50_universal", exist_ok=True)
    for task in order_list:
        f_file = f"./feature_resnet50_universal/feats_{task}.pt"
        if not os.path.exists(f_file):
            loader = DataLoader(UnifiedRealFakeDataset(global_real_paths, IMAGE_DIR[task]), batch_size=batch_size, num_workers=num_workers)
            extract_and_save_features(backbone_net, loader, f_file, device_obj)

# ==============================================================================
# 4. AVALANCHE CONTINUAL LEARNING & CUSTOM METRICS
# ==============================================================================

def run_avalanche_cl_sequence(
    order_list: list, run_id: str, args, device_obj
):
    print(f"\n{'='*60}\n=== INIZIO {run_id} | Sequenza: {order_list} ===\n{'='*60}")
    
    # 1. Carica le Ancore dal PRIMO TASK
    data_first = torch.load(f"./feature_resnet50_universal/feats_{order_list[0]}.pt")
    feats_full, labels_full = data_first["features"].cpu().float(), data_first["labels"].squeeze().cpu()
    
    real_mask = (labels_full == 0)
    real_feats = feats_full[real_mask]
    rng = torch.Generator().manual_seed(args.seed)
    idx = torch.randint(low=0, high=len(real_feats), size=(args.num_anchors,), generator=rng) if args.num_anchors > len(real_feats) else torch.randperm(len(real_feats), generator=rng)[:args.num_anchors]
    anchors = real_feats[idx]
    
    rel_module = RelativeRepresentation(anchors.to(device_obj))
    
    # 2. Prepara i Dataset per Avalanche 
    train_datasets = []
    test_datasets = []
    
    for task in order_list:
        data = torch.load(f"./feature_resnet50_universal/feats_{task}.pt")
        f_all, l_all = data["features"].cpu().float(), data["labels"].squeeze().cpu()
        
        f_train, f_test, l_train, l_test = train_test_split(f_all.numpy(), l_all.numpy(), test_size=0.2, random_state=args.seed, stratify=l_all.numpy())
        
        train_datasets.append(TensorDataset(torch.from_numpy(f_train), torch.from_numpy(l_train)))
        test_datasets.append(TensorDataset(torch.from_numpy(f_test), torch.from_numpy(l_test)))

    # 3. Inizializza Avalanche Scenario e Classifier
    scenario = dataset_benchmark(train_datasets=train_datasets, test_datasets=test_datasets)
    classifier = RelClassifier(rel_module, anchors.size(0), num_classes=2).to(device_obj)
    
    strategy = Naive(
        model=classifier,
        optimizer=torch.optim.Adam(classifier.parameters(), lr=args.lr),
        criterion=nn.CrossEntropyLoss(),
        train_mb_size=args.batch_size,
        train_epochs=args.epochs,
        eval_mb_size=args.batch_size,
        device=device_obj,
        evaluator=None # Spegniamo il logger di default di Avalanche
    )

    # Matrici per BWT/FWT in memoria: T x T
    T = len(order_list)
    acc_matrix = np.full((T, T), np.nan)
    auc_matrix = np.full((T, T), np.nan)

    # Inizializza il file CSV dettagliato (Step-by-step)
    detailed_csv = "detailed_metrics_results.csv"
    if not os.path.exists(detailed_csv) or os.path.getsize(detailed_csv) == 0:
        with open(detailed_csv, 'a') as f: f.write("Run_ID,Train_Task,Eval_Task,Accuracy,Precision,Recall,F1_Score,AUC\n")

    # 4. LOOP DI CONTINUAL LEARNING
    for exp_idx, experience in enumerate(scenario.train_stream):
        train_task = order_list[exp_idx]
        print(f"\n--- Train sull'Esperienza {exp_idx}: {train_task} ---")
        
        strategy.train(experience)
        
        # 5. VALUTAZIONE SU TUTTI I TASK
        strategy.model.eval()
        
        with open(detailed_csv, 'a') as f_csv:
            for test_idx, test_dataset in enumerate(test_datasets):
                test_task = order_list[test_idx]
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                
                all_labels, all_preds, all_probs = [], [], []
                
                with torch.no_grad():
                    for batch_f, batch_l in test_loader:
                        outputs = strategy.model(batch_f.to(device_obj)) 
                        probs = torch.softmax(outputs, dim=1)[:, 1]
                        _, preds = torch.max(outputs, 1)
                        
                        all_labels.extend(batch_l.numpy())
                        all_preds.extend(preds.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
                
                # Calcolo metriche
                counter = collections.Counter(all_labels)
                print(f"  -> [DEBUG TEST] {test_task:20} | Reali: {counter[0]} | Fake: {counter[1]}")
                acc = accuracy_score(all_labels, all_preds)
                prec = precision_score(all_labels, all_preds, zero_division=0)
                rec = recall_score(all_labels, all_preds, zero_division=0)
                f1 = f1_score(all_labels, all_preds, zero_division=0)
                try: auc_score = roc_auc_score(all_labels, all_probs)
                except ValueError: auc_score = 0.0

                cm = confusion_matrix(all_labels, all_preds)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn, fp, fn, tp = -1, -1, -1, -1  # Caso in cui manca una classe
                os.makedirs("confusion_matrices", exist_ok=True)
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - {train_task} -> {test_task}')
                plt.savefig(f"confusion_matrices/cm_{run_id}_{train_task}_to_{test_task}.png")
                plt.close()
                
                # Inserimento nelle Matrici
                acc_matrix[exp_idx, test_idx] = acc
                auc_matrix[exp_idx, test_idx] = auc_score
                
                print(f"  -> Test su {test_task:20} | Acc: {acc:.4f} | AUC: {auc_score:.4f}")
                
                # Salva log step by step
                f_csv.write(f"{run_id},{train_task},{test_task},{acc:.5f},{prec:.5f},{rec:.5f},{f1:.5f},{auc_score:.5f}\n")

        # ---> NOVITÀ: Stampa la Mean Accuracy e la Mean AUC alla fine di ogni singolo step di training!
        step_mean_acc = np.nanmean(acc_matrix[exp_idx, :])
        step_mean_auc = np.nanmean(auc_matrix[exp_idx, :])
        print(f"\n  ==> AVERAGE PERFORMANCE AFTER STEP {exp_idx} ({train_task}):")
        print(f"      Mean Accuracy: {step_mean_acc:.4f} | Mean AUC: {step_mean_auc:.4f}\n")

    # 6. CALCOLO METRICHE RIASSUNTIVE (A6, BWT, FWT)
    print(f"\n[*] Risultati finali per {run_id}:")
    
    # Accuratezza
    acc_a6 = np.nanmean(acc_matrix[-1, :]) if T > 0 else 0.0
    acc_bwt = np.nanmean([acc_matrix[-1, i] - acc_matrix[i, i] for i in range(T - 1)]) if T > 1 else 0.0
    acc_fwt = np.nanmean([acc_matrix[j-1, j] for j in range(1, T)]) if T > 1 else 0.0
    
    # AUC
    auc_a6 = np.nanmean(auc_matrix[-1, :]) if T > 0 else 0.0
    auc_bwt = np.nanmean([auc_matrix[-1, i] - auc_matrix[i, i] for i in range(T - 1)]) if T > 1 else 0.0
    auc_fwt = np.nanmean([auc_matrix[j-1, j] for j in range(1, T)]) if T > 1 else 0.0

    # ---> NOVITÀ: Stampo la matrice completa delle accuracy di tutti gli step per farti vedere cosa succede!
    print("\n--- MATRICE ACCURACY COMPLETA (T x T) ---")
    print(np.round(acc_matrix, 4))
    print("-----------------------------------------\n")

    print(f"  ACCURACY -> Avg Final (A_T / Mean Acc): {acc_a6:.4f} | BWT: {acc_bwt:.4f} | FWT: {acc_fwt:.4f}")
    print(f"  AUC      -> Avg Final (A_T / Mean AUC): {auc_a6:.4f} | BWT: {auc_bwt:.4f} | FWT: {auc_fwt:.4f}")

    # Salva le metriche riassuntive in un CSV separato
    summary_csv = "summary_continual_metrics.csv"
    if not os.path.exists(summary_csv):
        with open(summary_csv, 'w') as f: f.write("Run_ID,Acc_A6,Acc_BWT,Acc_FWT,AUC_A6,AUC_BWT,AUC_FWT\n")
    with open(summary_csv, 'a') as f:
        f.write(f"{run_id},{acc_a6:.5f},{acc_bwt:.5f},{acc_fwt:.5f},{auc_a6:.5f},{auc_bwt:.5f},{auc_fwt:.5f}\n")


# ==============================================================================
# MAIN
# ==============================================================================
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

    args = parser.parse_args()
    device_obj = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
    
    # LISTA DELLE SEQUENZE DI TRAINING
    sequences = [
        ['fake_progan256', 'fake_cycle_gan', 'fake_progan1024', 'fake_stargan'],
        #['fake_cycle_gan', , 'fake_progan256', 'fake_stargan', 'fake_progan1024'],
        # Aggiungi qui le altre tue sequenze...
    ]

    for run_idx, order_list in enumerate(sequences):
        run_id = f"Run_{run_idx + 1}"
        
        # 0. Verifica integrità
        #verify_datasets_integrity(order_list, IMAGE_DIR, TASK_REAL_MAPPING)

        # 1. Pool Universale Reals
        global_real_paths = get_task_specific_balanced_reals(
            order_list=order_list, image_dir=IMAGE_DIR, task_mapping=TASK_REAL_MAPPING, 
            total_samples=args.num_real_samples, seed=args.seed
        )

        # 2. Backbone Fine-Tuning (Fatto 1 volta sola per il primo task della Run)
        os.makedirs("checkpoint", exist_ok=True)
        backbone_model_path = f"checkpoint/resnet50_finetuned_{order_list[0]}.pth"
        if args.do_backbone_finetuning or not os.path.exists(backbone_model_path):
            backbone_model_path = finetune_backbone_routine(args, global_real_paths, order_list, device_obj)
        
        # 3. Estrazione Features
        precompute_all_features(backbone_model_path, global_real_paths, order_list, device_obj, args.batch_size, args.num_workers)

        # 4. Avalanche Continual Learning Loop
        run_avalanche_cl_sequence(order_list, run_id, args, device_obj)