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
import re 

# === Project imports - adjust these to your repo layout ===
from config import PRETRAINED_MODELS, IMAGE_DIR_DOGAN
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model
from src.utils import  BalancedBatchSampler, RelativeRepresentation, RelClassifier
from src.utils import extract_and_save_features2 as extract_and_save_features
from src.g_utils import train_one_epoch3 as train_one_epoch
from src.g_utils import evaluate4, save_features_only
from src.net import ResNet50BC

def train_backbone(backbone_path, fine_tuning_on='stylegan2', batch_size=64, device='cuda', epochs=1):
    """
    Train a backbone con supporto AMP e gestione corretta dello scheduler.
    """
    print(f"Training backbone for fine-tuning on {fine_tuning_on}...")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    train_dataset = RealSynthethicDataloader(
        IMAGE_DIR_DOGAN['real_DoGAN_facades'], 
        IMAGE_DIR_DOGAN[fine_tuning_on], 
        split=''
    )
    # Aumenta num_workers se possibile (es. 4 o 8) per non strozzare la GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    backbone = ResNet50BC().to(device)
    
    # Freeze logic
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Assicurati che l'ultimo layer (fc) sia trainable
    # Nota: verifica se il tuo ResNet50BC chiama il layer finale 'fc' o 'classifier'
    for param in backbone.resnet.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, backbone.parameters()), 
                                lr=1e-3, momentum=0.9)
    
    # T_max settato sul numero di epoche
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() # AMP

    backbone.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Autocast per AMP
            with torch.cuda.amp.autocast():
                outputs = backbone(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step() # Lo step va qui, a fine epoca
        print(f"Epoch {epoch+1}/{epochs}: Loss: {running_loss / len(train_loader):.4f}")

    torch.save({'state_dict': backbone.state_dict()}, backbone_path)
    print(f"Backbone saved to {backbone_path}")
# ---------------------------------------------
# Fine-tuning pipeline (main)
# ---------------------------------------------
#!/usr/bin/env python3
import os
import time
import warnings
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# === Project imports ===
from config import PRETRAINED_MODELS, IMAGE_DIR_DOGAN
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model
from src.utils import BalancedBatchSampler, RelativeRepresentation, RelClassifier
from src.utils import extract_and_save_features2 as extract_and_save_features
from src.g_utils import train_one_epoch3 as train_one_epoch
from src.g_utils import evaluate4

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------------------------------
# 1. Feature Management & Splitting Logic
# ---------------------------------------------
def get_split_features(
    backbone_net, 
    real_dir, 
    fake_dir, 
    feature_dir, 
    task_name, 
    device, 
    batch_size=32, 
    num_workers=4, 
    test_ratio=0.2,   # 20% dei dati per il test se non esiste split
    seed=42,
    force_recompute=False
):
    """
    Gestisce la creazione di Train/Test set fissi.
    Se i file .pt esistono, li carica.
    Se non esistono, estrae tutto, splitta determinísticamente e salva.
    """
    train_pt_path = os.path.join(feature_dir, f"train_{task_name}_features.pt")
    test_pt_path = os.path.join(feature_dir, f"test_{task_name}_features.pt")

    # A. Se esistono entrambi, carichiamo e basta
    if os.path.exists(train_pt_path) and os.path.exists(test_pt_path) and not force_recompute:
        print(f"Loading cached SPLIT features for {task_name}...")
        d_train = torch.load(train_pt_path)
        d_test = torch.load(test_pt_path)
        return d_train['features'], d_train['labels'], d_test['features'], d_test['labels']

    # B. Se non esistono, dobbiamo crearli partendo da TUTTI i dati
    print(f"Creating NEW persistent train/test split for {task_name}...")
    
    # 1. Carichiamo tutto il dataset (split='' o None per prendere tutto)
    full_dataset = RealSynthethicDataloader(real_dir, fake_dir, split='') 
    loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # 2. Estraiamo tutte le features (usiamo un nome temporaneo)
    temp_full_path = os.path.join(feature_dir, f"temp_full_{task_name}.pt")
    feats_all, labels_all, _ = extract_and_save_features(backbone_net, loader, temp_full_path, device)
    
    # 3. Mescolamento deterministico
    num_samples = len(feats_all)
    indices = np.arange(num_samples)
    # Usiamo numpy RandomState per isolare il seed da tutto il resto
    rs = np.random.RandomState(seed)
    rs.shuffle(indices)
    
    # 4. Calcolo split index
    test_size = int(num_samples * test_ratio)
    # Assicuriamoci di avere almeno qualche campione per entrambi
    if test_size < 1: test_size = 1
    if test_size > num_samples - 1: test_size = num_samples // 2
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # 5. Creazione tensori
    feats_train = feats_all[train_indices]
    labels_train = labels_all[train_indices]
    
    feats_test = feats_all[test_indices]
    labels_test = labels_all[test_indices]
    
    # 6. Salvataggio su disco (così rimarranno identici per sempre)
    print(f"Saving split: Train samples: {len(feats_train)}, Test samples: {len(feats_test)}")
    torch.save({'features': feats_train, 'labels': labels_train}, train_pt_path)
    torch.save({'features': feats_test, 'labels': labels_test}, test_pt_path)
    
    # Rimuoviamo il file temporaneo full
    if os.path.exists(temp_full_path):
        os.remove(temp_full_path)
        
    return feats_train, labels_train, feats_test, labels_test

# ---------------------------------------------
# 2. Fine-Tuning Routine
# ---------------------------------------------

def fine_tune(
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = '0',
    epochs: int = 10,  # Ridotto default
    lr: float = 1e-3,
    seed: int = 42,
    num_train_samples = 400,
    fine_tuning_on: str = 'stylegan2',
    backbone: str = 'stylegan1',
    num_anchors: int = 400,
    force_recompute_features: bool = False,
    eval_csv_path: str = None,
    checkpoint_file: str = "checkpoint/checkpoint_HELLO.pth",
    load_checkpoint: bool = False,
    prev_checkpoint_path: str = None, # NUOVO: Path specifico per caricare
    save_feats: bool = False,
    save_feats_prefix: str = None,
    order: list = [], # Lista dei task passati+futuri per la valutazione
    **kwargs
):
    device_obj = torch.device("cuda:"+device if torch.cuda.is_available() else "cpu")
    print(f"--- Processing Task: {fine_tuning_on} ---")

    feature_dir = f"./feature_{backbone}"
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    # Setup determinismo
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # --- A. Setup Backbone ---
    if not os.path.exists(PRETRAINED_MODELS[backbone]):
        print(f"Error: Backbone model {backbone} not found at {PRETRAINED_MODELS[backbone]}")
        return {}, None, None

    backbone_net = load_pretrained_model(PRETRAINED_MODELS[backbone])
    backbone_net.resnet.fc = nn.Identity()
    backbone_net.to(device_obj)
    backbone_net.half() # FP16 per inferenza features
    backbone_net.eval()

    # --- B. Load/Create Train & Test Features for CURRENT Task ---
    real_dir = IMAGE_DIR_DOGAN['real_DoGAN_facades']
    fake_dir = IMAGE_DIR_DOGAN[fine_tuning_on]
    
    # Qui usiamo la nuova funzione che gestisce lo split in assenza di cartelle test
    feats_train, labels_train, _, _ = get_split_features(
        backbone_net, real_dir, fake_dir, feature_dir, fine_tuning_on, 
        device_obj, batch_size, num_workers, seed=seed, force_recompute=force_recompute_features
    )

    # Subsampling Training (se richiesto)
    if num_train_samples is not None and num_train_samples < len(feats_train):
        indices = torch.randperm(len(feats_train))[:num_train_samples]
        feats_train = feats_train[indices]
        labels_train = labels_train[indices]
    
    print(f"Training on {len(feats_train)} samples (Real: {(labels_train==0).sum().item()}, Fake: {(labels_train!=0).sum().item()})")

    # --- C. Prepare Anchors (FROM TRAINING DATA ONLY) ---
    real_mask = labels_train == 0
    real_feats_pool = feats_train[real_mask]
    
    if real_feats_pool.size(0) == 0:
        raise RuntimeError("No real samples in training set to build anchors!")

    rng = torch.Generator().manual_seed(seed)
    if num_anchors > len(real_feats_pool):
        idx = torch.randint(0, len(real_feats_pool), (num_anchors,), generator=rng)
    else:
        idx = torch.randperm(len(real_feats_pool), generator=rng)[:num_anchors]
    anchors = real_feats_pool[idx]
    
    rel_module = RelativeRepresentation(anchors.to(device_obj))

    # --- D. Initialize Classifier ---
    classifier = RelClassifier(rel_module, anchors.size(0), num_classes=2).to(device_obj)

    # --- E. Load Previous Checkpoint (CL Logic) ---
    if load_checkpoint:
        path_to_load = prev_checkpoint_path if prev_checkpoint_path else checkpoint_file
        if os.path.exists(path_to_load):
            print(f"Loading CL weights from: {path_to_load}")
            ckpt = torch.load(path_to_load, map_location=device_obj)
            if 'state_dict' in ckpt:
                classifier.load_state_dict(ckpt['state_dict'])
            else:
                classifier.load_state_dict(ckpt)
        else:
            print(f"No checkpoint found at {path_to_load}. Starting fresh.")

    # --- F. Training Loop ---
    feat_dataset = TensorDataset(feats_train, labels_train)
    sampler = BalancedBatchSampler(labels_train, batch_size=batch_size)
    feat_loader = DataLoader(feat_dataset, batch_sampler=sampler)

    criterion = nn.CrossEntropyLoss().to(device_obj)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() # AMP

    # Check veloce bilanciamento nel primo batch
    first_batch_labels = next(iter(feat_loader))[1]
    print(f"DEBUG: Labels in first batch: Real={ (first_batch_labels==0).sum().item() }, Fake={ (first_batch_labels!=0).sum().item() }")

    print("Starting training...")
    for epoch in range(epochs):
        loss, acc = train_one_epoch(classifier, feat_loader, criterion, optimizer, device_obj, 
                                    save_dir="./logs/train", task_name=fine_tuning_on, scaler=scaler)
        # print(f"  Ep {epoch+1}/{epochs} - Loss: {loss:.4f}, Acc: {acc:.4f}")

    # Save Current Step Model
    torch.save({'state_dict': classifier.state_dict()}, checkpoint_file)
    print(f"Model saved to {checkpoint_file}")

    # --- G. Evaluation on ALL Tasks (Past, Current, Future) ---
    test_results = {}
    
    # Se order_list è vuoto, valutiamo almeno sul task corrente
    order = re.sub(r'[\[\]\s]', '', order).split(',')
    eval_tasks = order if len(order) > 0 else [fine_tuning_on]
    print(f"Evaluating on {len(eval_tasks)} tasks: {eval_tasks}")

    for task_name in eval_tasks:
        # Load TEST features for this task (using same split logic)
        r_dir = IMAGE_DIR_DOGAN['real_DoGAN_facades']
        f_dir = IMAGE_DIR_DOGAN[task_name]
        
        # Qui ci interessano solo i dati di TEST
        _, _, feats_test_task, labels_test_task = get_split_features(
            backbone_net, r_dir, f_dir, feature_dir, task_name, 
            device_obj, batch_size, num_workers, seed=seed, force_recompute=False
        )
        
        test_ds = TensorDataset(feats_test_task, labels_test_task)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        loss_val, acc_val, _, _ = evaluate4(
            classifier, test_loader, criterion, device_obj, 
            test_name=task_name, save_dir="./logs", task_name=fine_tuning_on, fake_type=task_name
        )
        test_results[task_name] = {"loss": loss_val, "acc": acc_val}

    # --- H. CSV Logging ---
    if eval_csv_path is None:
        eval_csv_path = os.path.join('logs_dogan', 'cl_results.csv')
    os.makedirs(os.path.dirname(eval_csv_path), exist_ok=True)

    # Scriviamo una riga: Training_Task, Acc_Task1, Acc_Task2, ...
    cols = ["Train_Step"] + eval_tasks
    
    # Header check
    if not os.path.exists(eval_csv_path) or os.path.getsize(eval_csv_path) == 0:
        with open(eval_csv_path, 'w') as f:
            f.write(','.join(cols) + '\n')

    row = [fine_tuning_on]
    for t in eval_tasks:
        val = test_results.get(t, {}).get('acc', 0.0)
        row.append(f"{val:.4f}")
    
    with open(eval_csv_path, 'a') as f:
        f.write(','.join(row) + '\n')

    # Return anchors logits just in case
    classifier.eval()
    anchors_logits = classifier(anchors.float()).detach().cpu().numpy()
    
    return test_results, anchors, anchors_logits


# ---------------------------------------------
# 3. Orchestrator
# ---------------------------------------------

def run_continual_learning(args):
    # Pulisce stringa ordine
    order_list = re.sub(r'[\[\]\s]', '', args.order).split(',')
    print(f"\n=== Starting Continual Learning Pipeline ===")
    print(f"Order: {order_list}")
    
    prev_ckpt = None
    
    for i, task in enumerate(order_list):
        print(f"\n>>> STEP {i+1}: Task {task}")
        
        # Definiamo nome file specifico per questo step
        current_step_ckpt = os.path.join("checkpoint", f"cl_step_{i}_{task}.pth")
        
        # Aggiorniamo args per questo step
        args.fine_tuning_on = task
        args.checkpoint_file = current_step_ckpt
        args.load_checkpoint = (i > 0) # Carica solo se non è il primo step
        args.order_list = order_list # Passiamo la lista completa per l'eval
        
        # Chiamata training
        results, _, _ = fine_tune(
            **vars(args),
            prev_checkpoint_path=prev_ckpt
        )
        
        # Il checkpoint corrente diventa il precedente per il prossimo step
        prev_ckpt = current_step_ckpt
        
        print(f">>> Finished Step {i+1}. Results saved.")

# ---------------------------------------------
# Main
# ---------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10) # Aumentato per training reale
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_samples', type=int, default=400) # Aumentato
    # Nota: fine_tuning_on viene sovrascritto dal loop CL
    parser.add_argument('--fine_tuning_on', type=str, default='cyclegan_facades') 
    parser.add_argument('--backbone', type=str, default='cyclegan_facades')
    parser.add_argument('--num_anchors', type=int, default=300)
    parser.add_argument('--force_recompute_features', action='store_true')
    parser.add_argument('--eval_csv_path', type=str, default='logs_dogan/cl_matrix.csv')
    parser.add_argument('--order', default='[cyclegan_facades,progan_celeb256,progan_1024_celebhq,glow_smiling,star_gan]')

    # Argomenti "dummy" per compatibilità, gestiti dall'orchestratore
    parser.add_argument('--load_checkpoint', action='store_true')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint/dummy.pth')
    parser.add_argument('--save_feats', action='store_true')
    parser.add_argument('--save_feats_prefix', type=str, default='')

    args = parser.parse_args()

    # Avvio Orchestratore
    run_continual_learning(args)