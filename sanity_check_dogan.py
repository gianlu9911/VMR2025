#!/usr/bin/env python3
import os
import time
import warnings
import argparse
import re

warnings.filterwarnings("ignore")
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model
from src.utils import  BalancedBatchSampler, RelativeRepresentation, RelClassifier, extract_and_save_features, evaluate
from src.g_utils import train_one_epoch2 as train_one_epoch
from src.g_utils import evaluate3, save_features_only

# ---------------------------------------------
# Fine-tuning pipeline (main)
# ---------------------------------------------

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split # <--- Per lo split professionale

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
    force_recompute_features: bool = False,
    eval_csv_path: str = None,
    checkpoint_file: str = "checkpoint/checkpoint_HELLO.pth",
    load_checkpoint: bool = False,
    **kwargs
):
    device_obj = torch.device("cuda:"+device if torch.cuda.is_available() else "cpu")
    print(f"--- Training on {fine_tuning_on} with AMP & Sklearn Split ---")

    # Setup determinismo
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Backbone: {backbone}")  

    # 1. Backbone & Feature Extraction
    backbone_net = load_pretrained_model(PRETRAINED_MODELS[backbone])
    backbone_net.resnet.fc = nn.Identity()
    backbone_net.to(device_obj)
    backbone_net.eval()
    
    # Usiamo FP16 per estrarre le feature più velocemente
    backbone_net.half() 

    # Caricamento dataset completo
    real_dir = IMAGE_DIR['real']
    real_dir = os.path.join(real_dir, 'train_set')
    fake_dir = IMAGE_DIR[fine_tuning_on] # Assicurati che punti alla cartella corretta
    if fine_tuning_on == 'stylegan1':
        fake_dir = os.path.join(fake_dir, 'train_set')
    dataset = RealSynthethicDataloader(real_dir, fake_dir, split='')
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    feature_file = os.path.join(f"./feature_{backbone}", f"full_feat_{fine_tuning_on}.pt")
    
    if force_recompute_features or not os.path.exists(feature_file):
        # Estrazione con autocast per velocità
        with torch.cuda.amp.autocast():
            feats_all, labels_all, _ = extract_and_save_features(backbone_net, loader, feature_file, device_obj)
    else:
        data = torch.load(feature_file)
        feats_all, labels_all = data["features"], data["labels"]

    # 2. Split Professionale con Scikit-Learn
    # Dividiamo le feature estratte in Train (80%) e Test (20%)
    f_train, f_test, l_train, l_test = train_test_split(
        feats_all.cpu().numpy(), 
        labels_all.cpu().numpy(), 
        test_size=0.2, 
        random_state=seed, 
        stratify=labels_all.cpu().numpy() # Mantiene le proporzioni Real/Fake
    )
    
    f_train, l_train = torch.from_numpy(f_train), torch.from_numpy(l_train)
    f_test, l_test = torch.from_numpy(f_test), torch.from_numpy(l_test)

    # 3. Setup Classificatore Relativo
    # Usiamo i real del training set come anchor
    real_train_mask = l_train == 0
    anchors = f_train[real_train_mask][:num_anchors]
    
    rel_module = RelativeRepresentation(anchors.to(device_obj).half()) # Anchors in FP16 per coerenza con il backbone
    classifier = RelClassifier(rel_module, anchors.size(0), num_classes=2).to(device_obj)
    
    if load_checkpoint and os.path.exists(checkpoint_file):
        classifier.load_state_dict(torch.load(checkpoint_file)['state_dict'])

    # 4. Training Loop con AMP
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device_obj)
    scaler = torch.cuda.amp.GradScaler() # GradScaler per AMP

    train_ds = TensorDataset(f_train, l_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    classifier.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_f, batch_l in train_loader:
            batch_f, batch_l = batch_f.to(device_obj), batch_l.to(device_obj)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(): # <--- Attivazione AMP
                outputs = classifier(batch_f)
                loss = criterion(outputs, batch_l)
            
            scaler.scale(loss).backward() # Scale loss
            scaler.step(optimizer)        # Optimizer step
            scaler.update()               # Update scaler
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")

    # 5. Valutazione sul Test Split (Dati mai visti)
    classifier.eval()
    all_results = {}

    task_to_eval = order_list
    print(f"\n=== Evaluation on Test Splits of: {task_to_eval} ===")
    for task in task_to_eval:
        print(f"\n>>> STEP {i+1}/{len(order_list)}: Evaluating on {task}")
        t_feat_file = os.path.join(f"./feature_{backbone}", f"full_feat_{task}.pt")
        if not os.path.exists(t_feat_file):
            print(f"Feature file {t_feat_file} not found. Skipping evaluation.")
            continue
        
        # Carico le feature
        data = torch.load(t_feat_file)
        feats_all, labels_all = data["features"], data["labels"]

        _, test_task, _, l_task = train_test_split(
            feats_all.cpu().numpy(), 
            labels_all.cpu().numpy(), 
            test_size=0.2, 
            random_state=seed, 
            stratify=labels_all.cpu().numpy() # Mantiene le proporzioni Real/Fake
        )
        t_ds = TensorDataset(torch.from_numpy(test_task), torch.from_numpy(l_task))
        t_loader = DataLoader(t_ds, batch_size=batch_size)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                _, test_acc, _, _ = evaluate3(classifier, t_loader, criterion, device_obj)
        all_results[task] = test_acc
        print(f"Test Accuracy on {task}: {test_acc:.4f}")
        

        
        # Carico weights
    if load_checkpoint and os.path.exists(checkpoint_file):
            classifier.load_state_dict(torch.load(checkpoint_file)['state_dict'])
    
    # Salvataggio
    torch.save({'state_dict': classifier.state_dict()}, checkpoint_file)
    
    return all_results, anchors, None


# ---------------------------------------------
# Main CLI (kept for convenience)
# ---------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help="Number of training samples to use. If None, use all available samples.")
    parser.add_argument('--fine_tuning_on', type=str, default='stylegan2',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4', 'stylegan3','sdv2_1'],
                        help="Which synthetic dataset to use for fine-tuning")
    parser.add_argument('--backbone', type=str, default='stylegan1',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4', 'stylegan3','sdv2_1'],
                        help="Which backbone feature extractor to use")
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

    # NEW CLI args for checkpoint load/save
    parser.add_argument('--load_checkpoint', action='store_true',
                        help="If set, attempt to load weights from --checkpoint_file before training (default: False).")
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint/checkpoint_HELLO.pth',
                        help="Path to load/save classifier weights (default: checkpoint/checkpoint_HELLO.pth)")
    parser.add_argument('--save_feats', action='store_true',
                        help="If set, save the FULL feature arrays (real/fake/anchors) for each test set during evaluation.")
    parser.add_argument('--save_feats_prefix', type=str, default='saved_numpy_features/step_prova',
                        help="Optional prefix (path+name) to use for saved feature .npy files. "
                             "If not provided, defaults to <feature_dir>/<test_name>_eval_feats")
    parser.add_argument('--order', default='[stylegan1,  progan_celeb256, progan_1024_celebhq]',
                        help="Order of model steps for saving features.")

    args = parser.parse_args()

    order_list = re.sub(r'[\[\]\s]', '', args.order).split(',')
    
    print(f"\n=== Starting Continual Learning on: {order_list} ===")
    
    prev_ckpt = None
    
    # 2. Ciclo su ogni task della lista order
    for i, task in enumerate(order_list):
        print(f"\n>>> STEP {i+1}/{len(order_list)}: Training on {task}")
        
        # Definiamo un checkpoint specifico per questo step
        current_ckpt = os.path.join("checkpoint", f"cl_step_{i}_{task}.pth")
        
        # Aggiorniamo i parametri per questa iterazione
        args.fine_tuning_on = task
        args.checkpoint_file = current_ckpt
        args.load_checkpoint = (i > 0) # Carica il modello precedente solo dal secondo task in poi
        
        # Se i > 0, dobbiamo passare il path del checkpoint del task precedente
        # Modifichiamo la chiamata per usare il prev_ckpt
        results, anchors, _ = fine_tune(**vars(args))
        
        # Salviamo il path per il prossimo step
        prev_ckpt = current_ckpt
        
        print(f">>> Finished {task}. Accuracy: {results.get('test_acc', 'N/A')}")
