#!/usr/bin/env python3
import os
import time
import warnings
import shutil
import re

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
from src.utils import  BalancedBatchSampler, RelativeRepresentation, RelClassifier, extract_and_save_features, evaluate, train_one_epoch
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
    # Aggiungiamo un run_id per tracciare i log se vuoi
    run_id: str = "Run_X"
):

    """Fine-tune relative-representation classifier.

    load_checkpoint: if True, attempt to load checkpoint_file into the classifier before training.
    checkpoint_file: path to load/save classifier weights (default: checkpoint/checkpoint_HELLO.pth)

    Returns:
        dict: test_results mapping dataset name -> {loss, acc, feat_time}
    """
    device = torch.device("cuda:"+device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    feature_dir = f"./feature_{backbone}"
    checkpoint_dir = "./checkpoint"
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_file_dir = os.path.dirname(checkpoint_file) if os.path.dirname(checkpoint_file) != '' else checkpoint_dir
    os.makedirs(checkpoint_file_dir, exist_ok=True)

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

    # Anchors
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
                elif isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        classifier.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        try:
                            classifier.load_state_dict(checkpoint)
                        except Exception as e:
                            print(f"Warning: couldn't interpret checkpoint dict format: {e}. Continuing without loading.")
                else:
                    try:
                        classifier.load_state_dict(checkpoint)
                    except Exception as e:
                        print(f"Warning: failed to load checkpoint: {e}. Continuing without loading.")
                print("Checkpoint loaded into classifier.")
            except Exception as e:
                print(f"Failed to load checkpoint from {checkpoint_file}: {e}. Continuing without loading.")
        else:
            print(f"Checkpoint file {checkpoint_file} not found. Continuing without loading.")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(classifier, feat_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.2f} minutes")

    # Save checkpoint
    checkpoint_path = checkpoint_file
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({'state_dict': classifier.state_dict()}, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Prepare test datasets (NON TOCCATO)
    dataloaders_test = {
        "real_vs_stylegan1": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan1'], split='test_set'),
        "real_vs_stylegan2": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan2'], split='test_set'),
        "real_vs_sdv14": RealSynthethicDataloader(real_dir, IMAGE_DIR['sdv1_4'], split='test_set'),
        "real_vs_stylegan3": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan3'], split='test_set'),
        "real_vs_styleganxl": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan_xl'], split='test_set'),
        "real_vs_sdv21": RealSynthethicDataloader(real_dir, IMAGE_DIR['sdv2_1'], split='test_set'), 
    }

    test_results = {}
    for name, dataset in dataloaders_test.items():
        feat_file_test = os.path.join(feature_dir, f"test_{name}_features.pt")
        print(f"Preparing test features for {name} -> {feat_file_test}")

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers)
        feats_test, labels_test, feat_time = extract_and_save_features(backbone_net, loader,
                                                                  feat_file_test, device, split='test_set')

        anchors_cpu = rel_module(anchors.to(device)).cpu()
        real_mask_eval = (labels_test == 0)
        fake_mask_eval = (labels_test == 1)
        real_feats_eval = feats_test[real_mask_eval]
        real_feats_eval = rel_module(real_feats_eval.to(device)).cpu()
        fake_feats_eval = feats_test[fake_mask_eval]
        fake_feats_eval = rel_module(fake_feats_eval.to(device)).cpu()

        if save_feats:
            try:
                os.makedirs("saved_numpy_features", exist_ok=True)
                import re as _re  
                fake_type = _re.sub(r'^.*real_vs_', '', name).split('*')[0]
                domain = fine_tuning_on.replace("_", "")
                prefix_to_use = os.path.join("saved_numpy_features", f"step_{domain}")

                real_file = f"{prefix_to_use}_real.npy"
                fake_file = f"{prefix_to_use}_fake_{fake_type}.npy"
                anchors_file = f"{prefix_to_use}_anchors.npy"

                real_np = real_feats_eval.cpu().numpy() if isinstance(real_feats_eval, torch.Tensor) else np.array(real_feats_eval)
                fake_np = fake_feats_eval.cpu().numpy() if isinstance(fake_feats_eval, torch.Tensor) else np.array(fake_feats_eval)
                anchors_np = anchors_cpu.cpu().numpy() if isinstance(anchors_cpu, torch.Tensor) else np.array(anchors_cpu)

                if real_np.size > 0: np.save(real_file, real_np)
                if fake_np.size > 0: np.save(fake_file, fake_np)
                if anchors_np.size > 0: np.save(anchors_file, anchors_np)

            except Exception as e:
                print(f"[save_feats] ERROR while saving features for {name}: {e}")

        os.makedirs("./logs", exist_ok=True)
        plot_save_path = os.path.join("./logs", f"feature_plot_{name}_{plot_method}.png")
        default_prefix = os.path.join(feature_dir, f"{name}_eval_feats")

        test_dataset = TensorDataset(feats_test, labels_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        loss, acc = evaluate(classifier, test_loader, criterion, device,
                             rel_module=rel_module, test_name=name, save_dir="./logs")
        test_results[name] = {"loss": loss, "acc": acc, "feat_time": feat_time}

    # --- Append evaluation results to CSV ---
    # (NON TOCCATO)
    csv_columns = [
        'fine_tuning_on',
        'real_vs_stylegan1',
        'real_vs_stylegan2',
        'real_vs_sdv14',
        'real_vs_stylegan3',
        'real_vs_styleganxl',
        'real_vs_sdv21' 
    ]

    if eval_csv_path is None:
        eval_csv_path = os.path.join('logs', 'test_accuracies.csv')
    os.makedirs(os.path.dirname(eval_csv_path), exist_ok=True)

    with open(eval_csv_path, 'a') as f:
        if os.path.getsize(eval_csv_path) == 0:
            # Aggiungo la colonna Run_ID all'header per distinguere le sequenze
            f.write('Run_ID,' + ','.join(csv_columns) + '\n')

        # Aggiungo il run_id alla riga
        row = [run_id, fine_tuning_on]
        for col in csv_columns[1:]:
            if col in test_results:
                row.append(f"{test_results[col]['acc']:.5f}")
            else:
                row.append('')
        f.write(','.join(row) + '\n')
    return test_results, anchors


# ---------------------------------------------
# Main CLI
# ---------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=5)
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

    args = parser.parse_args()

    # LE TUE 4 SEQUENZE DI TRAINING
    orders = [
        '[stylegan1, stylegan2, sdv1_4, stylegan3, stylegan_xl, sdv2_1]',
        '[stylegan1, stylegan2, stylegan3, stylegan_xl, sdv1_4, sdv2_1]',
        '[sdv1_4, sdv2_1, stylegan1, stylegan2, stylegan3, stylegan_xl]',
        '[stylegan2, sdv1_4, stylegan_xl, stylegan3, sdv2_1, stylegan1]',
    ]

    # LOOP SULLE SEQUENZE (Le 4 Run)
    for run_idx, order_str in enumerate(orders):
        run_name = f"Run_{run_idx + 1}"
        print(f"\n\n{'='*60}")
        print(f"=== INIZIO {run_name} | Sequenza: {order_str} ===")
        print(f"{'='*60}")

        # Parsing della stringa per estrarre la lista di task
        task_list = re.sub(r'[\[\]\s]', '', order_str).split(',')

        # Variabile per tenere in memoria il path del checkpoint del task precedente
        prev_ckpt = ""

        # LOOP SUI TASK DELLA SEQUENZA CORRENTE (Continual Learning)
        for step_idx, current_task in enumerate(task_list):
            print(f"\n>>> [{run_name}] Step {step_idx}: Training su '{current_task}'")

            # Definiamo il nome univoco del checkpoint per questo specifico step
            current_ckpt = os.path.join("checkpoint", f"cl_{run_name}_step_{step_idx}_{current_task}.pth")

            # Prepariamo gli argomenti per fine_tune partendo da quelli del parser
            func_args = vars(args).copy()
            func_args['fine_tuning_on'] = current_task
            func_args['checkpoint_file'] = current_ckpt
            func_args['run_id'] = run_name

            if step_idx == 0:
                # Primo task della sequenza: NON caricare nulla, parti da zero
                func_args['load_checkpoint'] = False
            else:
                # Dal secondo task in poi: CARICA il checkpoint del task precedente
                func_args['load_checkpoint'] = True
                
                # Per poter caricare il file precedente ma salvare su quello nuovo,
                # copiamo fisicamente il file .pth vecchio rinominandolo col nome nuovo.
                # La funzione fine_tune lo troverà, lo caricherà, e a fine epoca lo sovrascriverà.
                if os.path.exists(prev_ckpt):
                    shutil.copy(prev_ckpt, current_ckpt)
                    print(f"[*] Portati avanti i pesi da {prev_ckpt}")

            # Esecuzione del fine tuning
            results, anchors = fine_tune(**func_args)

            # Aggiorniamo prev_ckpt per lo step successivo
            prev_ckpt = current_ckpt

            # Stampa compatta dei risultati a fine task
            print(f"\n--- Risultati Test per [{run_name}] dopo training su {current_task} ---")
            for k, v in results.items():
                print(f" {k:20s}: Acc={v['acc']:.4f}")