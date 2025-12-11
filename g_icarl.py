#!/usr/bin/env python3
import os
import time
import warnings


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# === Project imports - adjust these to your repo layout ===
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model

from src.g_utils import evaluate3

def select_exemplars(model, dataset, m, device):
    """
    model: backbone addestrato sul task corrente
    dataset: dataloader del task corrente
    m: numero di esemplari da mantenere
    """
    model.eval()
    exemplars_imgs = []
    exemplars_labels = []
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = model(imgs)  # estrai feature
            # qui puoi usare ad esempio il metodo iCaRL originale: closest-to-mean selection
            # per semplicità, salviamo un subset casuale
            for img, label in zip(imgs.cpu(), labels.cpu()):
                if len(exemplars_imgs) < m:
                    exemplars_imgs.append(img)
                    exemplars_labels.append(label)
    
    imgs_tensor = torch.stack(exemplars_imgs)
    labels_tensor = torch.tensor(exemplars_labels)
    return imgs_tensor, labels_tensor


def train_one_epoch_icarl(model, teacher, dataloader, criterion, optimizer, device,
                          lambda_new=1.0, lambda_dist=1.0, lambda_exemplar=1.0, save_dir=None, task_name="task", exemplar_set=None):
    """
    model: backbone corrente da aggiornare
    teacher: vecchio backbone congelato (None se primo task)
    dataloader: batch di immagini + label del task corrente
    criterion: CrossEntropyLoss o simile sul nuovo task
    lambda_dist: peso della distillation loss
    """
    model.train()
    running_loss, running_acc, num_samples = 0.0, 0.0, 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # ---- forward pass ----
        outputs = model(imgs)  # feature / logits per nuovo task

        # ---- loss sul nuovo task ----
        loss_new = criterion(outputs, labels)

        # ---- distillation loss ----
        if teacher is not None:
            with torch.no_grad():
                feats_teacher = teacher(imgs)
            feats_model = outputs  # se model restituisce feature
            loss_dist = nn.MSELoss()(feats_model, feats_teacher)
        else:
            loss_dist = 0.0
        if exemplar_set is not None:
            outputs_exemplar = model(exemplar_set[0].to(device))
            loss_exemplar = criterion(outputs_exemplar, exemplar_set[1].to(device))
        else:
            loss_exemplar = 0.0


        # ---- combinazione loss ----
        loss_total = lambda_new * loss_new + loss_dist * lambda_dist + lambda_exemplar * loss_exemplar

        loss_total.backward()
        optimizer.step()

        # ---- accuracy sul batch ----
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean().item()

        batch_size = labels.size(0)
        running_loss += loss_total.item() * batch_size
        running_acc += acc * batch_size
        num_samples += batch_size

    epoch_loss = running_loss / num_samples
    epoch_acc = running_acc / num_samples

    return epoch_loss, epoch_acc

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
    checkpoint_file: str = "checkpoint/checkpoint_HELLO_icarl.pth",
    order: str = '[stylegan1, stylegan2, sdv1_4, stylegan3, stylegan_xl, sdv2_1]',
    lambda_new: float = 1.0,
    lambda_dist: float = 1.0,
    lambda_exemplar: float = 1.0,
    number_exemplars: int = 5000,
    exemplar_set: str = None,
):

    """Fine-tune relative-representation classifier.

    load_checkpoint: if True, attempt to load checkpoint_file into the classifier before training.
    checkpoint_file: path to load/save classifier weights (default: checkpoint/checkpoint_HELLO.pth)

    Returns:
        dict: test_results mapping dataset name -> {loss, acc, feat_time}
    """
    device = torch.device("cuda:"+device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = "./checkpoint_icarl"
    logs_dir = "./logs_icarl"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Backbone
    backbone_net = load_pretrained_model(PRETRAINED_MODELS[backbone])

    classifier = backbone_net.to(device)
    order_list = [x.strip() for x in order.strip('[]').split(',')]
    print(f"Using order: {order_list}") 
    task_idx = order_list.index(fine_tuning_on)

    if task_idx > 0:
        teacher = load_pretrained_model(checkpoint_file).to(device)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        print(f"Using teacher model: {checkpoint_file}")
    else:
        teacher = None

    # Dataset directories
    real_dir = IMAGE_DIR['real']
    fake_dir = IMAGE_DIR[fine_tuning_on]
    exemplar_set = None
    dataset = RealSynthethicDataloader(real_dir, fake_dir, num_training_samples=num_train_samples)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_icarl(classifier, teacher, train_loader, criterion, optimizer, device, save_dir="./logs_icarl/train", task_name=fine_tuning_on, lambda_new=lambda_new, lambda_dist=lambda_dist, lambda_exemplar=lambda_exemplar, exemplar_set=exemplar_set)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.2f} minutes")

    new_exemplar_set = select_exemplars(classifier, dataset, number_exemplars, device)
    imgs_new, labels_new = new_exemplar_set
    if exemplar_set is None:
        exemplar_set = (imgs_new, labels_new)
    else:
        imgs_old, labels_old = exemplar_set
        imgs_total = torch.cat([imgs_old, imgs_new])
        labels_total = torch.cat([labels_old, labels_new])
        
        # Riduci se superiamo number_exemplars
        if imgs_total.size(0) > number_exemplars:
            idx = torch.randperm(imgs_total.size(0))[:number_exemplars]
            imgs_total = imgs_total[idx]
            labels_total = labels_total[idx]

        exemplar_set = (imgs_total, labels_total)
    # Save checkpoint to the specified checkpoint_file (USER REQUEST)
    checkpoint_path = checkpoint_file
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({'state_dict': classifier.state_dict()}, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Prepare test datasets
    dataloaders_test = {
        "stylegan1": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan1'], split='test_set'),
        "stylegan2": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan2'], split='test_set'),
        "sdv1_4": RealSynthethicDataloader(real_dir, IMAGE_DIR['sdv1_4'], split='test_set'),
        "stylegan3": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan3'], split='test_set'),
        "stylegan_xl": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan_xl'], split='test_set'),
        "sdv2_1": RealSynthethicDataloader(real_dir, IMAGE_DIR['sdv2_1'], split='test_set'),  # Uncomment if sdv2_1 is available
    }





    test_results = {}
    for name, dataset in dataloaders_test.items():

        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers)

        fake_type = name
        print(f"Evaluating on {name} with fake type {fake_type}...")
        loss, acc, preds, labels = evaluate3(classifier, test_loader, criterion, device,
                              test_name=name, save_dir=logs_dir, task_name=fine_tuning_on,
                              fake_type=fake_type)
        test_results[name] = {"loss": loss, "acc": acc, "feat_time": 0, "preds": preds, "labels": labels}

    # --- Append evaluation results to CSV ---
    # CSV columns/order requested by user:
    csv_columns = []
    csv_columns.append("fine_tuning_on")
    for o in order:
        csv_columns.append(o)

    # Default path if not provided
    if eval_csv_path is None:
        eval_csv_path = os.path.join(logs_dir, "eval_results_icarl.csv")
    os.makedirs(os.path.dirname(eval_csv_path), exist_ok=True)

    with open(eval_csv_path, 'a') as f:
        if os.path.getsize(eval_csv_path) == 0:
            # write header if file is empty
            f.write(','.join(csv_columns) + '\n')

        row = [fine_tuning_on]
        for col in csv_columns[1:]:
            if col in test_results:
                row.append(f"{test_results[col]['acc']:.5f}")
            else:
                row.append('')
        f.write(','.join(row) + '\n')
    return test_results, exemplar_set
# ---------------------------------------------
# Main CLI (kept for convenience)
# ---------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_samples', type=int, default=100,
                        help="Number of training samples to use. If None, use all available samples.")
    parser.add_argument('--backbone', type=str, default='stylegan1',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4', 'stylegan3','sdv2_1'],
                        help="Which backbone feature extractor to use")    
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint/checkpoint_HELLO_icarl.pth',
                        help="Path to load/save classifier weights (default: checkpoint/checkpoint_HELLO.pth)")
    parser.add_argument('--order', default='[stylegan1, stylegan2, sdv1_4, stylegan3, stylegan_xl, sdv2_1]',
                        help="Order of model steps for saving features.")
    parser.add_argument('--lambda_new', type=float, default=1.0,
                    help="Peso della loss sul task corrente")
    parser.add_argument('--lambda_dist', type=float, default=1.0,
                    help="Peso della distillation loss per le classi vecchie")
    parser.add_argument('--lambda_exemplar', type=float, default=1.0,
                    help="Peso opzionale per la loss sugli esemplari (se usati)")
    parser.add_argument('--number_exemplars', type=int, default=5,
                    help="Numero di esemplari da mantenere (se usati)")
    args = parser.parse_args()

    exemplar = None
    order_list = [x.strip() for x in args.order.strip('[]').split(',')]
    for task in order_list:
        results, exemplar = fine_tune(**vars(args), fine_tuning_on=task, exemplar_set=exemplar)  
        print("All test results:")
        for k, v in results.items():
            print(f" - {k}: loss={v['loss']:.4f}, acc={v['acc']:.4f}, feat_time={v['feat_time']:.2f}s")