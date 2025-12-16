#!/usr/bin/env python3
import os
import time
import warnings


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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

from torch.utils.data import TensorDataset, DataLoader

def get_exemplar_loader(exemplar_set, batch_size=64, shuffle=True):
    if exemplar_set is None:
        return None
    imgs, labels = exemplar_set
    dataset = TensorDataset(imgs, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def train_one_epoch_ucir(model, teacher, dataloader, criterion, optimizer, device,
                         lambda_new=1.0, lambda_dist=1.0, lambda_exemplar=1.0,
                         save_dir=None, task_name="task", exemplar_loader=None):
    model.train()
    running_loss, running_acc, num_samples = 0.0, 0.0, 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # forward per batch corrente
        outputs = model(imgs)
        loss_new = criterion(outputs, labels)

        # loss sugli exemplars in batch
        loss_exemplar = 0.0
        if exemplar_loader is not None:
            for ex_imgs, ex_labels in exemplar_loader:
                ex_imgs, ex_labels = ex_imgs.to(device), ex_labels.to(device)
                ex_outputs = model(ex_imgs)
                loss_exemplar += criterion(ex_outputs, ex_labels)
            loss_exemplar /= len(exemplar_loader)  # media

        # distillation loss
        loss_dist = 0.0
        if teacher is not None:
            T = 2.0
            with torch.no_grad():
                logits_teacher = teacher(imgs)
                soft_targets = torch.softmax(logits_teacher / T, dim=1)
            loss_dist = nn.KLDivLoss(reduction='batchmean')(
                torch.log_softmax(outputs / T, dim=1),
                soft_targets
            ) * (T*T)

        # combinazione
        loss_total = lambda_new * loss_new + lambda_dist * loss_dist + lambda_exemplar * loss_exemplar
        loss_total.backward()
        optimizer.step()

        # accuracy batch
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean().item()
        batch_size = labels.size(0)
        running_loss += loss_total.item() * batch_size
        running_acc += acc * batch_size
        num_samples += batch_size

    return running_loss / num_samples, running_acc / num_samples


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

    checkpoint_dir = "./checkpoint_ucir"
    logs_dir = "./logs_ucir"
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
    dataset = RealSynthethicDataloader(real_dir, fake_dir, num_training_samples=num_train_samples)
    from torch.utils.data import ConcatDataset, TensorDataset

    # crea il dataset degli exemplars
    if exemplar_set is not None:
        imgs_ex, labels_ex = exemplar_set
        exemplar_dataset = TensorDataset(imgs_ex, labels_ex)
        full_dataset = ConcatDataset([dataset, exemplar_dataset])
    else:
        full_dataset = dataset

    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        exemplar_loader = get_exemplar_loader(exemplar_set, batch_size=batch_size)
        train_loss, train_acc = train_one_epoch_ucir(
            model=classifier,
            teacher=teacher,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            lambda_new=lambda_new,
            lambda_dist=lambda_dist,
            lambda_exemplar=lambda_exemplar,
            exemplar_loader=exemplar_loader,
            save_dir="./logs_ucir/train",
            task_name=fine_tuning_on
        )
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
        eval_csv_path = os.path.join(logs_dir, "eval_results_ucir.csv")
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
    parser.add_argument('--batch_size', type=int, default=64)
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
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint/checkpoint_HELLO_ucir.pth',
                        help="Path to load/save classifier weights (default: checkpoint/checkpoint_HELLO.pth)")
    parser.add_argument('--order', default='[stylegan1, stylegan2, sdv1_4, stylegan3, stylegan_xl, sdv2_1]',
                        help="Order of model steps for saving features.")
    parser.add_argument('--lambda_new', type=float, default=1.0,
                    help="Peso della loss sul task corrente")
    parser.add_argument('--lambda_dist', type=float, default=1.0,
                    help="Peso della distillation loss per le classi vecchie")
    parser.add_argument('--lambda_exemplar', type=float, default=0,
                    help="Peso opzionale per la loss sugli esemplari (se usati)")
    parser.add_argument('--number_exemplars', type=int, default=10,
                    help="Numero di esemplari da mantenere (se usati)")
    args = parser.parse_args()

    exemplar = None
    order_list = [x.strip() for x in args.order.strip('[]').split(',')]
    for task in order_list:
        results, exemplar = fine_tune(**vars(args), fine_tuning_on=task, exemplar_set=exemplar)  
        print("All test results:")
        for k, v in results.items():
            print(f" - {k}: loss={v['loss']:.4f}, acc={v['acc']:.4f}, feat_time={v['feat_time']:.2f}s")