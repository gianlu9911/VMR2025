import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model
def train_one_epoch(model, dataloader, criterion, optimizer, device, print_freq=10):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_samples = 0

    with tqdm(total=len(dataloader), unit=' batch', desc='Training: ') as bar:
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute batch accuracy
            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).float().mean().item()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            num_samples += batch_size

            if i % print_freq == 0:
                bar.set_postfix({
                    'loss': round(running_loss / num_samples, 4),
                    'acc': round(running_acc / num_samples, 4)
                })
            bar.update(1)

    return running_loss / num_samples, running_acc / num_samples


import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    num_samples = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation: '):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).float().mean().item()

            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size
            val_acc += acc * batch_size
            num_samples += batch_size

            # Store labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Build save path
    save_dir = os.path.join(
        args.log_folder, "fine_tune", args.pretrained_model, args.finetune_on
    )
    os.makedirs(save_dir, exist_ok=True)

    # Save confusion matrix as heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()

    return val_loss / num_samples, val_acc / num_samples


def fine_tune(args):
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    save_path = os.path.join('checkpoint', f'finetuned_{args.pretrained_model}_on_{args.finetune_on}_last layer.pth')    
    save_path = f'./checkpoint/finetuned_stylegan1_on_stylegan2.pth'

    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Prepare dataset: real + StyleGAN2
    real_dir = IMAGE_DIR['real']
    fake_st2_dir = IMAGE_DIR['stylegan2']
    fake_st1_dir = IMAGE_DIR['stylegan1']
    fake_sd14_dir = IMAGE_DIR['sdv1_4']
    fake_xl_dir = IMAGE_DIR['stylegan_xl']
    fake_st3_dir = IMAGE_DIR['stylegan3']
    fake_sdv21_dir = IMAGE_DIR['sdv2_1']

    dataset = RealSynthethicDataloader(real_dir, IMAGE_DIR[args.finetune_on], num_points=args.num_points)
    test_st1 = RealSynthethicDataloader(real_dir, fake_st1_dir, split='test_set', num_points=args.num_points)
    test_sd14 = RealSynthethicDataloader(real_dir, fake_sd14_dir, split='test_set', num_points=args.num_points)
    test_xl = RealSynthethicDataloader(real_dir, fake_xl_dir, split='test_set', num_points=args.num_points)
    test_sdv21 = RealSynthethicDataloader(real_dir, fake_sdv21_dir, split='test_set', num_points=args.num_points)
    test_st2 = RealSynthethicDataloader(real_dir, fake_st2_dir, split='test_set', num_points=args.num_points)
    test_st3 = RealSynthethicDataloader(real_dir, fake_st3_dir, split='test_set', num_points=args.num_points)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_st1 = DataLoader(test_st1, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloader_sd14 = DataLoader(test_sd14, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloader_xl = DataLoader(test_xl, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloader_st2 = DataLoader(test_st2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloader_st3 = DataLoader(test_st3, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloader_sdv21 = DataLoader(test_sdv21, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    

    if os.path.exists(save_path):
        model = load_pretrained_model(save_path)
        model.to(device)
        print(f'Loaded fine-tuned model from {save_path}')
        # Loss & optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    else:

        # Load pretrained StyleGAN1
        ckpt_path = PRETRAINED_MODELS[args.pretrained_model]
        model = load_pretrained_model(ckpt_path)
        model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze last layer
        for param in model.resnet.fc.parameters():
            param.requires_grad = True

        # Training loop
        start_time = time.time()
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, dataloader, criterion, optimizer, device)
            print(f'Epoch [{epoch+1}/{args.epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')

            # Optional: evaluate on held-out test set
            # val_loss, val_acc = evaluate(val_loader, model, criterion, args)
            # print(f'Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')

        # Save fine-tuned model
        end_time = time.time()
        print(f'Training completed in {(end_time - start_time)/60:.2f} minutes.')
        torch.save({'state_dict': model.state_dict()}, save_path)
        print(f'Model saved to {save_path}')
        # ----------------------
    # Final evaluation on all test sets
    # ----------------------
    print("\nEvaluating on all test sets:")

    test_sets = {
        "real vs stylegan1": dataloader_st1,
        "real vs stylegan2": dataloader_st2,
        "real vs styleganxl": dataloader_xl,
        "real vs sdv1_4": dataloader_sd14,
        "real vs sdv2_1": dataloader_sdv21,
        "real vs stylegan3": dataloader_st3,  # Assuming stylegan3 uses the same loader as stylegan2 for this example
    }
    results = {}

    for name, loader in test_sets.items():
        test_loss, test_acc = evaluate(model, loader, criterion, device)
        #extract the task name from the name
        task = name.split(" ")[2]
        print(f"[{task}] - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        print(f"[{name}] - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        results[task] = test_acc
    return pd.DataFrame([results])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained_model', type=str, default='stylegan1', choices=['stylegan1', 'stylegan2', 'sdv1.4'], help='Pretrained model to load')
    parser.add_argument('--finetune_on', type=str, default='stylegan2', choices=['stylegan2', 'sdv1.4'], help='Dataset to fine-tune on')
    parser.add_argument('--log_folder', type=str, default='./logs_finetune', help='Directory to save logs')
    parser.add_argument('--num_points', type=int, default=None, help='Number of real/fake images to use for training/testing - None uses all available data')
    args = parser.parse_args()
    orders = [
        "stylegan1, stylegan2, sdv1_4, stylegan3, stylegan_xl, sdv2_1",
        "stylegan1, stylegan2, stylegan3, stylegan_xl, sdv1_4, sdv2_1",
        "sdv1_4, sdv2_1, stylegan1, stylegan2, stylegan3, stylegan_xl",
        #random order from stylegan2
        "stylegan2, stylegan3,  sdv2_1,stylegan1,stylegan_xl, sdv1_4"
    ]
    import os

    for o in orders:
        print(f"Training order: {o}")
        args.tasks = o
        order_str = o.replace(" ", "").replace(",", "_")

        for task in o.split(","):
            task = task.strip()
            print(f"Fine-tuning on task: {task}")
            args.finetune_on = task
            results = fine_tune(args)

            path = f"logs_finetune/new_sequential_finetune_results_{order_str}.csv"

            # scrive header solo se il file NON esiste
            write_header = not os.path.exists(path)

            results.to_csv(
                path,
                mode="a",          # append
                header=write_header,
                index=False
            )
