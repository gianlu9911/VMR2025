import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    num_samples = 0

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

    return val_loss / num_samples, val_acc / num_samples

def fine_tune(args):
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    save_path = f'./checkpoint/finetuned_stylegan1_on_stylegan2_last layer.pth'

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

    dataset = RealSynthethicDataloader(real_dir, fake_st2_dir)
    test_st1 = RealSynthethicDataloader(real_dir, fake_st1_dir, split='test_set')
    test_sd14 = RealSynthethicDataloader(real_dir, fake_sd14_dir, split='test_set')
    test_xl = RealSynthethicDataloader(real_dir, fake_xl_dir, split='test_set')
    test_st2 = RealSynthethicDataloader(real_dir, fake_st2_dir, split='test_set')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_st1 = DataLoader(test_st1, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloader_sd14 = DataLoader(test_sd14, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloader_xl = DataLoader(test_xl, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloader_st2 = DataLoader(test_st2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    

    if os.path.exists(save_path):
        model = load_pretrained_model(save_path)
        model.to(device)
        print(f'Loaded fine-tuned model from {save_path}')
        # Loss & optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    else:

        # Load pretrained StyleGAN1
        ckpt_path = PRETRAINED_MODELS['stylegan1']
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
    }

    for name, loader in test_sets.items():
        test_loss, test_acc = evaluate(model, loader, criterion, device)
        print(f"[{name}] - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    fine_tune(args)
