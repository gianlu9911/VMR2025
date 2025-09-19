#!/usr/bin/env python3
"""
Training script that implements Learning without Forgetting (LwF) + replay on top of the provided baseline.

Features:
- Student is trained with CrossEntropy + Knowledge Distillation (teacher soft targets)
- Simple replay buffer (store past samples and replay a small batch each iteration)
- CLI options: number of images used in the training dataset, teacher weights, student init (random or checkpoint), dataset for training
- Evaluation after every epoch on all available test datasets listed in IMAGE_DIR

Assumptions:
- `load_pretrained_model(checkpoint_path)` returns a model instance (architecture + weights if checkpoint_path provided)
- `RealSynthethicDataloader(real_dir, fake_dir)` returns a torch.utils.data.Dataset
- `PRETRAINED_MODELS` and `IMAGE_DIR` come from config.py

"""

import os
import random
import argparse
from copy import deepcopy
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model


# ---------- utilities ----------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_weights(m):
    """Simple weight reset for common layers."""
    for layer in m.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            try:
                nn.init.kaiming_normal_(layer.weight)
            except Exception:
                pass
            if getattr(layer, 'bias', None) is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Linear):
            try:
                nn.init.xavier_normal_(layer.weight)
            except Exception:
                pass
            if getattr(layer, 'bias', None) is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.BatchNorm2d):
            try:
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            except Exception:
                pass


class ReplayBuffer:
    """A tiny replay buffer storing images and labels on CPU to be mixed into training batches."""
    def __init__(self, max_size=5000):
        self.max_size = max_size
        self.images = deque(maxlen=max_size)
        self.labels = deque(maxlen=max_size)

    def add_batch(self, imgs, labels):
        # imgs: tensor (B,C,H,W) on any device. We'll store on CPU.
        imgs_cpu = imgs.detach().cpu()
        labels_cpu = labels.detach().cpu()
        for i in range(imgs_cpu.size(0)):
            self.images.append(imgs_cpu[i])
            self.labels.append(labels_cpu[i])

    def sample(self, k):
        k = min(k, len(self.images))
        if k == 0:
            return None, None
        idx = np.random.choice(len(self.images), size=k, replace=False)
        imgs = torch.stack([self.images[i] for i in idx], dim=0)
        labels = torch.stack([self.labels[i] for i in idx], dim=0)
        return imgs, labels

    def __len__(self):
        return len(self.images)


# ---------- evaluation (adapted) ----------

def accuracy(output, labels):
    with torch.no_grad():
        batch_size = labels.size(0)
        _, predicted = torch.max(output.data, 1)
        res = (predicted == labels).sum().item() / batch_size
    return res


def evaluate(dataloader, model, criterion, device, print_freq=10):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    k = 0

    with torch.no_grad():
        with tqdm(total=len(dataloader), unit=' batch', desc='Evaluation: ', leave=False) as bar:
            for i, (images, target) in enumerate(dataloader):
                bar.update(1)
                images = images.to(device)
                target = target.to(device)

                output = model(images)
                val_loss_it = criterion(output, target)
                val_acc_it = accuracy(output, target)

                val_acc += val_acc_it * target.size(0)
                val_loss += val_loss_it.item() * target.size(0)
                k += target.size(0)

                if i % print_freq == 0:
                    bar.set_postfix({'batch_test_loss': round(val_loss_it.item(), 5),
                                     'batch_test_acc': round(val_acc_it, 5),
                                     'test_loss': round(val_loss / k, 5) if k > 0 else 0,
                                     'test_acc': round(val_acc / k, 5) if k > 0 else 0
                                     })
            bar.close()

    if k == 0:
        return float('nan'), float('nan')

    val_loss /= k
    val_acc /= k
    return val_loss, val_acc


# ---------- knowledge distillation loss ----------

def distillation_loss(student_logits, teacher_logits, T=2.0, reduction='batchmean'):
    """KL Divergence between soft targets. Returns a scalar."""
    # student: logits [B, C], teacher: logits [B, C]
    log_p = nn.functional.log_softmax(student_logits / T, dim=1)
    q = nn.functional.softmax(teacher_logits / T, dim=1)
    kd_loss = nn.KLDivLoss(reduction=reduction)(log_p, q) * (T * T)
    return kd_loss


# ---------- training loop ----------

def train(args):
    # device handling
    if args.device is not None and args.device != '':
        if torch.cuda.is_available():
            args.device = f'cuda:{args.device}'
            print(f'Using GPU: {args.device}')
        else:
            print('GPUs not available. Using CPU')
            args.device = 'cpu'
            args.num_workers = 1
    else:
        print('Using CPU')
        args.device = 'cpu'
        args.num_workers = 1

    device = torch.device(args.device)

    if args.seed is not None:
        set_seed(args.seed)

    # Teacher model
    if args.teacher_weights is not None and args.teacher_weights != '':
        teacher_ck = PRETRAINED_MODELS[args.teacher_weights] if args.teacher_weights in PRETRAINED_MODELS else args.teacher_weights
    else:
        # fallback: prefer 'stylegan1' if available, otherwise take the first entry in PRETRAINED_MODELS
        if 'stylegan1' in PRETRAINED_MODELS:
            teacher_ck = PRETRAINED_MODELS['stylegan1']
        else:
            teacher_ck = next(iter(PRETRAINED_MODELS.values()), None)
    if teacher_ck is None:
        raise ValueError('Teacher checkpoint not provided and PRETRAINED_MODELS is empty')

    print(f'Loading teacher from: {teacher_ck}')
    teacher = load_pretrained_model(teacher_ck)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student model: instantiate same architecture as teacher then optionally reset
    student = load_pretrained_model(PRETRAINED_MODELS[args.student_weights])  # architecture loaded
    print(f'Student initialized from: {args.student_weights}')


    student.to(device)

    # optimizer
    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # dataset for training
    if args.train_dataset not in IMAGE_DIR:
        raise ValueError(f"Train dataset '{args.train_dataset}' not found in IMAGE_DIR keys: {list(IMAGE_DIR.keys())}")

    real_dir = IMAGE_DIR['real']
    fake_dir = IMAGE_DIR[args.train_dataset]
    train_ds = RealSynthethicDataloader(real_dir, fake_dir)

    # allow limiting number of images
    if args.num_images is not None and args.num_images > 0 and args.num_images < len(train_ds):
        print(f'Subsetting training dataset to {args.num_images} images (randomly)')
        perm = torch.randperm(len(train_ds))[:args.num_images]
        train_ds = Subset(train_ds, perm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    criterion = nn.CrossEntropyLoss()

    # replay buffer
    replay = ReplayBuffer(max_size=args.replay_size) if args.replay_size > 0 else None

    # training loop
    print('\nStarting training...')
    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_kd = 0.0
        processed = 0

        with tqdm(total=len(train_loader), unit=' batch', desc=f'Epoch {epoch}/{args.epochs}') as pbar:
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # optionally sample replay and mix
                if replay is not None and len(replay) > 0 and args.replay_batch > 0:
                    r_imgs, r_labels = replay.sample(args.replay_batch)
                    if r_imgs is not None:
                        # move replayed samples to device
                        r_imgs = r_imgs.to(device)
                        r_labels = r_labels.to(device)
                        # concatenate
                        images = torch.cat([images, r_imgs], dim=0)
                        labels = torch.cat([labels, r_labels], dim=0)

                optimizer.zero_grad()

                # forward
                student_logits = student(images)
                with torch.no_grad():
                    teacher_logits = teacher(images)

                ce_loss = criterion(student_logits, labels)
                kd_loss = distillation_loss(student_logits, teacher_logits, T=args.temperature)
                loss = ce_loss + args.lambda_kd * kd_loss

                loss.backward()
                optimizer.step()

                # add original (non-mixed) batch to replay (so that replay remains pure)
                if replay is not None:
                    replay.add_batch(images[:args.batch_size].detach().cpu(), labels[:args.batch_size].detach().cpu())

                epoch_loss += loss.item() * images.size(0)
                epoch_ce += ce_loss.item() * images.size(0)
                epoch_kd += kd_loss.item() * images.size(0)
                processed += images.size(0)

                pbar.set_postfix({'loss': round(epoch_loss / processed, 5),
                                  'CE': round(epoch_ce / processed, 5),
                                  'KD': round(epoch_kd / processed, 5),
                                  'replay_len': len(replay) if replay is not None else 0
                                  })
                pbar.update(1)
            pbar.close()

        scheduler.step()

        # --- evaluation on all available test datasets ---
        print('\nEvaluating on all available test datasets:')
        results = {}
        for test_name, test_dir in IMAGE_DIR.items():
            # skip training directories (we want to evaluate on all available test datasets)
            rgb_real = IMAGE_DIR['real']
            rgb_fake = IMAGE_DIR[test_name]
            test_ds = RealSynthethicDataloader(rgb_real, rgb_fake)
            test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers)
            val_loss, val_acc = evaluate(test_loader, student, criterion, device)
            results[test_name] = (val_loss, val_acc)
            print(f' - {test_name}: loss={val_loss:.6f}, acc={val_acc:.6f}')

        # save student checkpoint
        if args.save_student is not None and args.save_student != '':
            save_path = args.save_student.format(epoch=epoch)
            torch.save(student.state_dict(), save_path)
            print(f'Student checkpoint saved to {save_path}')

    print('\nTraining finished')


# ---------- CLI ----------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train student with LwF + replay')

    parser.add_argument('--train_dataset', type=str, default='stylegan1', help='Which fake dataset to use for training (key in IMAGE_DIR). Default: stylegan2')
    parser.add_argument('--num_images', type=int, default=-1, help='Number of images to use from training dataset (random subset). -1 = use all')

    parser.add_argument('--teacher_weights', type=str, default='', help='Optional explicit path to teacher checkpoint. If empty, PRETRAINED_MODELS default (stylegan1 if present) is used')
    parser.add_argument('--student_weights', type=str, default='stylegan2', help="How to init student: 'random' or path to checkpoint. Default: random")
    parser.add_argument('--save_student', type=str, default='student_epoch{epoch}.pth', help='Path template to save student checkpoints (use {epoch}). Empty to not save')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0', help='GPU id or empty for CPU')

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--lambda_kd', type=float, default=1.0, help='Weight for KD loss')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for KD')

    parser.add_argument('--replay_size', type=int, default=5000, help='Maximum number of samples stored in replay buffer (0 to disable)')
    parser.add_argument('--replay_batch', type=int, default=16, help='How many replay samples to mix into each training batch')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # normalize num_images
    if args.num_images <= 0:
        args.num_images = None

    train(args)
