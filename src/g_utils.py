import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import csv
import os

from glob import glob
from PIL import Image
import random

from config import PRETRAINED_MODELS, IMAGE_DIR
from .net import load_pretrained_model
from .transform import data_transforms

def train_one_epoch(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss, running_acc, n = 0, 0, 0

    for images, labels in tqdm(train_loader, desc='Train: '):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels) # cross entropy loss
        loss.backward()
        optimizer.step() # adam optimizer

        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

        running_loss += loss.item() * labels.size(0)
        running_acc  += acc * labels.size(0)
        n += labels.size(0)

    return running_loss / n, running_acc / n

def accuracy(output, labels):
    with torch.no_grad():
        _, predicted = torch.max(output.data, 1)
        return (predicted == labels).float().mean().item()

def evaluate(dataloader, model, criterion, args):
    model.eval()
    val_loss = 0
    val_acc = 0
    k = 0

    with torch.no_grad():
        for images, target in tqdm(dataloader, desc='Evaluation: '):
            images, target = images.to(args.device), target.to(args.device)
            output = model(images)
            val_loss_it = criterion(output, target)
            val_acc_it = accuracy(output, target)

            val_loss += val_loss_it.item() * target.size(0)
            val_acc  += val_acc_it * target.size(0)
            k += target.size(0)

    val_loss /= k
    val_acc  /= k
    return val_loss, val_acc

def count_images_in_dir(base_dir, split):
    pngs = glob(os.path.join(base_dir, split, '*.png'))
    jpgs = glob(os.path.join(base_dir, split, '*.jpg'))
    pngs = list(set(pngs + jpgs))  # Deduplicate across both
    return len(pngs)