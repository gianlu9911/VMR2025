import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.utils.data import ConcatDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model
import numpy as np

def update_exemplar_set_icarl(model, dataset, old_exemplars, device, how_many=100, batch_size=64):
    model.eval()
    exemplars = [] if old_exemplars is None else old_exemplars.copy()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    with torch.no_grad():
        features_list = []
        labels_list = []
        for images, labels in dataloader:
            images = images.to(device)
            feats = model(images)  # use your model’s penultimate layer
            features_list.append(feats.cpu())
            labels_list.append(labels)
        features = torch.cat(features_list)
        labels = torch.cat(labels_list)

        # select herding exemplars for each class
        for cls in labels.unique():
            cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
            cls_features = features[cls_idx]
            # simplest: randomly pick how_many features
            selected_idx = cls_idx[torch.randperm(len(cls_idx))[:how_many]]
            exemplars.extend([(dataset[i][0], labels[i].item()) for i in selected_idx])

    return exemplars

import torch.nn.functional as F
from tqdm import tqdm


import torch

def evaluate_nme(model, exemplar_set, dataloader, device):
    """
    Evaluate using Nearest-Mean-of-Exemplars classifier.
    model: feature extractor part of the model
    exemplar_set: list of tuples (image_tensor, label)
    dataloader: test set
    """
    model.eval()
    
    # Compute class mean features
    class_means = {}
    for cls in set(l for _, l in exemplar_set):
        feats = []
        for img, label in exemplar_set:
            if label == cls:
                with torch.no_grad():
                    feat = model.feature_extractor(img.unsqueeze(0).to(device)).cpu()
                    feats.append(feat.squeeze(0))
        feats = torch.stack(feats)
        class_means[cls] = feats.mean(0)
    
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            feats = model.feature_extractor(images).cpu()
            for i, feat in enumerate(feats):
                # Compute L2 distance to each class mean
                dists = {cls: torch.norm(feat - mu) for cls, mu in class_means.items()}
                pred_class = min(dists, key=dists.get)
                if pred_class == labels[i].item():
                    correct += 1
                total += 1
    return correct / total

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            print(images.shape)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total

def train_one_epoch_icarl_binary(
    model,
    old_model,
    dataloader,
    optimizer,
    device,
    num_old_classes,
    lambda_distill=1.0
):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_samples = 0
    if old_model is not None:
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False

    with tqdm(total=len(dataloader), unit=' batch', desc='Training: ') as bar:
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # iCaRL losses
            if old_model is not None:
                with torch.no_grad():
                    old_probs = torch.sigmoid(old_model(images))

                distill_loss = F.binary_cross_entropy_with_logits(
                    outputs[:, :num_old_classes],
                    old_probs[:, :num_old_classes]
                ) if num_old_classes > 0 else 0.0

                is_new = labels >= num_old_classes
                ce_loss = F.cross_entropy(
                    outputs[is_new, num_old_classes:],
                    labels[is_new] - num_old_classes
                ) if is_new.any() else 0.0
            else:
                distill_loss = 0.0
                ce_loss = F.cross_entropy(outputs, labels)

            loss = ce_loss + lambda_distill * distill_loss
            loss.backward()
            optimizer.step()

            # Compute batch accuracy
            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).float().mean().item()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            num_samples += batch_size

            if i % 10 == 0:
                bar.set_postfix({
                    'loss': round(running_loss / num_samples, 4),
                    'acc': round(running_acc / num_samples, 4)
                })
            bar.update(1)

    return running_loss / num_samples, running_acc / num_samples


def fine_tune(
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = '0',
    epochs: int = 300,
    lr: float = 1e-3,
    seed: int = 42,
    pretrained_model: str = 'stylegan1',
    log_folder: str = './logs_icarl',
    num_points: int = None,
    order: str = '[stylegan1, stylegan2, sdv1.4, stylegan3, stylegan_xl, sdv2_1]',
    lambda_distill: float = 1.0,
    how_many: int = 100,
):
    """ICaRL fine-tune relative-representation classifier."""

    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(log_folder, exist_ok=True)

    # model is the pretrained model 
    model = load_pretrained_model(PRETRAINED_MODELS[pretrained_model])
    model.to(device)
    old_model = None
    num_old_classes = 0
    exemplar_set = None
    print(f'Loaded pretrained model from {PRETRAINED_MODELS[pretrained_model]}')


    for step in order.strip('[]').split(', '):
        # Dataset directories
        real_dir = IMAGE_DIR['real']
        fake_dir = IMAGE_DIR[step]
        dataset = RealSynthethicDataloader(real_dir, fake_dir, num_points=num_points)

        from torch.utils.data import ConcatDataset, TensorDataset

        if exemplar_set is not None and len(exemplar_set) > 0:
            old_images, old_labels = zip(*exemplar_set)
            old_dataset = TensorDataset(torch.stack(old_images), torch.tensor(old_labels))
            full_dataset = ConcatDataset([dataset, old_dataset])
        else:
            full_dataset = dataset

        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        start_time = time.time()
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch_icarl_binary(model, old_model, train_loader, optimizer, device, num_old_classes)
            print(f'Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        train_time = time.time() - start_time
        print(f'Training completed in {train_time/60:.2f} minutes')

        # Save checkpoint to the specified checkpoint_file (USER REQUEST)
        checkpoint_path = os.path.join("checkpoint", f'icarl_pretrained_{pretrained_model}.pth')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({'state_dict': model.state_dict()}, checkpoint_path)
        print(f'Model saved to {checkpoint_path}')
        old_model = copy.deepcopy(model)
        num_old_classes += 2  # Real and Fake classes added
        exemplar_set = update_exemplar_set_icarl(model, dataset, exemplar_set, device, how_many=100)

        # Prepare test datasets
        dataloaders_test = {
            "stylegan1": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan1'], split='test_set', num_points=100),
            "stylegan2": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan2'], split='test_set', num_points=100),
            "sdv1_4": RealSynthethicDataloader(real_dir, IMAGE_DIR['sdv1_4'], split='test_set', num_points=100),
            "stylegan3": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan3'], split='test_set', num_points=100),
            "stylegan_xl": RealSynthethicDataloader(real_dir, IMAGE_DIR['stylegan_xl'], split='test_set', num_points=100),
            "sdv2_1": RealSynthethicDataloader(real_dir, IMAGE_DIR['sdv2_1'], split='test_set', num_points=100),  # Uncomment if sdv2_1 is available

        }

        test_results = {}
        for name, dataset in dataloaders_test.items():
            test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_acc = evaluate(model, test_dataloader, device)
            print(f'Test Accuracy on {name}: {test_acc:.4f}')
            test_results[name] = test_acc
            print(f'Test Accuracy on {name}: {test_acc:.4f}')






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained_model', type=str, default='stylegan1', choices=['stylegan1', 'stylegan2', 'sdv1.4'], help='Pretrained model to load')
    parser.add_argument('--log_folder', type=str, default='./logs_icarl', help='Directory to save logs')
    parser.add_argument('--num_points', type=int, default=100, help='Number of real/fake images to use for training/testing - None uses all available data')
    parser.add_argument('--order', type=str, default='[stylegan1, stylegan2, sdv1.4, stylegan3, stylegan_xl, sdv2_1]', help='Order of model steps for saving features')
    parser.add_argument('--lambda_distill', type=float, default=1.0, help='Weight for distillation loss (iCaRL)')
    parser.add_argument('--how_many', type=int, default=100, help='Number of exemplars to select from each class')
    args = parser.parse_args()

    fine_tune(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        pretrained_model=args.pretrained_model,
        log_folder=args.log_folder,
        num_points=args.num_points,
        order=args.order
    )