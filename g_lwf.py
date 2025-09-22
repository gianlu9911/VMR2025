import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import PRETRAINED_MODELS, IMAGE_DIR
from src.dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model

def lwf_loss(student_outputs, teacher_outputs, labels, T=2.0, alpha=0.5):
    """
    Learning without Forgetting loss:
    - alpha: weight for cross-entropy (classification) loss
    - (1-alpha): weight for distillation (KL) loss
    - T: temperature for softening logits
    """
    ce_loss = F.cross_entropy(student_outputs, labels)
    p_teacher = F.softmax(teacher_outputs / T, dim=1)
    log_p_student = F.log_softmax(student_outputs / T, dim=1)
    distill_loss = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (T * T)
    loss = alpha * ce_loss + (1.0 - alpha) * distill_loss
    return loss, ce_loss.item(), distill_loss.item()

def train_one_epoch(student, teacher, dataloader, optimizer, device, args, print_freq=10):
    student.train()
    teacher.eval()  # teacher frozen
    running_loss = 0.0
    running_ce = 0.0
    running_distill = 0.0
    running_acc = 0.0
    num_samples = 0

    with tqdm(total=len(dataloader), unit=' batch', desc='Training: ') as bar:
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher(images)

            student_outputs = student(images)

            loss, ce_val, distill_val = lwf_loss(student_outputs, teacher_outputs, labels,
                                                 T=args.temperature, alpha=args.alpha)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(student_outputs, 1)
            acc = (preds == labels).float().mean().item()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_ce += ce_val * batch_size
            running_distill += distill_val * batch_size
            running_acc += acc * batch_size
            num_samples += batch_size

            if i % print_freq == 0:
                bar.set_postfix({
                    'loss': round(running_loss / num_samples, 4),
                    'ce': round(running_ce / num_samples, 4),
                    'distill': round(running_distill / num_samples, 4),
                    'acc': round(running_acc / num_samples, 4)
                })
            bar.update(1)

    return running_loss / num_samples, running_ce / num_samples, running_distill / num_samples, running_acc / num_samples

def evaluate(model, dataloader, criterion, device, args, set_name="test"):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    num_samples = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f'Validation ({set_name}): '):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).float().mean().item()

            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size
            val_acc += acc * batch_size
            num_samples += batch_size

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    save_dir = os.path.join(args.log_folder, "lwf", args.student_init, f"teacher_{args.teacher_init}", args.finetune_on)
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {set_name}")
    plt.tight_layout()

    safe_name = set_name.replace(" ", "_").replace("/", "_")
    save_path = os.path.join(save_dir, f"confusion_matrix_{safe_name}.png")
    plt.savefig(save_path)
    plt.close()

    return val_loss / num_samples, val_acc / num_samples

def fine_tune_lwf(args):
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    save_path = os.path.join('checkpoint', f'lwf_{args.student_init}_teacher_{args.teacher_init}_on_{args.finetune_on}.pth')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    real_dir = IMAGE_DIR['real']
    fake_st2_dir = IMAGE_DIR['stylegan2']
    fake_st1_dir = IMAGE_DIR['stylegan1']
    fake_sd14_dir = IMAGE_DIR['sdv1_4']
    fake_xl_dir = IMAGE_DIR['stylegan_xl']

    dataset = RealSynthethicDataloader(real_dir, IMAGE_DIR[args.finetune_on], num_points=args.num_points)
    test_st1 = RealSynthethicDataloader(real_dir, fake_st1_dir, split='test_set')
    test_sd14 = RealSynthethicDataloader(real_dir, fake_sd14_dir, split='test_set')
    test_xl = RealSynthethicDataloader(real_dir, fake_xl_dir, split='test_set')
    test_st2 = RealSynthethicDataloader(real_dir, fake_st2_dir, split='test_set')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_st1 = DataLoader(test_st1, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloader_sd14 = DataLoader(test_sd14, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloader_xl = DataLoader(test_xl, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloader_st2 = DataLoader(test_st2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load teacher and student from chosen pretrained checkpoints
    if args.teacher_init not in PRETRAINED_MODELS:
        raise ValueError(f"teacher_init '{args.teacher_init}' not found in PRETRAINED_MODELS.")
    if args.student_init not in PRETRAINED_MODELS:
        raise ValueError(f"student_init '{args.student_init}' not found in PRETRAINED_MODELS.")

    ckpt_path_teacher = PRETRAINED_MODELS[args.teacher_init]
    ckpt_path_student = PRETRAINED_MODELS[args.student_init]

    teacher = load_pretrained_model(ckpt_path_teacher)
    student = load_pretrained_model(ckpt_path_student)

    teacher.to(device)
    student.to(device)

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    # Resume if existing lwf checkpoint for this student/teacher exists
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device)
        if 'state_dict' in ckpt:
            student.load_state_dict(ckpt['state_dict'])
            print(f"Loaded existing LwF checkpoint from {save_path} into student.")
        else:
            student.load_state_dict(ckpt)
            print(f"Loaded existing LwF checkpoint from {save_path} into student.")

    train_losses = []
    train_ces = []
    train_distills = []
    train_accs = []

    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_loss, epoch_ce, epoch_distill, epoch_acc = train_one_epoch(
            student, teacher, dataloader, optimizer, device, args, print_freq=10
        )
        train_losses.append(epoch_loss)
        train_ces.append(epoch_ce)
        train_distills.append(epoch_distill)
        train_accs.append(epoch_acc)

        print(f'Epoch [{epoch+1}/{args.epochs}] - Loss: {epoch_loss:.4f}, CE: {epoch_ce:.4f}, Distill: {epoch_distill:.4f}, Acc: {epoch_acc:.4f}')

    end_time = time.time()
    print(f'Training completed in {(end_time - start_time)/60:.2f} minutes.')

    os.makedirs('checkpoint', exist_ok=True)
    torch.save({'state_dict': student.state_dict(), 'args': vars(args)}, save_path)
    print(f'Student model saved to {save_path}')

    plot_dir = os.path.join(args.log_folder, "lwf_plots", args.student_init, f"teacher_{args.teacher_init}")
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (LwF)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'train_loss.png'))
    plt.close()

    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy (LwF)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'train_acc.png'))
    plt.close()

    print("\nEvaluating on all test sets:")
    test_sets = {
        "real_vs_stylegan1": dataloader_st1,
        "real_vs_stylegan2": dataloader_st2,
        "real_vs_styleganxl": dataloader_xl,
        "real_vs_sdv1_4": dataloader_sd14,
    }

    for name, loader in test_sets.items():
        test_loss, test_acc = evaluate(student, loader, criterion, device, args, set_name=name)
        print(f"[{name}] - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained_model', type=str, default='stylegan1', choices=['stylegan1', 'stylegan2', 'sdv1.4'], help='(deprecated) Pretrained model to load (mantained per compatibilità)')
    parser.add_argument('--finetune_on', type=str, default='stylegan2', choices=['stylegan2', 'sdv1.4'], help='Dataset to fine-tune on')
    parser.add_argument('--log_folder', type=str, default='./logs/LwF/', help='Directory to save logs')
    parser.add_argument('--num_points', type=int, default=None, help='Number of real/fake images to use for training/testing - None uses all available data')

    # LwF specific
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for CE loss (vs distillation). CE_weight=alpha, Distill_weight=1-alpha')
    parser.add_argument('--temperature', type=float, default=2.0, help='temperature for knowledge distillation')

    # New arguments: initial weights for teacher and student
    parser.add_argument('--teacher_init', type=str, default='stylegan1', choices=list(PRETRAINED_MODELS.keys()),
                        help='Which pretrained weights to use for the teacher (default: stylegan1)')
    parser.add_argument('--student_init', type=str, default='stylegan2', choices=list(PRETRAINED_MODELS.keys()),
                        help='Which pretrained weights to use for the student initialization (default: stylegan2)')

    args = parser.parse_args()

    fine_tune_lwf(args)
