from g_rel import *
import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# === Project imports - adjust these to your repo layout ===
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model
from src.utils import extract_and_save_features, train_one_epoch, plot_features_with_anchors, RelativeRepresentation, RelClassifier

def pipeline(backbone_name, task, seed, batch_size, num_workers, num_points, num_anchors, saved_accuracy_path, lr, epochs,feature_path=None,
    checkpoint_path=None, device=None,force_recompute_features=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    class_weights = None
    print(f"Using device: {device}")
    in_folder = "num_train_samples_"+str(num_points) if num_points is not None else "all_train_samples"
    in_folder = in_folder + "_num_anchors_"+str(num_anchors)
    for t in task:
        saved_accuracy_path = os.path.join('results_relative', in_folder,f'relative_accuracy_{t}.csv')
        os.makedirs(os.path.dirname(saved_accuracy_path), exist_ok=True)
        checkpoint_path = os.path.join('checkpoints_rel', in_folder,f'relative_{backbone_name}_to_{t}.pth')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        feature_path = feature_path or os.path.join('features_relative', f'features_{backbone_name}_to_{t}.pt')
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)

    
        print(f"\n=== Task: {t} ===")
        results, checkpoint = fine_tune(backbone_name=backbone_name, fine_tuning_on=t, seed=seed, batch_size=batch_size, 
                                        num_workers=num_workers, num_points=num_points, num_anchors=num_anchors, 
                                        saved_accuracy_path=saved_accuracy_path, lr=lr, epochs=epochs, feature_path=feature_path,
                checkpoint_path=checkpoint_path, device=device,force_recompute_features=force_recompute_features, classifier_weight_path=class_weights) 
        class_weights = checkpoint
        
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help="Number of training samples to use. If None, use all available samples.")
    parser.add_argument('--task', type=str, default=['stylegan1', 'stylegan2','sdv1_4','stylegan3', 'stylegan_xl'])
    parser.add_argument('--num_anchors', type=int, default=5000,
                        help="Exact number of real features to use as anchors; if greater than available reals, sampling with replacement is used.")
    parser.add_argument('--backbone', type=str, default='stylegan1',
                        choices=['stylegan1', 'stylegan2', 'stylegan_xl', 'sdv1_4', 'stylegan3'])
    parser.add_argument('--force_recompute_features', action='store_true', default=False,
                        help="If set, forces re-computation of features even if they are already saved on disk.")
    parser.add_argument('--feature_path', type=str, default=None,
                        help="Path to save/load features. If None, a default path is used.")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="Path to save/load model checkpoints. If None, a default path is used.")
    parser.add_argument('--saved_accuracy_path', type=str, default=None,
                        help="Path to save/load accuracy results. If None, a default path is used.")
    args = parser.parse_args()
    pipeline(backbone_name=args.backbone, task=args.task, seed=args.seed, batch_size=args.batch_size, 
             num_workers=args.num_workers, num_points=args.num_train_samples, num_anchors=args.num_anchors, saved_accuracy_path='test_accuracies.csv', lr=args.lr, epochs=args.epochs,feature_path=None,
            checkpoint_path=None, device=None,force_recompute_features=False
        
    )



