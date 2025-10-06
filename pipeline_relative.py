#!/usr/bin/env python3
import os
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from src.utils import get_device, BalancedBatchSampler, RelativeRepresentation, RelClassifier, extract_and_save_features, evaluate, train_one_epoch, plot_features_with_anchors
from g_rel import fine_tune
# -----------------------

def run_sequential_finetunes(
    order=None,
    checkpoint_file: str = "checkpoint/checkpoint_HELLO.pth",
    verbose: bool = True,
    **fine_tune_kwargs
):
    """
    Run fine_tune sequentially over a list of datasets, loading the checkpoint
    produced by the previous step for every run after the first.

    Args:
        order (list or None): list of dataset keys to fine-tune on, in order.
                              Defaults to ['stylegan1','stylegan2','sdv1_4','stylegan3','stylegan_xl'].
        checkpoint_file (str): path used by fine_tune to save/load weights.
        verbose (bool): print progress messages.
        **fine_tune_kwargs: forwarded to fine_tune (e.g., epochs=, batch_size=, backbone=, etc.)

    Returns:
        dict: mapping fine_tuning_on -> fine_tune returned test_results
    """
    if order is None:
        order = ['stylegan1', 'stylegan2', 'sdv1_4', 'stylegan3', 'stylegan_xl']

    results_all = {}
    load_checkpoint = False  # first step must load nothing

    for domain in order:
        if verbose:
            print(f"\n>>> Starting fine-tune on '{domain}' (load_checkpoint={load_checkpoint})")

        # Ensure the fine_tune call uses the intended checkpoint_file and load flag
        kwargs = dict(fine_tune_kwargs)  # shallow copy
        kwargs.update({
            'fine_tuning_on': domain,
            'load_checkpoint': load_checkpoint,
            'checkpoint_file': checkpoint_file,
        })

        try:
            test_results = fine_tune(**kwargs)
            results_all[domain] = test_results
            if verbose:
                print(f">>> Finished '{domain}' — results keys: {list(test_results.keys())}")
        except Exception as e:
            # Stop and raise so user can inspect logs, or change this to continue on error
            print(f"Error during fine_tune on '{domain}': {e}")
            raise

        # from now on, load the checkpoint saved by the previous run
        load_checkpoint = True

    if verbose:
        print("\nAll sequential fine-tunes completed.")
    return results_all


# Example usage (put under your __main__ guard or call from other code):
if __name__ == "__main__":
    # quick example: run short experiments for debugging
    all_results = run_sequential_finetunes(
        order=['stylegan1', 'stylegan2', 'sdv1_4', 'stylegan3', 'stylegan_xl'],
        checkpoint_file="checkpoint/checkpoint_HELLO.pth",
        # any args forwarded to fine_tune:
        epochs=5,
        batch_size=512,
        num_workers=8,
        seed=42,
        num_train_samples=None,
        backbone='stylegan1',     # or whichever backbone you want
        plot_method='pca',
        force_recompute_features=False,
    )

    # print a compact summary
    for domain, res in all_results.items():
        print(f"\n=== Summary for {domain} ===")
        for test_name, metrics in res.items():
            print(f"  {test_name}: acc={metrics['acc']:.4f}, loss={metrics['loss']:.4f}")
