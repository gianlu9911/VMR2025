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
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import csv
import json
# === Project imports - adjust these to your repo layout ===
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model
from src.utils import  BalancedBatchSampler, RelativeRepresentation, RelClassifier, extract_and_save_features, evaluate, train_one_epoch, plot_features_with_anchors
from g_rel import fine_tune
# -----------------------


def run_sequential_finetunes(
    order=None,
    checkpoint_file: str = "checkpoint/checkpoint_HELLO.pth",
    verbose: bool = True,
    sample_show: int = 50,            # how many raw diffs to show in the "sample" CSV
    topk_show: int = 50,             # how many largest diffs to list in the top-k CSV
    dump_full_threshold: int = 100000, # above this number of elements we don't dump full diffs
    float_fmt: str = "{:.6g}",      # formatting for sample outputs (human-readable)
    **fine_tune_kwargs
):
    """
    Run fine_tune sequentially over a list of datasets, loading the checkpoint
    produced by the previous step for every run after the first.

    This version saves **human-friendly** summaries of anchor differences for
    easier inspection. For each domain you'll get a compact `summary_<domain>.csv`
    plus a small `sample` file and a `topk` file listing the largest absolute
    differences. When anchors are dicts, the same per-key summaries are saved
    under `anchros/<domain>/`.
    """
    if order is None:
        order = ['stylegan1', 'stylegan2', 'sdv1_4', 'stylegan3', 'stylegan_xl','sdv2_1']

    results_all = {}
    load_checkpoint = False  # first step must load nothing
    anchros = None

    # helper: convert many possible tensor/array types to numpy
    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        try:
            return np.asarray(x)
        except Exception:
            return None

    # helper: create a human-friendly summary for a 1D numpy array 'diff'
    def summarize_and_dump(diff, out_dir, name_prefix):
        """Writes summary_<name_prefix>.csv, sample_<name_prefix>.csv, topk_<name_prefix>.csv
        and optionally the full results_<name_prefix>.csv if array is small enough.
        Returns the summary dict."""
        os.makedirs(out_dir, exist_ok=True)
        diff = np.asarray(diff).flatten()
        n = diff.size
        if n == 0:
            summary = {"num_elements": 0}
            with open(os.path.join(out_dir, f"summary_{name_prefix}.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerow(["num_elements", 0])
            return summary

        mean = float(np.mean(diff))
        std = float(np.std(diff))
        median = float(np.median(diff))
        q25 = float(np.percentile(diff, 25))
        q75 = float(np.percentile(diff, 75))
        minv = float(np.min(diff))
        maxv = float(np.max(diff))
        mad = float(np.mean(np.abs(diff - mean)))
        num_pos = int(np.sum(diff > 0))
        num_neg = int(np.sum(diff < 0))
        num_zero = int(np.sum(diff == 0))

        summary = {
            "num_elements": int(n),
            "mean": mean,
            "std": std,
            "median": median,
            "25%": q25,
            "75%": q75,
            "min": minv,
            "max": maxv,
            "mad": mad,
            "num_positive": num_pos,
            "num_negative": num_neg,
            "num_zero": num_zero,
        }

        # Save a compact machine-friendly CSV summary
        summary_path = os.path.join(out_dir, f"summary_{name_prefix}.csv")
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in summary.items():
                writer.writerow([k, v])

        # Save a JSON summary too (handy for programmatic loading)
        with open(os.path.join(out_dir, f"summary_{name_prefix}.json"), 'w') as jf:
            json.dump(summary, jf, indent=2)

        # Save a human-readable SAMPLE of the first sample_show values (formatted)
        sample_n = min(sample_show, n)
        sample_path = os.path.join(out_dir, f"sample_{name_prefix}.csv")
        with open(sample_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "difference", "abs_difference"])
            for i in range(sample_n):
                d = diff[i]
                writer.writerow([i, float_fmt.format(d), float_fmt.format(abs(d))])

        # Save top-k largest absolute differences for quick inspection
        topk = min(topk_show, n)
        abs_idx = np.argsort(np.abs(diff))[::-1][:topk]
        topk_path = os.path.join(out_dir, f"topk_{name_prefix}.csv")
        with open(topk_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "index", "difference", "abs_difference"]) 
            for rank, idx in enumerate(abs_idx, start=1):
                d = float(diff[idx])
                writer.writerow([rank, int(idx), float_fmt.format(d), float_fmt.format(abs(d))])

        # Optionally dump full differences if array is small
        if n <= dump_full_threshold:
            full_path = os.path.join(out_dir, f"results_{name_prefix}.csv")
            with open(full_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["index", "difference", "abs_difference"]) 
                for i, d in enumerate(diff):
                    writer.writerow([i, float(d), float(abs(d))])

        return summary
    for domain in order:
        if verbose:
            print(f">>> Starting fine-tune on '{domain}' (load_checkpoint={load_checkpoint})")

        # Ensure the fine_tune call uses the intended checkpoint_file and load flag
        kwargs = dict(fine_tune_kwargs)  # shallow copy
        kwargs.update({
            'fine_tuning_on': domain,
            'load_checkpoint': load_checkpoint,
            'checkpoint_file': checkpoint_file,
            'save_feats_prefix': f'saved_numpy_features/step_{domain}',
            'save_feats': True,
        })
        os.makedirs("anchros", exist_ok=True)

        try:
            test_results, new_anchros = fine_tune(**kwargs)
            results_all[domain] = test_results

            # --- Compare old anchors and new anchors and save differences + human summaries ---
            if anchros is not None:
                
                # Treat anchors as single arrays / tensors
                old_arr = to_numpy(anchros)
                new_arr = to_numpy(new_anchros)
                if old_arr is None or new_arr is None:
                    if verbose:
                        print("Could not convert anchros to numpy arrays for diffing. Skipping.")
                else:
                    # attempt to align shapes
                    if old_arr.shape != new_arr.shape:
                        if verbose:
                            print(f"Warning: shape mismatch for anchros: old={old_arr.shape}, new={new_arr.shape}. Flattening and aligning to min length.")
                        old_flat = old_arr.flatten()
                        new_flat = new_arr.flatten()
                        min_len = min(old_flat.size, new_flat.size)
                        diff = new_flat[:min_len] - old_flat[:min_len]
                    else:
                        diff = new_arr - old_arr

                    # Write nice human-friendly summaries into anchros/<domain>/
                    outdir = os.path.join("anchros", domain)
                    os.makedirs(outdir, exist_ok=True)
                    summary = summarize_and_dump(diff, outdir, domain)


            # --- Update anchors for next step ---
            anchros = new_anchros
            if verbose:
                print(f">>> Finished '{domain}' — results keys: {list(test_results.keys())}")
        except Exception as e:
            # Stop and raise so user can inspect logs, or change this to continue on error
            print(f"Error during fine_tune on '{domain}': {e}")
            raise

        # from now on, load the checkpoint saved by the previous run
        load_checkpoint = True

    if verbose:
        print("All sequential fine-tunes completed.")
    return results_all


# Example usage (put under your __main__ guard or call from other code):
if __name__ == "__main__":
    # quick example: run short experiments for debugging
    order = ['sdv1_4', 'sdv2_1','stylegan1', 'stylegan2', 'stylegan3', 'stylegan_xl',]
    randminzed_order = np.random.permutation(order).tolist()
    print(f"Running sequential fine-tunes in order: {randminzed_order}")
    all_results = run_sequential_finetunes(
        order=randminzed_order,
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
        print(f"=== Summary for {domain} ===")
        for test_name, metrics in res.items():
            print(f"  {test_name}: acc={metrics['acc']:.4f}, loss={metrics['loss']:.4f}")
