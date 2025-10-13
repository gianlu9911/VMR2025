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
from pipeline_relative import run_sequential_finetunes
# -----------------------


import os

def run_param_sweep(
    backbones,
    num_train_samples_list,
    num_anchors_list,
    order=None,
    base_checkpoint_template: str = "checkpoint/checkpoint_HELLO_{backbone}.pth",
    force_recompute_features_first_run: bool = False,
    remove_existing_checkpoint_before_run: bool = False,
    verbose: bool = True,
    **fine_tune_kwargs
):
    """
    Esegue la catena sequenziale di fine-tune per tutte le combinazioni di:
      - backbones (lista di backbone keys)
      - num_train_samples (lista di int or None; None -> 'all')
      - num_anchors (lista di int or None; None -> 'all')

    Per ogni combinazione crea un CSV di eval il cui filename contiene backbone, ntrain e nanchors.
    Restituisce un dict con i risultati.

    Args:
        backbones (list[str]): es. ['stylegan1'] o più chiavi da PRETRAINED_MODELS.
        num_train_samples_list (list[int|None]): numeri di training sample da testare (None == use all).
        num_anchors_list (list[int|None]): numeri di anchors da testare (None == use all).
        order (list[str]|None): ordine dei domini nella catena (default: ['stylegan1','stylegan2','sdv1_4','stylegan3','stylegan_xl']).
        base_checkpoint_template (str): template del checkpoint per-backbone (deve contenere '{backbone}').
        force_recompute_features_first_run (bool): se True, forza `force_recompute_features=True` nella prima chiamata di ogni chain per quel backbone.
        remove_existing_checkpoint_before_run (bool): se True, rimuove checkpoint_file esistente prima di ogni combinazione per partire puliti.
        verbose (bool): stampa messaggi.
        **fine_tune_kwargs: argomenti inoltrati a fine_tune (es. epochs, batch_size, device, backbone verrà sovrascritto internamente).

    Returns:
        dict: mapping (backbone, ntrain_repr, nanchors_repr) -> results_from_run_sequential_finetunes
    """
    if order is None:
        order = ['stylegan1', 'stylegan2', 'sdv1_4', 'stylegan3', 'stylegan_xl']

    results = {}

    for backbone in backbones:
        for ntrain in num_train_samples_list:
            # friendly string for filenames
            ntrain_str = 'all' if ntrain is None else str(ntrain)
            for nanchors in num_anchors_list:
                nanchors_str = 'all' if nanchors is None else str(nanchors)

                # create CSV path that encodes the config
                csv_fname = f"logs/eval_{backbone}_ntrain{ntrain_str}_nanchors{nanchors_str}.csv"
                os.makedirs(os.path.dirname(csv_fname), exist_ok=True)

                # per-backbone checkpoint
                checkpoint_file = base_checkpoint_template.format(backbone=backbone)
                if remove_existing_checkpoint_before_run and os.path.exists(checkpoint_file):
                    try:
                        os.remove(checkpoint_file)
                        if verbose:
                            print(f"[info] Removed existing checkpoint {checkpoint_file} to start fresh.")
                    except Exception as e:
                        print(f"[warning] Couldn't remove checkpoint {checkpoint_file}: {e}")

                if verbose:
                    print("\n" + "="*80)
                    print(f"[RUN] backbone={backbone} | ntrain={ntrain_str} | nanchors={nanchors_str}")
                    print(f"[RUN] csv -> {csv_fname}")
                    print("="*80)

                # prepare kwargs to pass to run_sequential_finetunes (which will call fine_tune)
                kwargs = dict(fine_tune_kwargs)
                # set per-run params (these will be forwarded to each fine_tune call)
                kwargs.update({
                    'num_train_samples': ntrain,
                    'num_anchors': nanchors,
                    'eval_csv_path': csv_fname,
                    # ensure fine_tune uses the correct backbone
                    'backbone': backbone,
                    # ensure feature recompute behavior for the first call in the chain
                    'force_recompute_features': force_recompute_features_first_run,
                })

                # call the sequential runner (it will save/load checkpoint_file as it goes)
                try:
                    run_results = run_sequential_finetunes(
                        order=order,
                        checkpoint_file=checkpoint_file,
                        verbose=verbose,
                        **kwargs
                    )
                    results[(backbone, ntrain_str, nanchors_str)] = run_results
                except Exception as e:
                    print(f"[error] Sweep failed for backbone={backbone}, ntrain={ntrain_str}, nanchors={nanchors_str}: {e}")
                    # decide se interrompere o continuare: qui continuiamo e registriamo l'errore
                    results[(backbone, ntrain_str, nanchors_str)] = {'error': str(e)}
    return results


if __name__ == "__main__":
    # sweep di esempio: prova 3 valori di num_train_samples e 2 di num_anchors su un solo backbone
    sweep_results = run_param_sweep(
        backbones=['stylegan1'],  # o passa più backbone
        num_train_samples_list=[50, 100, None],   # None == use all samples
        num_anchors_list=[100, 5000],
        epochs=300,                # per test veloce; metti quello che ti serve
        batch_size=16,
        num_workers=8,
        seed=42,
        force_recompute_features_first_run=False,
        verbose=True,
    )

    # stampa un riassunto compatto
    for k, v in sweep_results.items():
        print(f"\n=== Config {k} ===")
        if 'error' in v:
            print("  ERROR:", v['error'])
        else:
            for domain, metrics in v.items():
                print(f"  {domain}:")
                for test_name, vals in metrics.items():
                    print(f"    - {test_name}: acc={vals['acc']:.4f}, loss={vals['loss']:.4f}")
