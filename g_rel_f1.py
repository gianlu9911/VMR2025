
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
from sklearn.metrics import roc_auc_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# === Avalanche Imports ===
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import (
    accuracy_metrics, 
    forgetting_metrics,  # Sostituisce il vecchio BWT
    ExperienceForgetting,
    ExperienceForwardTransfer
)
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    ExperienceForgetting
)
# Import specifico per il Forward Transfer (FWT)
from avalanche.evaluation.metrics.forward_transfer import (
    forward_transfer_metrics, 
    ExperienceForwardTransfer
)

from avalanche.evaluation import PluginMetric, Metric
from avalanche.evaluation.metrics.accuracy import ExperienceAccuracy
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TextLogger

# === Project imports ===
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model
from src.utils import RelativeRepresentation, RelClassifier, extract_and_save_features

# ------------------------------------------------------------------
# 1. CUSTOM METRICS: AUC e AUC-based Transfer
# ------------------------------------------------------------------
class ExperienceAUC(PluginMetric[float]):
    def __init__(self):
        super().__init__(AUCMetric(), reset_at='experience', emit_at='experience', mode='eval')

    # Rimuovi EvaluationContext o usa 'any' / 'TemplateDesignatedStrategy'
    def after_eval_iteration(self, strategy): 
        # strategy contiene i mini-batch correnti (mb_y e mb_output)
        self.metric.update(strategy.mb_y, strategy.mb_output)

    def __str__(self):
        return "AUC"
    
class AUCMetric(Metric[float]):
    def __init__(self):
        self._y_true = []
        self._y_score = []

    def update(self, y_true, y_score):
        self._y_true.extend(y_true.cpu().numpy())
        # Probabilità della classe 1 (Fake)
        logits = torch.softmax(y_score, dim=1)[:, 1].cpu().detach().numpy()
        self._y_score.extend(logits)

    def result(self) -> float:
        if len(np.unique(self._y_true)) < 2: return 0.5
        return roc_auc_score(self._y_true, self._y_score)

    def reset(self):
        self._y_true, self._y_score = [], []

class ExperienceAUC(PluginMetric[float]):
    def __init__(self):
        super().__init__(AUCMetric(), reset_at='experience', emit_at='experience', mode='eval')

    def after_eval_iteration(self, strategy: 'EvaluationContext'):
        self.metric.update(strategy.mb_y, strategy.mb_output)

    def __str__(self):
        return "AUC_Exp"

# ------------------------------------------------------------------
# 2. HELPER FUNCTIONS (Data & Anchors)
# ------------------------------------------------------------------

def get_feature_dataset(backbone_net, source_name, split, args, device):
    real_dir = IMAGE_DIR['real']
    fake_dir = IMAGE_DIR[source_name]
    dataset = RealSynthethicDataloader(real_dir, fake_dir, split=split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(split=='train_set'), num_workers=args.num_workers)
    
    feat_file = os.path.join(args.feature_dir, f"{split}_{source_name}_features.pt")
    if args.force_recompute_features or not os.path.exists(feat_file):
        feats, labels, _ = extract_and_save_features(backbone_net, loader, feat_file, device, split=split)
    else:
        data = torch.load(feat_file)
        feats, labels = data["features"], data["labels"]
    return TensorDataset(feats, labels), feats, labels

def get_global_anchors(real_feats, num_anchors, seed):
    rng = torch.Generator().manual_seed(seed)
    if num_anchors > len(real_feats):
        idx = torch.randint(low=0, high=len(real_feats), size=(num_anchors,), generator=rng)
    else:
        idx = torch.randperm(len(real_feats), generator=rng)[:num_anchors]
    return real_feats[idx]

# ------------------------------------------------------------------
# 3. TRAINING LOOP (Per Ordine)
# ------------------------------------------------------------------

def run_continual_order(order, order_idx, args, backbone_net, device):
    print(f"\n{'='*60}\nORDINE {order_idx}: {order}\n{'='*60}")
    
    train_datasets, test_datasets = [], []
    global_anchors = None
    
    # Estrazione feature per tutti i task dell'ordine
    for source in order:
        tr_ds, tr_f, tr_l = get_feature_dataset(backbone_net, source, 'train_set', args, device)
        te_ds, _, _ = get_feature_dataset(backbone_net, source, 'test_set', args, device)
        train_datasets.append(tr_ds)
        test_datasets.append(te_ds)
        
        if global_anchors is None:
            global_anchors = get_global_anchors(tr_f[tr_l == 0], args.num_anchors, args.seed)

    benchmark = dataset_benchmark(train_datasets=train_datasets, test_datasets=test_datasets)
    
    # Modello e Strategia
    rel_module = RelativeRepresentation(global_anchors.to(device))
    classifier = RelClassifier(rel_module, args.num_anchors, num_classes=2).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    
    # --- Configurazione Metriche ---
    # Definiamo le istanze base per l'AUC
    auc_exp_metric = ExperienceAUC()
    
    eval_plugin = EvaluationPlugin(
        # 1. Accuratezza, BWT e FWT standard
        accuracy_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        fwt_metrics(experience=True, stream=True),
        
        # 2. AUC per ogni task
        auc_exp_metric,
        
        # 3. BWT e FWT basati su AUC
        # Usiamo ExperienceForgetting (che calcola il decremento, ovvero il negativo del BWT)
        ExperienceForgetting(auc_exp_metric), 
        ExperienceForwardTransfer(auc_exp_metric),
        
        loggers=[InteractiveLogger(), TextLogger(open(f"logs/order_{order_idx}_metrics.txt", 'w'))]
    )

    strategy = Naive(
        model=classifier, optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
        train_mb_size=args.batch_size, train_epochs=args.epochs,
        device=device, evaluator=eval_plugin
    )

    # Training
    for experience in benchmark.train_stream:
        print(f"\n--- Training su: {order[experience.current_experience]} ---")
        strategy.train(experience)
        strategy.eval(benchmark.test_stream)

# ------------------------------------------------------------------
# 4. MAIN EXECUTION
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_anchors', type=int, default=1000)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--feature_dir', type=str, default='./features')
    parser.add_argument('--force_recompute_features', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backbone', type=str, default='stylegan1')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.feature_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    backbone_net = load_pretrained_model(PRETRAINED_MODELS[args.backbone])
    backbone_net.resnet.fc = nn.Identity()
    backbone_net.to(device).eval()

    orders = [
        ['stylegan1', 'stylegan2', 'sdv1_4', 'stylegan3', 'stylegan_xl', 'sdv2_1'],
        ['stylegan1', 'stylegan2', 'stylegan3', 'stylegan_xl','sdv1_4', 'sdv2_1'],
        ['sdv1_4', 'sdv2_1', 'stylegan1', 'stylegan2', 'stylegan3', 'stylegan_xl'],
        ['stylegan2', 'sdv1_4', 'stylegan_xl', 'stylegan3', 'sdv2_1', 'stylegan1']
    ]

    for i, o in enumerate(orders):
        run_continual_order(o, i + 1, args, backbone_net, device)