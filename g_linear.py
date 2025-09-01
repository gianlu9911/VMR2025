import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
import csv
import itertools
from torch.utils.data import DataLoader, Dataset, Sampler
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ==========================
# Balanced Batch Sampler
# ==========================
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels.cpu().numpy()
        self.batch_size = batch_size
        self.num_classes = len(np.unique(self.labels))
        assert batch_size % self.num_classes == 0, "batch_size must be multiple of num_classes"

        self.class_indices = {cls: np.where(self.labels == cls)[0] for cls in np.unique(self.labels)}
        self.min_class_len = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = self.min_class_len // (batch_size // self.num_classes)

    def __iter__(self):
        per_class = self.batch_size // self.num_classes
        class_iters = {cls: iter(np.random.permutation(idxs)) for cls, idxs in self.class_indices.items()}

        for _ in range(self.num_batches):
            batch = []
            for cls in self.class_indices:
                try:
                    batch.extend([next(class_iters[cls]) for _ in range(per_class)])
                except StopIteration:
                    return
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# ==========================
# Relative Representation
# ==========================
class RelativeRepresentation(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        anchors = anchors / anchors.norm(dim=1, keepdim=True)
        self.register_buffer("anchors", anchors)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return torch.matmul(x, self.anchors.T)


def to_relative(features_list, rel_modules, mode="mean"):
    reps = []
    for feats, rel in zip(features_list, rel_modules):
        reps.append(rel(feats.to(rel.anchors.device)))
    reps = torch.stack(reps)  # (num_backbones, batch, dim_rel)

    if mode == "mean":
        return reps.mean(dim=0).cpu()
    elif mode == "max":
        return reps.max(dim=0).values.cpu()
    else:
        raise ValueError("mode must be 'mean' or 'max'")


# ==========================
# Datasets
# ==========================
class MultiSourceDatasetConcat(Dataset):
    def __init__(self, sources, all_features, split):
        self.data, self.labels = [], []
        for src in sources:
            if split not in all_features[src]:
                continue
            b1, b2, y = all_features[src][split]
            feats = torch.cat([b1, b2], dim=1)
            self.data.append(feats)
            self.labels.append(y)
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]


class MultiSourceDatasetRelative(Dataset):
    def __init__(self, sources, all_features, rel_modules, split, agg_mode="mean"):
        self.data, self.labels = [], []
        for src in sources:
            if split not in all_features[src]:
                continue
            feats_list = [all_features[src][split][i] for i in range(len(rel_modules))]
            y = all_features[src][split][-1]
            relX = to_relative(feats_list, rel_modules, mode=agg_mode)
            self.data.append(relX)
            self.labels.append(y)
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]


# ==========================
# Main
# ==========================
def main(args):
    print(f"Running with mode: {args.mode}")
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    TRAIN_DATASETS = ["real", "stylegan1", "stylegan2"]
    ALL_DATASETS = ["real", "stylegan1", "stylegan2", "styleganxl", "sd14"]
    SPLITS = ["train_set", "val_set", "test_set"]
    PCTS = [0.01, 0.02, 0.03, 0.04, 0.05]

    # Load features
    all_features = {}
    for d in ALL_DATASETS:
        all_features[d] = {}
        for split in SPLITS:
            b1_path = f"features/{d}_{split}_features_b1.npy"
            b2_path = f"features/{d}_{split}_features_b2.npy"
            l_path  = f"features/{d}_{split}_labels.npy"
            if not (os.path.exists(b1_path) and os.path.exists(b2_path) and os.path.exists(l_path)):
                continue
            all_features[d][split] = (
                torch.tensor(np.load(b1_path), dtype=torch.float32),
                torch.tensor(np.load(b2_path), dtype=torch.float32),
                torch.tensor(np.load(l_path), dtype=torch.long)
            )

    summary_path = f"logs/{args.mode}_pct_summary.csv"
    if not os.path.exists(summary_path):
        with open(summary_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['Combination', 'Pct', 'ValAcc', 'DurationSec'] + 
                [f"real+{d}_TestAcc" for d in ALL_DATASETS if d != "real"]
            )

    for combo in itertools.combinations(TRAIN_DATASETS, 3):
        if "real" not in combo:
            continue
        combo_name = "_".join(combo)
        print(f"\n=== Training with sources: {combo} ===")

        # Anchors se in modalità relative
        rel_modules = None
        if args.mode == "relative":
            train_b1_real, train_b2_real, _ = all_features["real"]["train_set"]
            idx_b1 = torch.randperm(len(train_b1_real))[:args.num_anchors]
            idx_b2 = torch.randperm(len(train_b2_real))[:args.num_anchors]
            rel_b1 = RelativeRepresentation(train_b1_real[idx_b1]).to(device)
            rel_b2 = RelativeRepresentation(train_b2_real[idx_b2]).to(device)
            rel_modules = [rel_b1, rel_b2]

        for pct in PCTS:
            print(f"--- Using {int(pct*100)}% of each training source ---")

            if args.mode == "concat":
                train_dataset_full = MultiSourceDatasetConcat(combo, all_features, "train_set")
                val_dataset_full   = MultiSourceDatasetConcat(combo, all_features, "val_set")
            else:
                train_dataset_full = MultiSourceDatasetRelative(combo, all_features, rel_modules, "train_set", args.agg_mode)
                val_dataset_full   = MultiSourceDatasetRelative(combo, all_features, rel_modules, "val_set", args.agg_mode)

            # subset
            num_samples = int(pct * len(train_dataset_full))
            indices = torch.randperm(len(train_dataset_full))[:num_samples]
            train_dataset = torch.utils.data.Subset(train_dataset_full, indices)
            print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset_full)}")

            train_loader = DataLoader(
                train_dataset,
                batch_sampler=BalancedBatchSampler(train_dataset_full.labels[indices], args.batch_size)
            )
            val_loader = DataLoader(val_dataset_full, batch_size=args.batch_size, shuffle=False)

            input_dim = train_dataset_full.data.shape[1]
            model = nn.Linear(input_dim, 2).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss()

            # Training
            start_time = time.time()
            for epoch in range(args.epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch.to(device))
                    loss = criterion(outputs, y_batch.to(device))
                    loss.backward()
                    optimizer.step()
            duration = time.time() - start_time
            print(f"Training completed in {duration:.2f} seconds")

            # Validation
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch.to(device))
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds.cpu() == y_batch).sum().item()
                    total += len(y_batch)
            val_acc = correct / total if total > 0 else None
            print(f"Validation accuracy: {val_acc:.4f}" if val_acc else "No validation data")

            # Test
            eval_results = {}
            for fake in [d for d in ALL_DATASETS if d != "real"]:
                if "test_set" not in all_features[fake] or "test_set" not in all_features["real"]:
                    continue
                if args.mode == "concat":
                    X_test_ds = MultiSourceDatasetConcat(["real", fake], all_features, "test_set")
                else:
                    X_test_ds = MultiSourceDatasetRelative(["real", fake], all_features, rel_modules, "test_set", args.agg_mode)
                test_loader = DataLoader(X_test_ds, batch_size=args.batch_size, shuffle=False)

                correct, total = 0, 0
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        outputs = model(X_batch.to(device))
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds.cpu() == y_batch).sum().item()
                        total += len(y_batch)
                acc = correct / total
                eval_results[f"real+{fake}"] = acc
                print(f"[{combo_name}] {pct*100:.0f}% Tested on real+{fake}: Acc={acc:.4f}")

            # Save row
            row = [combo_name, int(pct*100), val_acc, duration] + [eval_results.get(f"real+{d}", None) for d in ALL_DATASETS if d != "real"]
            with open(summary_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)

            # Save checkpoint
            checkpoint = {
                "combo_name": combo_name,
                "pct": pct,
                "mode": args.mode,
                "input_dim": input_dim,
                "classifier_state": model.state_dict(),
            }
            if args.mode == "relative":
                checkpoint["rel_modules"] = [rel_modules[0].state_dict(), rel_modules[1].state_dict()]
                checkpoint["num_anchors"] = args.num_anchors
                checkpoint["agg_mode"] = args.agg_mode

            model_filename = f"checkpoint/{combo_name}_{int(pct*100)}pct_{args.mode}.pth"
            torch.save(checkpoint, model_filename)
            print(f"Model saved to {model_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default="relative", choices=["concat", "relative"])
    parser.add_argument('--num_anchors', type=int, default=600)
    parser.add_argument('--agg_mode', type=str, default="mean", choices=["mean", "max"])
    args = parser.parse_args()
    main(args)
