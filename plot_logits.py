import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

LOGITS_DIR = "logs/train/logits"
OUTPUT_DIR = "logs/plots_pdf_train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Softmax function ----
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

# ---- Parse filename ----
def parse_filename(fname):
    base = fname.replace(".npy", "")
    parts = base.split("_")

    is_real = parts[0] == "real"
    is_fake = parts[0] == "fake"
    is_anchors = parts[0] == "anchors"

    step = None
    if "step" in parts:
        step_idx = parts.index("step")
        step = parts[step_idx + 1]

    faketype = None
    if is_fake and "faketype" in parts:
        fidx = parts.index("faketype")
        faketype = "_".join(parts[fidx + 1:])

    return is_real, is_fake, is_anchors, step, faketype

# ---- Load logits ----
logits_by_step = defaultdict(lambda: defaultdict(list))
how_many = 50  # limit for loading logits
for fname in os.listdir(LOGITS_DIR):
    if not fname.endswith(".npy"):
        continue

    is_real, is_fake, is_anchors, step, faketype = parse_filename(fname)
    logits = np.load(os.path.join(LOGITS_DIR, fname))  # shape (N, 2)
    print(f"{fname}: shape={logits.shape}")

    if is_real:
        logits_by_step[step]["real"].append(logits)
    elif is_fake:
        logits_by_step[step].setdefault("fake", defaultdict(list))
        logits_by_step[step]["fake"][faketype].append(logits)
    elif is_anchors:
        logits_by_step[step].setdefault("anchors", [])
        logits_by_step[step]["anchors"].append(logits)

# ---- Scatterplot + SAVE PDF (first 10 logits) ----
for step, groups in logits_by_step.items():
    plt.figure(figsize=(8, 6))
    alpha_val = 0.5  # transparency

    # Real (stars)
    if "real" in groups:
        all_real = np.concatenate(groups["real"], axis=0)[:how_many]  # take first 10
        #all_real = softmax(all_real, axis=1)
        plt.scatter(all_real[:, 0], all_real[:, 1],
                    s=60, alpha=alpha_val, marker='*', label="real")

    # Anchors (triangles)
    if "anchors" in groups:
        all_anchors = np.concatenate(groups["anchors"], axis=0)[:how_many]
        #all_anchors = softmax(all_anchors, axis=1)
        plt.scatter(all_anchors[:, 0], all_anchors[:, 1],
                    s=60, alpha=alpha_val, marker='^', label="anchors")

    # Fake (circles)
    if "fake" in groups:
        for faketype, arr_list in groups["fake"].items():
            all_fake = np.concatenate(arr_list, axis=0)[:how_many]
            #all_fake = softmax(all_fake, axis=1)
            plt.scatter(all_fake[:, 0], all_fake[:, 1],
                        s=60, alpha=alpha_val, marker='o', label=f"fake")

    plt.xlabel("Probability class 0")
    plt.ylabel("Probability class 1")
    plt.title(f"2D Logits Scatterplot - step {step}")
    plt.legend(markerscale=1.5)
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"logits_scatter_step_{step}.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved PDF: {out_path}")

