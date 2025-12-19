import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

LOGITS_DIR = "logs/logits"
OUTPUT_DIR = "logs/plots_pdf_train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STEPS_ORDER = [
    "stylegan1",
    "stylegan2",
    "sdv1_4",
    "stylegan3",
    "stylegan_xl",
    "sdv2_1",
]

# ============================================================
# UTILS
# ============================================================
def parse_filename(fname):
    base = os.path.basename(fname).replace(".npy", "")
    parts = base.split("_")

    is_real = parts[0].lower() == "real"
    is_fake = parts[0].lower() == "fake"
    is_anchors = parts[0].lower() == "anchors"

    faketype = None
    if "faketype" in parts:
        fidx = parts.index("faketype")
        faketype = "_".join(parts[fidx + 1:]) if fidx + 1 < len(parts) else None
        stop_idx_for_step = fidx
    else:
        stop_idx_for_step = len(parts)

    step = None
    for i, p in enumerate(parts):
        if p.lower().startswith("step"):
            if p.lower() == "step":
                tokens = parts[i + 1:stop_idx_for_step]
                step = "_".join(tokens)
            else:
                after = p[len("step"):]
                tokens = [after] + parts[i + 1:stop_idx_for_step]
                step = "_".join([t for t in tokens if t])
            break

    if step is None and len(parts) >= 2:
        step = parts[1]

    return is_real, is_fake, is_anchors, step, faketype


def mean_logits(arr_list):
    """Concatenate on first axis then mean across samples -> vector"""
    return np.mean(np.concatenate(arr_list, axis=0), axis=0)

def l2(a, b):
    return np.linalg.norm(a - b)

# ============================================================
# LOAD LOGITS
# ============================================================

real_logits = defaultdict(list)                      # real[step] -> list of arrays
anchor_logits = defaultdict(list)                    # anchors[step] -> list of arrays
fake_logits = defaultdict(lambda: defaultdict(list)) # fake[faketype][step] -> list of arrays

for fname in os.listdir(LOGITS_DIR):
    if not fname.endswith(".npy"):
        continue

    is_real, is_fake, is_anchors, step, faketype = parse_filename(fname)
    if step is None:
        continue

    path = os.path.join(LOGITS_DIR, fname)
    try:
        logits = np.load(path)
    except Exception as e:
        print(f"Skipping {fname}: {e}")
        continue

    if is_real:
        real_logits[step].append(logits)
    elif is_anchors:
        anchor_logits[step].append(logits)
    elif is_fake and faketype is not None:
        fake_logits[faketype][step].append(logits)

# ============================================================
# COMPUTE DISTANCES (ANCHORS ↔ REAL / SUBFAKES)
# Results aligned with STEPS_ORDER (lists for each label)
# ============================================================

# labels: "real" + all faketypes (sorted for deterministic plotting)
faketypes_sorted = sorted(fake_logits.keys())
all_labels = ["real"] + faketypes_sorted

# initialize distances: label -> list (aligned with STEPS_ORDER)
distances = {lbl: [] for lbl in all_labels}

for step in STEPS_ORDER:
    # if no anchors for this step -> append NaN for every label
    if step not in anchor_logits:
        for lbl in all_labels:
            distances[lbl].append(np.nan)
        continue

    anchors_mean = mean_logits(anchor_logits[step])

    # REAL
    if step in real_logits and len(real_logits[step]) > 0:
        real_mean = mean_logits(real_logits[step])
        distances["real"].append(l2(anchors_mean, real_mean))
    else:
        distances["real"].append(np.nan)

    # SUBFAKES (per faketype, not aggregated)
    for faketype in faketypes_sorted:
        if step in fake_logits[faketype] and len(fake_logits[faketype][step]) > 0:
            fake_mean = mean_logits(fake_logits[faketype][step])
            distances[faketype].append(l2(anchors_mean, fake_mean))
        else:
            distances[faketype].append(np.nan)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(14, 7))

for label in all_labels:
    y = distances[label]
    plt.plot(
        STEPS_ORDER,
        y,
        marker="o",
        linewidth=2,
        label=label
    )

plt.xlabel("Training step")
plt.ylabel("Mean L2 distance to anchors")
plt.title("Anchor ↔ Real / Subfake distance per step")
plt.grid(alpha=0.3)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "anchor_distance_per_step_real_and_subfakes.pdf")
plt.savefig(out_path)
plt.close()

print(f"Saved plot: {out_path}")
