import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

LOGITS_DIR = "logs/logits"
ANCHORS_DIR = "logs/train/logits"
OUTPUT_DIR = "logs/plots_pdf_train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Softmax function ----
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

# ---- Robust Parse filename ----
def parse_filename(fname):
    """
    Esempi gestiti:
    - fake_step_sdv1_4_faketype_stylegan3.npy  -> step='sdv14', faketype='stylegan3'
    - anchors_step_stylegan2.npy              -> step='stylegan2', anchors=True
    - real_step_1000.npy                      -> step='1000'
    - fake_step100_faketype_stylegan_xl.npy   -> step='100', faketype='stylegan_xl'
    """
    base = os.path.basename(fname).replace(".npy", "")
    parts = base.split("_")

    # tipo
    is_real = parts[0].lower() == "real"
    is_fake = parts[0].lower() == "fake"
    is_anchors = parts[0].lower() == "anchors"

    # trova indice di 'faketype' se presente
    faketype = None
    if "faketype" in parts:
        fidx = parts.index("faketype")
        # lascia underscores nel faketype per preservare stylegan_xl
        faketype = "_".join(parts[fidx + 1:]) if fidx + 1 < len(parts) else None
        stop_idx_for_step = fidx
    else:
        stop_idx_for_step = len(parts)

    # trova indice di 'step' (o token che inizia con 'step')
    step = None
    step_idx = None
    for i, p in enumerate(parts):
        if p.lower().startswith("step"):
            step_idx = i
            break

    if step_idx is not None:
        # se token è esattamente 'step' prendo i token successivi fino a faketype/end
        start_token = parts[step_idx]
        if start_token.lower() == "step":
            # prendi tutti i token tra 'step' e 'faketype' (es. sdv1, 4 -> sdv14)
            tokens = parts[step_idx + 1:stop_idx_for_step]
            step = "".join(tokens) if tokens else None
        else:
            # es. 'step100' oppure 'stepv1' -> prendo la parte dopo 'step' e poi eventuali token successivi
            after = start_token[len("step"):]
            tokens = [after] + parts[step_idx + 1:stop_idx_for_step]
            # rimuovo eventuali token vuoti e unisco senza underscore
            tokens = [t for t in tokens if t]
            step = "".join(tokens) if tokens else None

    # fallback: se non ho trovato 'step' ma il filename ha solo un token tipo 'anchors_stylegan2'
    if step is None:
        # se c'è un token dopo il tipo 'anchors'/'real'/'fake' e non è 'faketype', usalo
        if len(parts) >= 2 and parts[1].lower() != "faketype":
            step = parts[1]

    return is_real, is_fake, is_anchors, step, faketype

# ---- Load logits ----
logits_by_step = defaultdict(lambda: defaultdict(list))
how_many = 50  # limit for loading logits (per gruppo)
for fname in os.listdir(LOGITS_DIR):
    if not fname.endswith(".npy"):
        continue

    is_real, is_fake, is_anchors, step, faketype = parse_filename(fname)
    if step is None:
        step = "unknown"  # fallback to avoid None keys

    fullpath = os.path.join(LOGITS_DIR, fname)
    try:
        logits = np.load(fullpath)  # shape (N, 2) expected
    except Exception as e:
        print(f"Skipping {fname}: cannot load ({e})")
        continue

    print(f"{fname}: shape={getattr(logits, 'shape', None)} -> parsed: real={is_real} fake={is_fake} anchors={is_anchors} step='{step}' faketype='{faketype}'")

    if is_real:
        logits_by_step[step]["real"].append(logits)
    elif is_fake:
        # per ogni step teniamo un dict di faketypes -> lista di arrays
        logits_by_step[step].setdefault("fake", defaultdict(list))
        fk = faketype or "unknown"
        logits_by_step[step]["fake"][fk].append(logits)
    elif is_anchors:
        logits_by_step[step].setdefault("anchors", [])
        logits_by_step[step]["anchors"].append(logits)

# ---- Scatterplot + SAVE PDF (primi how_many logits per gruppo) ----
for step, groups in logits_by_step.items():
    plt.figure(figsize=(8, 6))
    alpha_val = 0.5  # transparency
    plotted_labels = set()

    # Real (stars)
    if "real" in groups and len(groups["real"]) > 0:
        all_real = np.concatenate(groups["real"], axis=0)[:how_many]
        # all_real = softmax(all_real, axis=1)
        plt.scatter(all_real[:, 0], all_real[:, 1],
                    s=60, alpha=alpha_val, marker='*', label="real")
        plotted_labels.add("real")

    # Anchors (triangles)
    if "anchors" in groups and len(groups["anchors"]) > 0:
        all_anchors = np.concatenate(groups["anchors"], axis=0)[:how_many]
        # all_anchors = softmax(all_anchors, axis=1)
        plt.scatter(all_anchors[:, 0], all_anchors[:, 1],
                    s=60, alpha=alpha_val, marker='^', label="anchors")
        plotted_labels.add("anchors")

    # Fake (circles) - un label per faketype
    if "fake" in groups:
        for faketype, arr_list in groups["fake"].items():
            if not arr_list:
                continue
            all_fake = np.concatenate(arr_list, axis=0)[:how_many]
            # all_fake = softmax(all_fake, axis=1)
            label = f"fake ({faketype})"
            # evita doppio label identico nella legenda
            if label in plotted_labels:
                plt.scatter(all_fake[:, 0], all_fake[:, 1],
                            s=60, alpha=alpha_val, marker='o')
            else:
                plt.scatter(all_fake[:, 0], all_fake[:, 1],
                            s=60, alpha=alpha_val, marker='o', label=label)
                plotted_labels.add(label)

    plt.xlabel("Class 0")
    plt.ylabel("Class 1")
    plt.title(f"2D Logits Scatterplot - step {step}")
    plt.legend(markerscale=1.5)
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"logits_scatter_step_{step}.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved PDF: {out_path}")

import numpy as np
import matplotlib.pyplot as plt
import os

import os
import numpy as np
import matplotlib.pyplot as plt

STEPS_ORDER = [
    "stylegan1",
    "stylegan2",
    "sdv1_4",
    "stylegan3",
    "stylegan_xl",
    "sdv2_1",
]

OUTPUT_DIR = "logs/plots_pdf_train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ANCHORS_LOGITS_DIR = "anchros_logits"

distances = []
step_pairs = []

for i in range(1, len(STEPS_ORDER)):
    curr = STEPS_ORDER[i]
    prev = STEPS_ORDER[i - 1]

    anchors_curr = np.load(os.path.join(
        ANCHORS_LOGITS_DIR, f"step_{curr}_1000.npy"
    ))
    anchors_prev = np.load(os.path.join(
        ANCHORS_LOGITS_DIR, f"step_{prev}_1000.npy"
    ))

    dist = np.mean((anchors_curr - anchors_prev))
    distances.append(dist)
    step_pairs.append(f"{prev} → {curr}")

    print(f"{prev} → {curr}: {dist:.4f}")
plt.figure(figsize=(10, 5))

plt.plot(
    step_pairs,
    distances,
    marker="o",
    linestyle="-",
)

plt.ylabel("Mean L2 distance between anchor logits")
plt.xlabel("Generator transition")
plt.title("Anchor drift across generator steps")
plt.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "anchor_drift_across_steps.pdf")
plt.savefig(out_path)
plt.close()

print(f"Saved plot: {out_path}")



    

    
