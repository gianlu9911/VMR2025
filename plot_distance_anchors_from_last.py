import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ============================================================
# CONFIGURAZIONE
# ============================================================
BASE_OUTPUT_DIR = "logs/plots_sequences"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

orders = [
    ['stylegan1', 'stylegan2', 'sdv1_4', 'stylegan3', 'stylegan_xl', 'sdv2_1'],
    ['sdv1_4', 'sdv2_1', 'stylegan1', 'stylegan2', 'stylegan3', 'stylegan_xl'],
    ['stylegan1', 'stylegan2', 'stylegan3', 'stylegan_xl', 'sdv1_4', 'sdv2_1'],
    ['stylegan2', 'sdv1_4', 'stylegan_xl', 'stylegan3', 'sdv2_1', 'stylegan1']
]

# ============================================================
# LOOP DI ELABORAZIONE
# ============================================================

for idx, current_order in enumerate(orders):
    # Costruisce il nome cartella esatto: anchros_logits_['item1', 'item2', ...]
    anchors_dir = f"anchros_logits_{current_order}"
    seq_id = idx + 1
    
    print(f"\n--- Elaborazione Sequenza {seq_id} ---")
    
    drifts_logits = []
    drifts_softmax = []
    valid_steps = []
    prev_mean_logits = None
    prev_probs = None

    for step in current_order:
        path = os.path.join(anchors_dir, f"step_{step}_100.npy")
        
        if not os.path.exists(path):
            print(f"[WARN] File mancante: {path}")
            continue

        anchors = np.load(path)
        curr_mean_logits = np.mean(anchors, axis=0)

        logits_tensor = torch.from_numpy(curr_mean_logits).float()
        curr_probs = F.softmax(logits_tensor, dim=0).numpy()

        if prev_mean_logits is not None:
            # Calcolo drift (L2 distance)
            drifts_logits.append(np.linalg.norm(curr_mean_logits - prev_mean_logits))
            drifts_softmax.append(np.linalg.norm(curr_probs - prev_probs))
            # Questo 'step' è il nome del secondo elemento del confronto
            valid_steps.append(step)

        prev_mean_logits = curr_mean_logits
        prev_probs = curr_probs

    # --- PLOTTING (VERSIONE COMPLETA) ---
    # Qui NON usiamo [1:], così partiamo dal primo drift disponibile (Step 2)
    if not drifts_logits:
        continue

    # Plot Logits
    plt.figure(figsize=(10, 5))
    plt.plot(valid_steps, drifts_logits, marker='o', color='black', linewidth=1.5)
    plt.title(f"Anchor Drift (Logits) - Sequence {seq_id}")
    plt.xlabel("Target Step (compared to previous)")
    plt.ylabel("L2 Distance")
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, f"seq{seq_id}_logits_complete.pdf"))
    plt.close()

    # Plot Softmax
    plt.figure(figsize=(10, 5))
    plt.plot(valid_steps, drifts_softmax, marker='s', color='blue', linewidth=1.5)
    plt.title(f"Anchor Drift (Softmax) - Sequence {seq_id}")
    plt.xlabel("Target Step (compared to previous)")
    plt.ylabel("L2 Distance (Probabilities)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, f"seq{seq_id}_softmax_complete.pdf"))
    plt.close()

print(f"\n[FINITO] Tutte le sequenze elaborate in: {BASE_OUTPUT_DIR}")