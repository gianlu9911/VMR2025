import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

# Nuova configurazione cartella
logits_dir = "logs_finetuning_stylegan1_stylegan2_sdv1_4_stylegan3_stylegan_xl_sdv2_1"
output_dir = "plots_finetuning_analysis"
os.makedirs(output_dir, exist_ok=True)

# Caricamento accuratezze (stesso file di prima)
acc_df = pd.read_csv("logs/test_accuracies.csv", index_col='fine_tuning_on')

# Mappa degli step (per associare 'step1' al nome corretto nel CSV)
step_map = {
    "step1": "stylegan1",
    "step2": "stylegan2",
    "step3": "sdv1_4",
    "step4": "stylegan3",
    "step5": "stylegan_xl",
    "step6": "sdv2_1"
}

style_map = {
    'stylegan1':   {'color': '#1f77b4', 'marker': 'v'},
    'stylegan2':   {'color': '#ff7f0e', 'marker': '^'},
    'sdv1_4':      {'color': '#2ca02c', 'marker': 's'},
    'stylegan3':   {'color': '#d62728', 'marker': 'D'},
    'stylegan_xl': {'color': '#9467bd', 'marker': 'P'},
    'sdv2_1':      {'color': '#8c564b', 'marker': '*'},
}

samples_per_type = 15

# Cicliamo sugli step (da step1 a step6)
for step_id, step_name_csv in step_map.items():
    plt.figure(figsize=(12, 8))
    
    # 1. Carica i REAL per questo step
    # Cerchiamo un file che inizi con lo step_id e finisca con _real_logits.npy
    all_files = os.listdir(logits_dir)
    real_file = [f for f in all_files if f.startswith(f"{step_id}_") and f.endswith("_real_logits.npy")]
    
    if real_file:
        real_data = np.load(os.path.join(logits_dir, real_file[0]))[:samples_per_type]
        plt.scatter(real_data[:, 0], real_data[:, 1], 
                    c='white', edgecolors='green', linewidths=2, label='REAL', 
                    alpha=0.9, s=100, zorder=10)

    # 2. Carica tutti i FAKE per questo step
    # Pattern: stepX_real_vs_FAKETYPE_fake_logits.npy
    fake_files = [f for f in all_files if f.startswith(f"{step_id}_") and "_fake_logits.npy" in f]
    
    found_fake = False
    for f_name in fake_files:
        # Estraiamo il faketype dal nome del file
        # Esempio: "step1_real_vs_sdv1_4_fake_logits.npy" -> "sdv1_4"
        parts = f_name.split("_vs_")
        if len(parts) > 1:
            faketype_part = parts[1].replace("_fake_logits.npy", "")
            
            fake_path = os.path.join(logits_dir, f_name)
            fake_data = np.load(fake_path)[:samples_per_type]
            
            style = style_map.get(faketype_part, {'color': 'gray', 'marker': 'o'})
            
            # Recupero accuratezza (usando i nomi mappati per il CSV)
            try:
                accuracy = acc_df.loc[step_name_csv, faketype_part]
                label_text = f"{faketype_part} ({accuracy*100:.1f}%)"
            except:
                label_text = faketype_part

            # Se il faketype è quello su cui stiamo facendo fine-tuning in questo step, bordo nero
            is_current_task = (faketype_part == step_name_csv)
            edge_c = 'black' if is_current_task else 'none'
            
            plt.scatter(fake_data[:, 0], fake_data[:, 1], 
                        c=style['color'], marker=style['marker'], 
                        edgecolors=edge_c, linewidths=1.5,
                        label=label_text, alpha=0.7, s=70, zorder=5)
            found_fake = True

    if found_fake:
        plt.title(f"Logits Analysis - {step_id.upper()} (Fine-tuning on: {step_name_csv})", fontsize=14)
        plt.xlabel("Real Score")
        plt.ylabel("Fake Score")
        
        # Assi bilanciati
        plt.axhline(0, color='grey', lw=1, alpha=0.3)
        plt.axvline(0, color='grey', lw=1, alpha=0.3)
        
        all_axes = plt.gca()
        x_lims, y_lims = all_axes.get_xlim(), all_axes.get_ylim()
        limit = max(max(abs(np.array(x_lims))), max(abs(np.array(y_lims))))
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
        
        plt.plot([-limit, limit], [-limit, limit], 'k--', alpha=0.2, label='Decision Boundary')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Faketype (Acc %)")
        plt.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"plot_{step_id}_{step_name_csv}.png")
        plt.savefig(save_path, dpi=200)
        print(f"Generato: {save_path}")

    plt.close()