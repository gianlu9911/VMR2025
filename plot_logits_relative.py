import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

# --- CONFIGURAZIONE ---
logits_dir = "logs/logits"
output_dir = "plots_balanced_final"
os.makedirs(output_dir, exist_ok=True)

# 1. Carica il CSV e gestisci duplicati/tipi
acc_df = pd.read_csv("logs/test_accuracies.csv", index_col='fine_tuning_on')
acc_df.index = acc_df.index.str.strip()
acc_df.columns = acc_df.columns.str.strip()

steps = ['stylegan1', 'stylegan2', 'sdv1_4', 'stylegan3', 'stylegan_xl', 'sdv2_1']
style_map = {
    'stylegan1':   {'color': '#1f77b4', 'marker': 'v'},
    'stylegan2':   {'color': '#ff7f0e', 'marker': '^'},
    'sdv1_4':      {'color': '#2ca02c', 'marker': 's'},
    'stylegan3':   {'color': '#d62728', 'marker': 'D'},
    'stylegan_xl': {'color': '#9467bd', 'marker': 'P'},
    'sdv2_1':      {'color': '#8c564b', 'marker': '*'},
}

samples_total = 10 

def get_balanced_samples(data, is_real, accuracy, n_total, name_debug):
    """Seleziona campioni rispettando la proporzione dell'accuratezza e stampa il log."""
    # Assicuriamoci che accuracy sia un float singolo (gestione Series se ci sono duplicati)
    if isinstance(accuracy, pd.Series):
        accuracy = accuracy.iloc[0]
    accuracy = float(accuracy)

    if is_real:
        correct_mask = data[:, 0] > data[:, 1] # Real: Score0 > Score1
    else:
        correct_mask = data[:, 1] > data[:, 0] # Fake: Score1 > Score0
    
    idx_correct = np.where(correct_mask)[0]
    idx_wrong = np.where(~correct_mask)[0]
    
    # Calcolo quanti dovrebbero essere corretti/sbagliati secondo l'accuratezza
    n_correct_target = int(round(n_total * accuracy))
    n_wrong_target = n_total - n_correct_target
    
    # Selezione effettiva (limitata dalla disponibilità reale nel file .npy)
    n_correct_final = min(len(idx_correct), n_correct_target)
    n_wrong_final = min(len(idx_wrong), n_wrong_target)
    
    sel_correct = idx_correct[:n_correct_final]
    sel_wrong = idx_wrong[:n_wrong_final]
    
    # Stampa del report richiesto
    type_str = "REAL" if is_real else f"FAKE ({name_debug})"
    print(f"      {type_str:15} | Acc: {accuracy:.2%} | Presi: {n_correct_final} OK, {n_wrong_final} ERR (Target: {n_correct_target}/{n_wrong_target})")

    return data[np.concatenate([sel_correct, sel_wrong])]

# --- CICLO PRINCIPALE ---
for current_step in steps:
    print(f"\n--- Elaborazione Step: {current_step} ---")
    plt.figure(figsize=(10, 8))
    
    # 1. Plot dei REAL
    real_filename = f"real_step_{current_step}.npy"
    real_path = os.path.join(logits_dir, real_filename)
    
    if os.path.exists(real_path):
        real_data = np.load(real_path)
        # Assumiamo i Real quasi sempre corretti (95%) per questo plot
        real_plot = get_balanced_samples(real_data, True, 0.95, samples_total, "REAL")
        plt.scatter(real_plot[:, 0], real_plot[:, 1], c='white', edgecolors='green', 
                    linewidths=2, label='REAL', s=100, zorder=10)

    # 2. Plot dei FAKE
    all_files = os.listdir(logits_dir)
    fake_prefix = f"fake_step_{current_step}_faketype_"
    fake_files = [f for f in all_files if f.startswith(fake_prefix)]
    
    for f_name in fake_files:
        faketype_name = f_name.replace(fake_prefix, "").replace(".npy", "")
        
        try:
            acc = acc_df.loc[current_step, faketype_name]
            fake_path = os.path.join(logits_dir, f_name)
            fake_data = np.load(fake_path)
            
            fake_plot = get_balanced_samples(fake_data, False, acc, samples_total, faketype_name)
            
            style = style_map.get(faketype_name, {'color': 'gray', 'marker': 'o'})
            edge_c = 'black' if faketype_name == current_step else 'none'
            
            plt.scatter(fake_plot[:, 0], fake_plot[:, 1], 
                        c=style['color'], marker=style['marker'], 
                        edgecolors=edge_c, linewidths=1.5,
                        label=f"{faketype_name} ({float(acc)*100:.1f}%)", alpha=0.7, s=70)
        except KeyError:
            print(f"      Salto {faketype_name}: riga/colonna non trovata nel CSV.")

    # Estetica finale
    lim = 6
    plt.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label='Decision Boundary')
    plt.title(f"Logits Analysis - Task: {current_step}\n(Campionamento proporzionale all'accuratezza)")
    plt.xlabel("Score Real (Logit 0)")
    plt.ylabel("Score Fake (Logit 1)")
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"balanced_plot_{current_step}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"      -> Grafico salvato.")

print("\nProcesso completato.") 