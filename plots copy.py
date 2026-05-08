import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ---- CONFIGURAZIONE PERCORSI ----
BASE_PATH = "/andromeda/personal/ggiuliani/VMR2025/results/ffhq/relative"
OUTPUT_DIR = "logs/plots_probability_space"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STEPS_ORDER = ["stylegan1", "stylegan2", "sdv1_4", "stylegan3", "stylegan_xl", "sdv2_1"]

WARM_COLORS = {
    'stylegan1': '#e63946',   # Rosso acceso
    'stylegan2': '#f4a261',   # Arancio sabbia
    'sdv1_4': '#e9c46a',      # Oro/Giallo
    'stylegan3': '#d62828',   # Rosso scuro
    'stylegan_xl': '#f77f00', # Arancio scuro
    'sdv2_1': '#fcbf49',      # Giallo ocra
    'default': '#ffb703'      # Colore di sicurezza
}

MAX_POINTS_REAL = 80
MAX_POINTS_FAKE = 80
ALPHA_VAL = 0.5
JITTER_AMOUNT = 0.015 

# ---- LOOP DI GENERAZIONE ----
for step in STEPS_ORDER:
    step_folder = os.path.join(BASE_PATH, f"after_{step}", "confusion_matrices")
    
    if not os.path.exists(step_folder):
        print(f"[SKIP] Cartella non trovata: {step_folder}")
        continue
    
    print(f"\n--- Elaborazione Training Step: {step.upper()} ---")
    
    csv_files = glob.glob(os.path.join(step_folder, "raw_preds_*.csv"))
    if not csv_files:
        print(f"  [ATTENZIONE] Nessun CSV in {step_folder}")
        continue

    plt.figure(figsize=(11, 8))
    plotted_real = False 
    plotted_something = False

    for csv_file in csv_files:
        # Estrae il nome pulito dal file: raw_preds_sdv1_4.csv -> sdv1_4
        test_set_name = os.path.basename(csv_file).replace("raw_preds_", "").replace(".csv", "")
        
        df = pd.read_csv(csv_file)
        
        df['Correct'] = df['Correct'].astype(bool)
        acc_real = df[df['Target'] == 0]['Correct'].mean()
        acc_fake = df[df['Target'] == 1]['Correct'].mean()
        acc_real = 0 if pd.isna(acc_real) else acc_real
        acc_fake = 0 if pd.isna(acc_fake) else acc_fake

        # --- PLOT REAL (Target 0) ---
        real_data = df[df['Target'] == 0]
        n_real_to_plot = min(len(real_data), MAX_POINTS_REAL)
        
        # Plottiamo i Real una sola volta per step
        if not plotted_real and n_real_to_plot > 0:
            sample_real = real_data.sample(n=n_real_to_plot)
            
            j_x = np.random.normal(0, JITTER_AMOUNT, size=len(sample_real))
            j_y = np.random.normal(0, JITTER_AMOUNT, size=len(sample_real))
            
            plt.scatter(
                sample_real['prob_fake'] + j_x, sample_real['prob_real'] + j_y, 
                s=35, alpha=ALPHA_VAL, color='#2a9d8f', marker='D', 
                label=f'Real (Acc: {acc_real:.1%})', zorder=1
            )
            plotted_real = True
            plotted_something = True

        # --- PLOT FAKE (Target 1) ---
        fake_data = df[df['Target'] == 1]
        n_fake_to_plot = min(len(fake_data), MAX_POINTS_FAKE)
        
        if n_fake_to_plot > 0:
            # Ora prenderà il colore corretto dal dizionario!
            color = WARM_COLORS.get(test_set_name, WARM_COLORS['default'])
            sample_fake = fake_data.sample(n=n_fake_to_plot)
            
            j_x = np.random.normal(0, JITTER_AMOUNT, size=len(sample_fake))
            j_y = np.random.normal(0, JITTER_AMOUNT, size=len(sample_fake))
            
            plt.scatter(
                sample_fake['prob_fake'] + j_x, sample_fake['prob_real'] + j_y, 
                s=45, alpha=ALPHA_VAL, color=color, marker='o',
                edgecolors='white', linewidths=0.3,
                label=f'{test_set_name} (Acc: {acc_fake:.1%})', zorder=2
            )
            plotted_something = True

    if plotted_something:
        # Linea decisionale a 0.5
        plt.plot([-0.1, 1.1], [1.1, -0.1], color='gray', linestyle='--', alpha=0.5, label='Decision Boundary')

        plt.title(f"Probability Space | Training Step: {step.upper()}\n(Top-Left = High Real Confidence | Bottom-Right = High Fake Confidence)", fontsize=13)
        plt.xlabel("Network Confidence: FAKE (Target 1)")
        plt.ylabel("Network Confidence: REAL (Target 0)")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle=':', alpha=0.4)
        
        # Legenda esterna
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()

        plt.savefig(os.path.join(OUTPUT_DIR, f"step_{step}_probs.png"), dpi=300)
        plt.savefig(os.path.join(OUTPUT_DIR, f"step_{step}_probs.pdf"), bbox_inches='tight')
        print(f"  -> Grafico salvato per {step}")
    else:
        print(f"  -> [ATTENZIONE] Grafico vuoto per {step}")
        
    plt.close()

print(f"\n[FINE] Boom! Colori differenziati applicati con successo.")