import numpy as np
import pandas as pd
import os
import re

def process_all_steps(base_dir, output_csv="global_scatter_data.csv", n_samples=100):
    all_sampled_data = []

    # Cerchiamo tutti i file delle predizioni per identificare i vari test set e step
    # Il pattern cerca file che finiscono con _predictions.csv
    files = [f for f in os.listdir(base_dir) if f.endswith('_predictions.csv')]

    for csv_name in files:
        # Estraiamo step e test_set dal nome del file
        # Esempio: step1_real_vs_sdv1_4_predictions.csv
        prefix = csv_name.replace('_predictions.csv', '')
        
        # Tentativo di estrarre lo step (es. "step1")
        step_match = re.search(r'step\d+', prefix)
        step_label = step_match.group(0) if step_match else "unknown"
        
        # Il test set è tutto ciò che sta tra lo step e le predizioni
        test_set_label = prefix.replace(f"{step_label}_", "")

        print(f"Elaborazione: {step_label} | Test Set: {test_set_label}")

        # Costruiamo i percorsi per i file npy corrispondenti
        real_npy_path = os.path.join(base_dir, f"{prefix}_real_logits.npy")
        fake_npy_path = os.path.join(base_dir, f"{prefix}_fake_logits.npy")
        csv_preds_path = os.path.join(base_dir, csv_name)

        if not (os.path.exists(real_npy_path) and os.path.exists(fake_npy_path)):
            print(f"⚠️ Saltato: File .npy non trovati per {prefix}")
            continue

        # 1. Caricamento Logits
        logits_real = np.load(real_npy_path)
        logits_fake = np.load(fake_npy_path)
        all_points = np.concatenate([logits_real, logits_fake])

        # 2. Caricamento CSV e assegnazione coordinate
        df = pd.read_csv(csv_preds_path)
        
        # Protezione: verifichiamo che il numero di righe coincida
        if len(df) != len(all_points):
            print(f"❌ Errore: Lunghezza disallineata per {prefix}")
            continue

        df['x'] = all_points[:, 0]
        df['y'] = all_points[:, 1]
        df['step'] = step_label
        df['test_set'] = test_set_label

        # 3. Campionamento
        real_correct = df[(df['label'] == 0) & (df['pred'] == 0)]
        real_wrong   = df[(df['label'] == 0) & (df['pred'] == 1)]
        fake_correct = df[(df['label'] == 1) & (df['pred'] == 1)]
        fake_wrong   = df[(df['label'] == 1) & (df['pred'] == 0)]

        def safe_sample(data, n):
            return data.sample(n=min(len(data), n), random_state=42)

        sampled_df = pd.concat([
            safe_sample(real_correct, n_samples),
            safe_sample(real_wrong, n_samples),
            safe_sample(fake_correct, n_samples),
            safe_sample(fake_wrong, n_samples)
        ])

        # 4. Aggiunta categoria descrittiva
        conditions = [
            (sampled_df['label'] == 0) & (sampled_df['pred'] == 0),
            (sampled_df['label'] == 0) & (sampled_df['pred'] == 1),
            (sampled_df['label'] == 1) & (sampled_df['pred'] == 1),
            (sampled_df['label'] == 1) & (sampled_df['pred'] == 0)
        ]
        choices = ['Real Correct', 'Real Wrong (FP)', 'Fake Correct', 'Fake Wrong (FN)']
        sampled_df['category'] = np.select(conditions, choices, default='unknown')

        all_sampled_data.append(sampled_df)

    # Unione finale
    final_global_df = pd.concat(all_sampled_data, ignore_index=True)
    final_global_df.to_csv(output_csv, index=False)
    print(f"\n✅ Finito! Salvati {len(final_global_df)} punti totali in '{output_csv}'")

if __name__ == "__main__":
    # Inserisci qui il path della tua cartella logs
    LOGS_DIR = 'logs_finetuning_stylegan1_stylegan2_sdv1_4_stylegan3_stylegan_xl_sdv2_1'
    process_all_steps(LOGS_DIR)