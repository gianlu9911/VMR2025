import numpy as np
import pandas as pd
import os
import re

def process_all_steps_finetune_fd(base_dir, output_csv="fd_scatter_data.csv", n_samples=100):
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

#if __name__ == "__main__":
    # Inserisci qui il path della tua cartella logs
    #LOGS_DIR = 'logs_fd'
    #process_all_steps_finetune_fd(LOGS_DIR)

#!/usr/bin/env python3
#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

# --- Utils di calcolo ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def find_npy_files(base_dir):
    files = []
    for root, _, filenames in os.walk(base_dir):
        for f in filenames:
            if f.lower().endswith('.npy'):
                files.append(os.path.join(root, f))
    return files
def parse_npy_name(fname):
    name = os.path.basename(fname).replace('.npy', '').lower()
    
    # Gestione REAL: cattura tutto dopo 'real_step_'
    if name.startswith('real_step_'):
        step = name.replace('real_step_', '')
        return step, 'real', None

    # Gestione FAKE: usiamo una regex che cerca il blocco centrale '_faketype_'
    # (.+) è "greedy" e prenderà tutto fino all'ultimo match utile
    m = re.match(r'fake_step_(.+)_faketype_(.+)', name)
    if m:
        step, faketype = m.groups()
        return step, 'fake', faketype

    return None, None, None

def load_logits(path):
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def safe_sample(data, n, seed=42):
    if len(data) == 0: return data
    return data.sample(n=min(len(data), n), random_state=seed)

# --- Core ---

def build_dataframe(real_logits, fake_logits, step_id, faketype):
    real_n = real_logits.shape[0]
    fake_n = fake_logits.shape[0]
    
    all_logits = np.concatenate([real_logits, fake_logits], axis=0)
    labels = np.concatenate([np.zeros(real_n, dtype=int), np.ones(fake_n, dtype=int)])

    # Calcolo Probabilità e Predizioni
    if all_logits.shape[1] >= 2:
        # Caso Multi-class o 2-way logits
        probs = softmax(all_logits)
        prob_real = probs[:, 0]
        prob_fake = probs[:, 1]
        preds = np.argmax(all_logits, axis=1)
        x = all_logits[:, 0]
        y = all_logits[:, 1]
    else:
        # Caso Binary (1 logit)
        p_fake = sigmoid(all_logits[:, 0])
        prob_fake = p_fake
        prob_real = 1.0 - p_fake
        preds = (prob_fake > 0.5).astype(int)
        x = all_logits[:, 0]
        y = np.zeros_like(x) # Se non c'è Y, usiamo 0

    df = pd.DataFrame({
        'label': labels,
        'pred': preds,
        'prob_real': prob_real,
        'prob_fake': prob_fake,
        'x': x,
        'y': y,
        'step': f"step{step_id}",
        'test_set': f"real_vs_{faketype}" if faketype else f"real_step_{step_id}"
    })

    return df

def process_all_steps_from_npy(base_dir, output_csv, n_samples):
    npy_files = find_npy_files(base_dir)
    steps = defaultdict(lambda: {'real': None, 'fake': {}})

    for f in npy_files:
        step, kind, faketype = parse_npy_name(f)
        if step is None: continue
        if kind == 'real': steps[step]['real'] = f
        else: steps[step]['fake'][faketype] = f

    all_sampled = []

    for step_id, data in sorted(steps.items()):
        if data['real'] is None: continue
        real_logits = load_logits(data['real'])

        for faketype, fake_path in data['fake'].items():
            fake_logits = load_logits(fake_path)
            df_all = build_dataframe(real_logits, fake_logits, step_id, faketype)

            # Categorizzazione per il campionamento
            conditions = [
                (df_all.label == 0) & (df_all.pred == 0),
                (df_all.label == 0) & (df_all.pred == 1),
                (df_all.label == 1) & (df_all.pred == 1),
                (df_all.label == 1) & (df_all.pred == 0)
            ]
            choices = ['Real Correct', 'Real Wrong (FP)', 'Fake Correct', 'Fake Wrong (FN)']
            df_all['category'] = np.select(conditions, choices, default='unknown')

            # Campionamento bilanciato per categoria
            for cat in choices:
                subset = df_all[df_all.category == cat]
                all_sampled.append(safe_sample(subset, n_samples))

    if not all_sampled:
        print("❌ Nessun dato trovato.")
        return

    final_df = pd.concat(all_sampled, ignore_index=True)
    
    # Creazione della colonna IDX come nel tuo esempio (indice progressivo)
    final_df.index.name = 'idx'
    final_df = final_df.reset_index()

    # Riordino colonne per matchare la richiesta
    cols = ['idx', 'label', 'pred', 'prob_real', 'prob_fake', 'x', 'y', 'step', 'test_set', 'category']
    final_df = final_df[cols]

    final_df.to_csv(output_csv, index=False)
    print(f"\n✅ CSV creato con successo: {output_csv}")
    print(final_df.head())

#if __name__ == "__main__":
    # Parametri hardcoded o via argparse
    #BASE_DIR = 'logs_ucir/logits'
    #OUTPUT_CSV = 'ucir_scatter_data_from_npy.csv'
    #N_SAMPLES = 100 
    
    #process_all_steps_from_npy(BASE_DIR, OUTPUT_CSV, N_SAMPLES)

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def process_icarl_logits(base_dir, output_csv, n_samples=100):
    all_data = []
    
    # 1. Trova tutti i file di logits
    logits_files = [f for f in os.listdir(base_dir) if f.startswith('logits_') and f.endswith('.npy')]
    
    if not logits_files:
        print(f"❌ Nessun file trovato in {base_dir}")
        return

    print(f"🔍 Trovati {len(logits_files)} coppie di file. Inizio elaborazione...")

    for log_file in tqdm(logits_files):
        # Estrai step e task dal nome: logits_step1_sdv1_4.npy
        # Pattern: logits_(step\d+)_(.+).npy
        match = re.match(r'logits_(step\d+)_([^\.]+)\.npy', log_file)
        if not match:
            continue
            
        step_id, task_id = match.groups()
        
        # Percorsi file
        path_logits = os.path.join(base_dir, log_file)
        path_labels = os.path.join(base_dir, log_file.replace('logits_', 'labels_'))
        
        if not os.path.exists(path_labels):
            print(f"⚠️ Label mancanti per {log_file}, salto.")
            continue
            
        # 2. Caricamento dati
        logits = np.load(path_logits)
        labels = np.load(path_labels).flatten()
        
        # Calcolo probabilità e predizioni
        probs = softmax(logits)
        prob_real = probs[:, 0]
        prob_fake = probs[:, 1]
        preds = np.argmax(logits, axis=1)
        
        # Coordinate (Logits grezzi)
        x = logits[:, 0]
        y = logits[:, 1]
        
        # 3. Creazione DataFrame temporaneo per questo file
        df_temp = pd.DataFrame({
            'label': labels,
            'pred': preds,
            'prob_real': prob_real,
            'prob_fake': prob_fake,
            'x': x,
            'y': y,
            'step': step_id,
            'test_set': task_id
        })
        
        # 4. Assegnazione categorie
        conditions = [
            (df_temp.label == 0) & (df_temp.pred == 0),
            (df_temp.label == 0) & (df_temp.pred == 1),
            (df_temp.label == 1) & (df_temp.pred == 1),
            (df_temp.label == 1) & (df_temp.pred == 0)
        ]
        choices = ['Real Correct', 'Real Wrong (FP)', 'Fake Correct', 'Fake Wrong (FN)']
        df_temp['category'] = np.select(conditions, choices, default='unknown')
        
        # 5. Campionamento (come richiesto, n_samples per categoria per ogni test set)
        for cat in choices:
            subset = df_temp[df_temp.category == cat]
            if len(subset) > 0:
                sampled_subset = subset.sample(n=min(len(subset), n_samples), random_state=42)
                all_data.append(sampled_subset)

    # 6. Finalizzazione
    if not all_data:
        print("❌ Nessun dato raccolto. Controlla i file .npy")
        return

    final_df = pd.concat(all_data).reset_index(drop=True)
    final_df.index.name = 'idx'
    final_df = final_df.reset_index()
    
    # Riordino colonne esatto come richiesto
    cols = ['idx', 'label', 'pred', 'prob_real', 'prob_fake', 'x', 'y', 'step', 'test_set', 'category']
    final_df = final_df[cols]
    
    final_df.to_csv(output_csv, index=False)
    print(f"\n✅ CSV creato: {output_csv}")
    print(f"📊 Totale righe: {len(final_df)}")
    print(final_df.head())

#if __name__ == "__main__":
#    process_icarl_logits(
#        base_dir='icarl_logits', 
#        output_csv='icarl_analysis_results.csv', 
#        n_samples=100
#    )


import os
import re
import numpy as np
import pandas as pd

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def process_relative_logits(base_dir, output_csv, n_samples=100):
    # Lista tassativa dei task della tua sequenza originale per evitare la "nuova sequenza"
    valid_tasks = ['stylegan1', 'stylegan2', 'sdv1_4', 'stylegan3', 'stylegan_xl', 'sdv2_1']
    
    all_sampled = []
    
    print(f"🚀 Analisi selettiva in {base_dir}...")

    # 1. Mappatura dei file REAL validi
    # Cerchiamo: real_step_sdv1_4.npy
    real_files = {}
    for task in valid_tasks:
        path = os.path.join(base_dir, f"real_step_{task}.npy")
        if os.path.exists(path):
            real_files[task] = path

    # 2. Scansione dei file FAKE validi
    # Cerchiamo: fake_step_sdv2_1_faketype_stylegan2.npy
    for f in os.listdir(base_dir):
        if not f.startswith("fake_step_") or not f.endswith(".npy"):
            continue
            
        # Regex "ingorda" per gestire gli underscore nei nomi dei modelli
        match = re.match(r'fake_step_(.+)_faketype_(.+)\.npy', f)
        if not match:
            continue
            
        step_id, faketype = match.groups()
        
        # Filtro: se lo step_id non è nella nostra lista originale, lo ignoriamo (nuova sequenza)
        if step_id not in valid_tasks:
            continue
            
        # Percorsi
        path_fake = os.path.join(base_dir, f)
        path_real = real_files.get(step_id)
        
        if not path_real:
            # Se non abbiamo il Real per questo step, non possiamo fare il confronto bilanciato
            continue

        print(f"📦 Processando: Step {step_id} | FakeType {faketype}")

        # 3. Caricamento e Softmax
        l_real = np.load(path_real)
        l_fake = np.load(path_fake)
        
        # Assicuriamoci siano in formato (N, 2)
        if l_real.ndim == 1: l_real = l_real.reshape(-1, 1)
        if l_fake.ndim == 1: l_fake = l_fake.reshape(-1, 1)

        # Creazione dati concatenati
        logits = np.concatenate([l_real, l_fake], axis=0)
        labels = np.concatenate([np.zeros(len(l_real)), np.ones(len(l_fake))])
        
        # Probabilità e predizioni
        probs = softmax(logits)
        pr, pf = probs[:, 0], probs[:, 1]
        preds = np.argmax(logits, axis=1)

        # 4. DataFrame temporaneo
        df = pd.DataFrame({
            'label': labels.astype(int),
            'pred': preds,
            'prob_real': pr,
            'prob_fake': pf,
            'x': logits[:, 0],
            'y': logits[:, 1] if logits.shape[1] > 1 else np.zeros(len(logits)),
            'step': f"step_{step_id}",
            'test_set': f"real_vs_{faketype}"
        })

        # Categorizzazione
        conds = [
            (df.label == 0) & (df.pred == 0),
            (df.label == 0) & (df.pred == 1),
            (df.label == 1) & (df.pred == 1),
            (df.label == 1) & (df.pred == 0)
        ]
        choices = ['Real Correct', 'Real Wrong (FP)', 'Fake Correct', 'Fake Wrong (FN)']
        
        # AGGIUNTO default='unknown' per evitare il TypeError tra int e str
        df['category'] = np.select(conds, choices, default='unknown')
        # 5. Campionamento bilanciato
        for cat in df['category'].unique():
            subset = df[df.category == cat]
            all_sampled.append(subset.sample(n=min(len(subset), n_samples), random_state=42))

    # 6. Salvataggio finale
    if all_sampled:
        final_df = pd.concat(all_sampled).reset_index(drop=True)
        final_df.index.name = 'idx'
        final_df = final_df.reset_index()
        
        # Header richiesto
        cols = ['idx', 'label', 'pred', 'prob_real', 'prob_fake', 'x', 'y', 'step', 'test_set', 'category']
        final_df[cols].to_csv(output_csv, index=False)
        print(f"\n✅ Finito! Salvato in {output_csv} con {len(final_df)} righe.")
    else:
        print("❌ Nessun file corrispondente ai criteri trovato.")

if __name__ == "__main__":
    process_relative_logits(
        base_dir='logs/logits', 
        output_csv='relative_analysis_results.csv', 
        n_samples=100
    )