import pandas as pd
import numpy as np
import os

# Mappatura per accorciare i nomi in tabella
NAME_MAP = {
    'stylegan1': 'SG1',
    'stylegan2': 'SG2',
    'stylegan3': 'SG3',
    'stylegan_xl': 'SG-XL',
    'sdv1_4': 'SDv1.4',
    'sdv2_1': 'SDv2.1'
}

def generate_latex_tables(csv_path):
    if not os.path.exists(csv_path):
        print(f"Errore: Il file {csv_path} non esiste!")
        return

    df = pd.read_csv(csv_path)

    # Trova tutte le run uniche (es. Run_1, Run_2, ecc.)
    runs = df['Run_ID'].unique()

    print("%%% --- INIZIO CODICE LATEX --- %%%\n")

    for run in runs:
        # Filtra i dati solo per la run corrente
        df_run = df[df['Run_ID'] == run]

        # Estrae l'ordine cronologico dei task (garantisce che la matrice sia ordinata)
        task_order = df_run['Train_Task'].unique()

        # Crea la matrice pivot: Righe = Train_Task, Colonne = Eval_Task, Valori = Accuracy
        pivot_df = df_run.pivot(index='Train_Task', columns='Eval_Task', values='Accuracy')

        # Riordina righe e colonne in base all'ordine cronologico
        valid_cols = [c for c in task_order if c in pivot_df.columns]
        pivot_df = pivot_df.reindex(index=task_order, columns=valid_cols)
        
        # --- CALCOLO METRICHE ACC, BWT, FWT ---
        # R è la nostra matrice numerica T x T
        R = pivot_df.values
        T = len(task_order)
        
        # ACC: Accuratezza media sull'ultima riga (dopo aver addestrato su tutto)
        acc = np.nanmean(R[-1, :]) if T > 0 else 0.0
        
        # BWT: Differenza tra l'accuratezza finale e l'accuratezza subito dopo aver imparato il task
        bwt = np.nanmean([R[-1, i] - R[i, i] for i in range(T - 1)]) if T > 1 else 0.0
        
        # FWT: Zero-shot accuracy (accuratezza sul task j testando il modello addestrato fino a j-1)
        fwt = np.nanmean([R[j-1, j] for j in range(1, T)]) if T > 1 else 0.0

        # Applica i nomi accorciati per il paper
        pivot_df.rename(index=NAME_MAP, columns=NAME_MAP, inplace=True)

        print(f"% ==========================================")
        print(f"% TABELLA PER: {run}")
        print(f"% ==========================================")

        num_cols = len(valid_cols)
        col_format = "l | " + " ".join(["c"] * num_cols)

        print(f"\\begin{{table}}[htbp]")
        print(f"\\centering")
        print(f"\\caption{{Accuracy results for {run}}}")
        print(f"\\begin{{tabular}}{{{col_format}}}")
        print(f"\\toprule")

        # Intestazione con diagbox per eleganza
        header_cols = [NAME_MAP.get(c, c.replace('_', '\\_')) for c in valid_cols]
        header_str = "\\diagbox[width=6.5em, height=2em, trim=l]{\\textbf{Train}}{\\textbf{Test}} & " + " & ".join(header_cols) + " \\\\"
        print(header_str)
        print(f"\\midrule")

        # Righe della matrice
        for idx, row in pivot_df.iterrows():
            row_vals = []
            for val in row:
                if pd.isna(val):
                    row_vals.append("-") # Mette un trattino se manca il dato
                else:
                    row_vals.append(f"{val:.4f}") # Formatta a 4 cifre decimali
            
            row_name = str(idx)
            print(f"{row_name} & " + " & ".join(row_vals) + " \\\\")

        # --- INSERIMENTO METRICHE IN FONDO ALLA TABELLA ---
        print(f"\\midrule")
        metric_str = f"\\textbf{{ACC:}} {acc:.4f} \\quad \\textbf{{BWT:}} {bwt:.4f} \\quad \\textbf{{FWT:}} {fwt:.4f}"
        print(f"\\multicolumn{{{num_cols + 1}}}{{c}}{{{metric_str}}} \\\\")

        print(f"\\bottomrule")
        print(f"\\end{{tabular}}")
        print(f"\\label{{tab:{run.lower()}}}")
        print(f"\\end{{table}}\n\n")

if __name__ == "__main__":
    # Assicurati che il path punti al tuo CSV generato in precedenza
    generate_latex_tables('logs_original_dataset/detailed_metrics_results.csv')