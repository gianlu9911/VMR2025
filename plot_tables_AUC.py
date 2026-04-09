import pandas as pd
import numpy as np
import os

# Mappatura per accorciare i nomi dei dataset in tabella
NAME_MAP = {
    'stylegan1': 'SG1',
    'stylegan2': 'SG2',
    'stylegan3': 'SG3',
    'stylegan_xl': 'SG-XL',
    'sdv1_4': 'SDv1.4',
    'sdv2_1': 'SDv2.1',
    'fake_cycle_gan': 'CycleGAN',
    'fake_progan256': 'ProGAN256',
    'fake_progan1024': 'ProGAN1024',
    'fake_glow': 'Glow',
    'fake_stargan': 'StarGAN'
}

# Mappatura per i titoli delle run e per forzare l'ordine nel 2x2
RUN_TITLES = {
    'Run_1': 'Temporal Sequence',
    'Run_2': 'StyleGAN First',
    'Run_3': 'SDV First',
    'Run_4': 'Random Order'
}

def generate_2x2_latex_table(csv_path):
    if not os.path.exists(csv_path):
        print(f"Errore: Il file {csv_path} non esiste!")
        return

    df = pd.read_csv(csv_path)
    # Assicuriamoci di ciclare esattamente in questo ordine (da 1 a 4) per avere la griglia giusta
    runs = ['Run_1', 'Run_2', 'Run_3', 'Run_4']

    print("%%% --- INIZIO CODICE LATEX 2x2 --- %%%\n")
    
    # APERTURA DEL SUPER-BLOCCO
    print("\\begin{table*}[htbp]")
    print("\\centering")

    for i, run in enumerate(runs):
        if run not in df['Run_ID'].values:
            print(f"% [ATTENZIONE: Nessun dato trovato per {run}]")
            continue

        df_run = df[df['Run_ID'] == run]
        task_order = df_run['Train_Task'].unique()
        run_title = RUN_TITLES.get(run, run)

        # Creiamo DUE pivot: una per ACC e una per AUC
        pivot_acc = df_run.pivot(index='Train_Task', columns='Eval_Task', values='Accuracy')
        pivot_auc = df_run.pivot(index='Train_Task', columns='Eval_Task', values='AUC')

        # Riordiniamo entrambe cronologicamente
        valid_cols = [c for c in task_order if c in pivot_acc.columns]
        pivot_acc = pivot_acc.reindex(index=task_order, columns=valid_cols)
        pivot_auc = pivot_auc.reindex(index=task_order, columns=valid_cols)
        
        # --- CALCOLO METRICHE ACC E AUC ---
        R_acc = pivot_acc.values
        R_auc = pivot_auc.values
        T = len(task_order)
        
        acc_mean = np.nanmean(R_acc[-1, :]) if T > 0 else 0.0
        acc_bwt = np.nanmean([R_acc[-1, k] - R_acc[k, k] for k in range(T - 1)]) if T > 1 else 0.0
        acc_fwt = np.nanmean([R_acc[j-1, j] for j in range(1, T)]) if T > 1 else 0.0

        auc_mean = np.nanmean(R_auc[-1, :]) if T > 0 else 0.0
        auc_bwt = np.nanmean([R_auc[-1, k] - R_auc[k, k] for k in range(T - 1)]) if T > 1 else 0.0
        auc_fwt = np.nanmean([R_auc[j-1, j] for j in range(1, T)]) if T > 1 else 0.0

        # Rinominiamo indici e colonne
        pivot_acc.rename(index=NAME_MAP, columns=NAME_MAP, inplace=True)
        pivot_auc.rename(index=NAME_MAP, columns=NAME_MAP, inplace=True)

        # --- GENERAZIONE MINIPAGE LATEX ---
        num_cols = len(valid_cols)
        col_format = "l | " + " ".join(["c"] * num_cols)
        safe_label = run_title.lower().replace(' ', '_')

        print(f"\n% --- Sub-Table {i+1}: {run_title} ---")
        print(f"\\begin{{minipage}}[b]{{0.48\\linewidth}}")
        print("\\centering")
        print("\\resizebox{\\linewidth}{!}{%")
        print(f"\\subfloat[{run_title}\\label{{tab:sub_{safe_label}}}]{{")
        print(f"\\begin{{tabular}}{{{col_format}}}")
        print("\\toprule")

        # Intestazione con diagbox ridimensionato
        header_cols = [NAME_MAP.get(c, c.replace('_', '\\_')) for c in valid_cols]
        header_str = "\\diagbox[width=6.5em, height=2.5em, trim=l]{\\textbf{Train}}{\\textbf{Test}} & " + " & ".join(header_cols) + " \\\\"
        print(header_str)
        print("\\midrule")

        # Righe della matrice
        for idx in pivot_acc.index:
            row_vals = []
            for col in pivot_acc.columns:
                val_acc = pivot_acc.loc[idx, col]
                val_auc = pivot_auc.loc[idx, col]
                if pd.isna(val_acc) or pd.isna(val_auc):
                    row_vals.append("-")
                else:
                    row_vals.append(f"{val_acc:.3f} ({val_auc:.3f})")
            print(f"{idx} & " + " & ".join(row_vals) + " \\\\")

        # Inserimento metriche
        print("\\midrule")
        metric_str = (f"\\textbf{{ACC (AUC) -}} "
                      f"\\textbf{{Avg:}} {acc_mean:.3f} ({auc_mean:.3f}) \\quad "
                      f"\\textbf{{BWT:}} {acc_bwt:.3f} ({auc_bwt:.3f}) \\quad "
                      f"\\textbf{{FWT:}} {acc_fwt:.3f} ({auc_fwt:.3f})")
        print(f"\\multicolumn{{{num_cols + 1}}}{{c}}{{{metric_str}}} \\\\")

        print("\\bottomrule")
        print("\\end{tabular}}")
        print("}") # Chiude resizebox
        print("\\end{minipage}")

        # LOGICA DI SPAZIATURA PER IL 2x2
        if i == 0:
            print("\\hfill % Spazio orizzontale tra colonna 1 e colonna 2")
        elif i == 1:
            print("\\vspace{0.5cm} % Spazio verticale tra riga 1 e riga 2")
        elif i == 2:
            print("\\hfill % Spazio orizzontale tra colonna 1 e colonna 2")

    # CHIUSURA DEL SUPER-BLOCCO
    print("\n\\caption{Performance comparison across different Continual Learning curriculum orders. Values are reported as \\textbf{Accuracy (AUC)}.}")
    print("\\label{tab:curriculum_orders_2x2}")
    print("\\end{table*}")
    
    print("\n%%% --- FINE CODICE LATEX 2x2 --- %%%")

if __name__ == "__main__":
    generate_2x2_latex_table('logs_original_dataset/detailed_metrics_results.csv')