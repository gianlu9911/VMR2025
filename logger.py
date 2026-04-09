import numpy as np
import pandas as pd
def logging(csv_path="logs_finetuning/sequential_finetune_results_sdv1_4_sdv2_1_stylegan1_stylegan2_stylegan3_stylegan_xl.csv"):
    # --- Carica la matrice dal CSV ---
    df = pd.read_csv(csv_path, index_col=0)

    # logs_finetuning/sequential_finetune_results.csv
    # logs/eval_stylegan1_ntrainall_nanchorsall.csv
    # logs_fd/sequential_fd_results.csv
    R = df.to_numpy()
    tasks = df.columns.tolist()
    T = R.shape[0]

    # --- Mean Accuracy (ACC) ---
    ACC = np.mean(R[T-1, :])

    # --- Backward Transfer (BWT) ---
    # misura quanto l'apprendimento di nuovi task ha influenzato i precedenti
    BWT = np.mean(R[T-1, :-1] - np.diag(R)[:-1])

    # --- Forward Transfer (FWT) ---
    # classica formula richiede baseline: R_bar_i = performance prima di vedere task i
    # se non hai baseline, puoi stimarla con la prima riga (dopo primo task)
    R_bar = R[0, :]  # baseline approssimata: performance prima di vedere i task futuri
    FWT = np.mean(R[np.arange(1, T), np.arange(1, T)] - R_bar[1:])

    # --- Stampa risultati ---
    print("📊 Continual Learning Metrics (from accuracy matrix):")
    print(f"Mean Accuracy (ACC): {ACC:.4f}")
    print(f"Backward Transfer (BWT): {BWT:.4f}")
    print(f"Forward Transfer (FWT): {FWT:.4f}")

    with open(csv_path, "a") as f:
        f.write(f"Acc:{ACC:.4f},BWT:{BWT:.4f},FWT:{FWT:.4f}\n")
    return ACC, BWT, FWT


import pandas as pd
import numpy as np

def calcola_metriche_cl(csv_path='logs_original_dataset/detailed_metrics_results.csv'):
    print(f"Leggendo i risultati da: {csv_path} ...\n")
    df = pd.read_csv(csv_path)
    
    # Raggruppiamo per singola Run (es. Run_1, Run_2...)
    runs = df['Run_ID'].unique()
    
    risultati_finali = []
    
    for run in runs:
        df_run = df[df['Run_ID'] == run]
        
        # Recuperiamo l'ordine cronologico dei task in questa specifica run
        tasks_in_order = df_run['Train_Task'].unique()
        T = len(tasks_in_order)
        
        if T <= 1:
            continue # Se c'è solo un task, non possiamo calcolare BWT e FWT
            
        # Costruiamo la matrice R (T x T)
        # R[i, j] = Accuratezza sul task j-esimo dopo aver addestrato sul task i-esimo
        R = np.zeros((T, T))
        for i, t_train in enumerate(tasks_in_order):
            for j, t_eval in enumerate(tasks_in_order):
                # Estraiamo l'accuracy per questa combinazione
                val = df_run[(df_run['Train_Task'] == t_train) & (df_run['Eval_Task'] == t_eval)]['Accuracy'].values
                if len(val) > 0:
                    R[i, j] = val[0]
                    
        # 1. Average Accuracy (ACC): Accuratezza media su tutti i task visti alla fine dell'addestramento
        # Corrisponde alla media dell'ultima riga della matrice
        acc = np.mean(R[-1, :])
        
        # 2. Backward Transfer (BWT): Quanto abbiamo dimenticato dei task vecchi?
        # BWT = (1 / T-1) * Somma_da_i=1_a_T-1 [ R(T, i) - R(i, i) ]
        bwt_vals = []
        for i in range(T - 1):
            # Accuratezza alla fine (R[-1, i]) meno l'accuratezza appena imparato il task (R[i, i])
            bwt_vals.append(R[-1, i] - R[i, i])
        bwt = np.mean(bwt_vals)
        
        # 3. Forward Transfer (FWT): Quanto imparare task precedenti aiuta su quelli futuri? (Zero-shot)
        # FWT = (1 / T-1) * Somma_da_j=2_a_T [ R(j-1, j) ]
        fwt_vals = []
        for j in range(1, T):
            # Accuratezza sul task j *prima* di addestrarci sopra (dopo aver visto j-1)
            fwt_vals.append(R[j-1, j]) 
        fwt = np.mean(fwt_vals)
        
        risultati_finali.append({
            'Run_ID': run,
            'Final_ACC': round(acc, 4),
            'BWT': round(bwt, 4),
            'FWT': round(fwt, 4)
        })
        
    # Mostriamo i risultati
    df_res = pd.DataFrame(risultati_finali)
    print("=== RISULTATI CONTINUAL LEARNING ===")
    print(df_res.to_string(index=False))
    
    # Salviamo in un nuovo file
    out_file = 'logs/CL_Transfer_Metrics.csv'
    df_res.to_csv(out_file, index=False)
    print(f"\nTabellone riassuntivo salvato in: {out_file}")

if __name__ == '__main__':
    logging("/andromeda/personal/ggiuliani/VMR2025/icarl_results_table_dogan.csv")
