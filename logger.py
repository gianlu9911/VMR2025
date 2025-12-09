import numpy as np
import pandas as pd
def logging(csv_path="logs_fd/sequential_fd_results.csv"):
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
    return ACC, BWT, FWT
