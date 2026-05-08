import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def compute_cl_metrics(matrix):
    """
    Calcola Average, BWT e FWT.
    Versione blindata: calcola i confini sicuri (limit) basandosi 
    sulla dimensione minima tra righe e colonne.
    """
    R = np.array(matrix)
    rows, cols = R.shape
    
    # Calcoliamo il limite massimo per non uscire mai dai bordi (es. max 6)
    limit = min(rows, cols)
    
    if limit <= 1:
        # Nessun transfer misurabile se abbiamo solo 1 step valido
        return np.nanmean(R[-1, :]), 0.0, 0.0

    # 1. Final Average (A_T): Media dell'ultima riga
    avg_final = np.nanmean(R[-1, :])

    # 2. Backward Transfer (BWT)
    # limit-1 assicura che i non superi mai le colonne esistenti
    bwt = np.nanmean([R[-1, i] - R[i, i] for i in range(limit - 1)])

    # 3. Forward Transfer (FWT)
    fwt = np.nanmean([R[j-1, j] for j in range(1, limit)])

    return avg_final, bwt, fwt


# ---> MODIFICA 1: Aggiunti final_probs e current_step_name alla firma
def export_all_results(save_dir, task_order, acc_matrix, auc_matrix, final_targets, final_preds, final_probs=None, current_step_name="FINAL"):
    """Genera CSV, Report e Heatmap delle Confusion Matrices."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. CALCOLO E SALVATAGGIO MATRICI IN CSV
    acc_avg, acc_bwt, acc_fwt = compute_cl_metrics(acc_matrix)
    auc_avg, auc_bwt, auc_fwt = compute_cl_metrics(auc_matrix)

    # BLINDATURA PANDAS
    row_labels_acc = [f"Eval_Step_{i+1}" for i in range(acc_matrix.shape[0])]
    row_labels_auc = [f"Eval_Step_{i+1}" for i in range(auc_matrix.shape[0])]

    df_acc = pd.DataFrame(acc_matrix, columns=task_order, index=row_labels_acc)
    df_auc = pd.DataFrame(auc_matrix, columns=task_order, index=row_labels_auc)

    df_acc.to_csv(os.path.join(save_dir, "custom_accuracy_matrix.csv"))
    df_auc.to_csv(os.path.join(save_dir, "custom_auc_matrix.csv"))

    # File di riepilogo metriche
    with open(os.path.join(save_dir, "custom_global_metrics_summary.txt"), "w") as f:
        f.write("=== METRICHE CUSTOM (Calcolate indipendentemente da Avalanche) ===\n\n")
        f.write(f"ACCURACY -> Mean(A_T): {acc_avg*100:.2f}% | BWT: {acc_bwt*100:.2f}% | FWT: {acc_fwt*100:.2f}%\n")
        f.write(f"AUC      -> Mean(A_T): {auc_avg*100:.2f}% | BWT: {auc_bwt*100:.2f}% | FWT: {auc_fwt*100:.2f}%\n")

    # 2. DEBUG REPORT (F1, RECALL, PRECISION)
    report_path = os.path.join(save_dir, "classification_debug_reports.txt")
    with open(report_path, "w") as f_rep:
        f_rep.write("=== FINAL EVALUATION CLASSIFICATION REPORTS ===\n\n")
        for i in range(min(len(task_order), len(final_targets))):
            task_name = task_order[i]
            y_true = final_targets[i]
            y_pred = final_preds[i]
            rep = classification_report(y_true, y_pred, target_names=["Real (0)", "Fake (1)"], zero_division=0)
            f_rep.write(f"--- TEST SU TASK: {task_name.upper()} ---\n")
            f_rep.write(rep)
            f_rep.write("\n" + "-"*50 + "\n\n")

    # 3. PLOT CONFUSION MATRICES E SALVATAGGIO RAW PREDICTIONS AVANZATO
    cm_dir = os.path.join(save_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    for i in range(min(len(task_order), len(final_targets))):
        task_name = task_order[i]
        y_true = final_targets[i]
        y_pred = final_preds[i]
        cm = confusion_matrix(y_true, y_pred)
        
        # ---> MODIFICA 2: Costruzione dinamica del dizionario per il CSV
        raw_data = {
            'idx': np.arange(len(y_true)),
            'Target': y_true,
            'Prediction': y_pred,
            'Correct': np.array(y_true) == np.array(y_pred),
            'Step': current_step_name,
            'Test_Set': task_name
        }
        
        # Se abbiamo passato le probabilità dal plugin, le estraiamo e le salviamo
        if final_probs is not None and i < len(final_probs):
            probs_array = np.array(final_probs[i])
            
            # Assumendo che l'output sia [batch_size, 2] (prob_real, prob_fake)
            if probs_array.ndim == 2 and probs_array.shape[1] >= 2:
                raw_data['prob_real'] = probs_array[:, 0]
                raw_data['prob_fake'] = probs_array[:, 1]
            elif probs_array.ndim == 1:
                # Caso in cui la rete sputi una singola probabilità (es. Sigmoide)
                raw_data['prob_fake'] = probs_array
                raw_data['prob_real'] = 1.0 - probs_array

        df_raw = pd.DataFrame(raw_data)
        
        # Aggiungiamo la colonna category per comodità testuale
        def get_category(row):
            if row['Target'] == 0 and row['Prediction'] == 0: return "Real Correct"
            if row['Target'] == 0 and row['Prediction'] == 1: return "Real Misclassified"
            if row['Target'] == 1 and row['Prediction'] == 1: return "Fake Correct"
            if row['Target'] == 1 and row['Prediction'] == 0: return "Fake Misclassified"
            return "Unknown"
            
        df_raw['category'] = df_raw.apply(get_category, axis=1)

        # Salvataggio CSV espanso
        df_raw.to_csv(os.path.join(cm_dir, f"raw_preds_{task_name}.csv"), index=False)
        
        # Plot della matrice
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
        plt.title(f"Confusion Matrix - Test on {task_name.upper()}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(cm_dir, f"cm_{task_name}.png"), dpi=150)
        plt.close()