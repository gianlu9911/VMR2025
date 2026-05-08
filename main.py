# main.py
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- Import dai tuoi moduli custom ---
import config
from net import ResNet50BC, load_pretrained_model
from dataset import get_dataset_paths, prepare_train_test_paths
from strategy import build_avalanche_scenario_strict, get_cl_strategy

def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning Training Pipeline")
    
    # --- Selezione Dataset ---
    parser.add_argument('--dataset_group', type=str, default='ffhq',
                        choices=['ffhq', 'dogan_faces', 'dogan_vair'],
                        help='Scegli il macro-gruppo di dataset da utilizzare.')
    
    # --- Campionamento ---
    parser.add_argument('--num_samples', type=int, default=50000, 
                        help='Numero di campioni di training per classe (real/fake).')
    
    # --- Metodi e Iperparametri ---
    parser.add_argument('--method', type=str, default='relative',
                        choices=['finetuning', 'distillation', 'icarl','ucir', 'relative'], 
                        help='Tecnica di Continual Learning da applicare (Memory-Free).')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate per l\'ottimizzatore.')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Numero di epoche per ogni task.')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Dimensione del batch per il training e la valutazione.')
    parser.add_argument('--pretrained_weights', type=str, default='stylegan1',
                        help='Chiave per selezionare i pesi pre-addestrati da config.py. Lasciare vuoto per ImageNet.')
    parser.add_argument('--task_order', nargs='+', 
                        default=['progan256', 'progan1024', 'stargan'],
                        #default=['stylegan1', 'stylegan2', 'sdv1_4', 'stylegan3', 'stylegan_xl', 'sdv2_1'],
                        help='Lista dei task nell\'ordine esatto di training.')
    parser.add_argument('--num_anchors', type=int, default=10000,
                        help='Numero di anchor points per ogni task.')
    parser.add_argument('--num_workers', type=int, default=16, 
                        help='Numero di workers per i DataLoader di PyTorch.')

    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print(f"🚀 RUN PIPELINE: {args.method.upper()}")
    print(f"📁 DATASET GROUP: {args.dataset_group.upper()}")
    print(f"⚙️  TRAIN SAMPLES/CLASS: {args.num_samples} | LR: {args.lr} | BATCH: {args.batch_size}")
    print("="*60)

    # 1. Recupera le liste dei path grezzi
    real_dirs, fake_tasks = get_dataset_paths(args.dataset_group)
    if not fake_tasks or not real_dirs:
        raise ValueError(f"Dati mancanti per {args.dataset_group}. Controlla config.py")

    # 2. SPLIT RIGOROSO: Prepara i path di Train e Test
    tasks_split_dict = prepare_train_test_paths(
        real_dirs=real_dirs, 
        fake_tasks=fake_tasks, 
        num_train_samples=args.num_samples, 
        test_ratio=0.2
    )

    # 3. AVALANCHE: Crea lo scenario
    print("\n[INFO] Creazione dello Scenario Avalanche in corso...")
    # Nel main.py, riga dello scenario:
    scenario, task_order = build_avalanche_scenario_strict(tasks_split_dict, args.task_order)
    print(f"\n[SUCCESS] Scenario creato! Ordine dei task: {task_order}")

    # =====================================================================
    # 4. SETUP MODELLO E STRATEGIA
    # =====================================================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device attivo: {device}")

    # Inizializza ResNet50 custom
    if args.pretrained_weights and args.pretrained_weights.lower() != 'none':
        if args.pretrained_weights not in config.PRETRAINED_MODELS:
            raise ValueError(f"Chiave '{args.pretrained_weights}' non trovata in config.PRETRAINED_MODELS.")
        model = load_pretrained_model(config.PRETRAINED_MODELS[args.pretrained_weights])
        print(f"[INFO] Modello caricato con pesi: {args.pretrained_weights}")
    else:
        model = ResNet50BC()
        print("[INFO] Nessun peso specifico richiesto. Inizializzazione ResNet50BC (ImageNet base).")
        
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Chiamata SINGOLA alla factory per avere Strategia + Plugin
    strategy, metrics_plugin = get_cl_strategy(args, model, optimizer, criterion, device)

    # =====================================================================
    # 5. LOOP DI CONTINUAL LEARNING
    # =====================================================================
    print("\n" + "="*60)
    print(" 🚀 INIZIO TRAINING AVALANCHE (CON METRICHE NATIVE E CUSTOM)")
    print("="*60)

    print("\n[INFO] Esecuzione Valutazione Zero-Shot per il FWT di Avalanche...")
    strategy.eval(scenario.test_stream) # zero shot!

    from metrics import export_all_results 

    for exp_idx, experience in enumerate(scenario.train_stream):
        task_name = task_order[exp_idx]
        print(f"\n---> [TRAIN E EVAL] Task: {task_name.upper()}")
        
        # 1. ADDESTRAMENTO PURO (Niente eval automatica qui!)
        strategy.train(experience)

        # 2. VALUTAZIONE MANUALE (Una e una sola volta per task)
        strategy.eval(scenario.test_stream)

        # --- ESPORTAZIONE PARZIALE ---
        current_save_dir = os.path.join("results", args.dataset_group, args.method, f"after_{task_name}")
        
        temp_acc_matrix = np.array(metrics_plugin.acc_matrix)
        temp_auc_matrix = np.array(metrics_plugin.auc_matrix)

        # Mossa ninja: togliamo sempre la prima riga dello zero-shot
        if temp_acc_matrix.shape[0] > 1:
            temp_acc_matrix = temp_acc_matrix[1:, :]
            temp_auc_matrix = temp_auc_matrix[1:, :]

        export_all_results(
            save_dir=current_save_dir,
            task_order=task_order, 
            acc_matrix=temp_acc_matrix,
            auc_matrix=temp_auc_matrix,
            final_targets=metrics_plugin.last_eval_targets,
            final_preds=metrics_plugin.last_eval_preds,
            final_probs=metrics_plugin.last_eval_probas,
            current_step_name=f"step_{task_name}"
        )
    # =====================================================================
    # 6. REPORT FINALE E SALVATAGGIO
    # =====================================================================
    print("\n" + "="*60)
    print(" 🎉 TRAINING COMPLETATO! SALVATAGGIO RISULTATI FINALI...")
    print("="*60)

    final_acc_matrix = np.array(metrics_plugin.acc_matrix)
    final_auc_matrix = np.array(metrics_plugin.auc_matrix)
    
    # Mossa ninja finale
    if final_acc_matrix.shape[0] > len(task_order):
        final_acc_matrix = final_acc_matrix[1:, :]
        final_auc_matrix = final_auc_matrix[1:, :]
    
    final_save_dir = os.path.join("results", args.dataset_group, args.method, "FINAL_RESULTS")
    
    export_all_results(
        save_dir=final_save_dir,
        task_order=task_order,
        acc_matrix=final_acc_matrix,
        auc_matrix=final_auc_matrix,
        final_targets=metrics_plugin.last_eval_targets,
        final_preds=metrics_plugin.last_eval_preds,
        current_step_name=f"FINAL_RESULTS"
    )
    
    print(f"📁 Ibrido completato! Risultati finali in: '{final_save_dir}/'\n")
if __name__ == "__main__":
    main()