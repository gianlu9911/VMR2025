import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# === Project imports ===
# Assicurati che questi file esistano nel tuo progetto
from config import PRETRAINED_MODELS, IMAGE_DIR
from src.g_dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model
from src.g_utils import evaluate3


def train_one_epoch_ucir(model, teacher, dataloader, criterion, optimizer, device,
                         lambda_new=1.0, lambda_dist=1.0, scaler=None): # Aggiunto scaler
    model.train()
    running_loss, running_acc, num_samples = 0.0, 0.0, 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # AMP: forward pass con autocast
        with torch.cuda.amp.autocast():
            # 1. Forward sul task corrente
            outputs = model(imgs)
            loss_new = criterion(outputs, labels)

            # 2. Distillation Loss
            loss_dist = 0.0
            if teacher is not None:
                T = 2.0
                with torch.no_grad():
                    # Anche il teacher può beneficiare di autocast per velocità
                    logits_teacher = teacher(imgs)
                    soft_targets = torch.softmax(logits_teacher / T, dim=1)
                
                loss_dist = nn.KLDivLoss(reduction='batchmean')(
                    torch.log_softmax(outputs / T, dim=1),
                    soft_targets
                ) * (T * T)

            loss_total = lambda_new * loss_new + lambda_dist * loss_dist 

        # AMP: backward pass scalato
        if scaler is not None:
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            optimizer.step()

        # Calcolo metriche (rimane uguale)
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean().item()
        batch_size = labels.size(0)
        running_loss += loss_total.item() * batch_size
        running_acc += acc * batch_size
        num_samples += batch_size

    return running_loss / num_samples, running_acc / num_samples

def train_one_epoch_ucir_no_ampe(model, teacher, dataloader, criterion, optimizer, device,
                         lambda_new=1.0, lambda_dist=1.0):
    model.train()
    running_loss, running_acc, num_samples = 0.0, 0.0, 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        # 1. Forward sul task corrente
        outputs = model(imgs)
        loss_new = criterion(outputs, labels)

        # 2. Distillation Loss (Knowledge Distillation)
        # Il teacher (modello precedente) guida il modello attuale a non dimenticare
        loss_dist = 0.0
        if teacher is not None:
            T = 2.0  # Temperature
            with torch.no_grad():
                logits_teacher = teacher(imgs)
                soft_targets = torch.softmax(logits_teacher / T, dim=1)
            
            loss_dist = nn.KLDivLoss(reduction='batchmean')(
                torch.log_softmax(outputs / T, dim=1),
                soft_targets
            ) * (T * T)

        # 3. Loss Totale
        loss_total = lambda_new * loss_new + lambda_dist * loss_dist 
        loss_total.backward()
        optimizer.step()

        # Calcolo metriche batch
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean().item()
        batch_size = labels.size(0)
        running_loss += loss_total.item() * batch_size
        running_acc += acc * batch_size
        num_samples += batch_size

    return running_loss / num_samples, running_acc / num_samples

def fine_tune(
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = '0',
    epochs: int = 1,
    lr: float = 1e-4,
    seed: int = 42,
    num_train_samples = 100,
    fine_tuning_on: str = 'stylegan2',
    backbone: str = 'stylegan1',
    checkpoint_file: str = "checkpoint/ckpt_seq.pth", # File condiviso per passare i pesi
    order: str = '',
    lambda_new: float = 1.0,
    lambda_dist: float = 1.0,
):
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> Training Task: {fine_tuning_on} (Backbone arch: {backbone})")

    logs_dir = "./logs_ucir"
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 1. CARICAMENTO MODELLO (Logica Sequenziale) ---
    # Iniziamo caricando l'architettura base
    classifier = load_pretrained_model(PRETRAINED_MODELS[backbone]).to(device)
    teacher = None

    if os.path.exists(checkpoint_file):
        print(f"--> Caricamento pesi dal task precedente: {checkpoint_file}")
        # Carica lo stato del modello addestrato al passo precedente
        ckpt = torch.load(checkpoint_file, map_location=device)
        classifier.load_state_dict(ckpt['state_dict'])
        
        # Il teacher è una copia ESATTA del modello PRIMA di iniziare questo task
        teacher = load_pretrained_model(PRETRAINED_MODELS[backbone]).to(device)
        teacher.load_state_dict(ckpt['state_dict'])
        teacher.eval()
        for p in teacher.parameters(): 
            p.requires_grad = False
        print("--> Teacher inizializzato per Distillation")
    else:
        print(f"--> Nessun checkpoint trovato. Inizio dal Backbone puro: {backbone}")

    # --- 2. DATASET E TRAINING ---
    dataset = RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR[fine_tuning_on], num_training_samples=num_train_samples)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_ucir(
            classifier, teacher, train_loader, criterion, optimizer, device,
            lambda_new, lambda_dist, scaler
        )
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    # --- 3. SALVATAGGIO CHECKPOINT PER IL PROSSIMO TASK ---
    # Sovrascriviamo il file per passarlo al prossimo step del ciclo
    torch.save({'state_dict': classifier.state_dict()}, checkpoint_file)
    print(f"Modello salvato in {checkpoint_file} per il prossimo task.")

    # --- 4. VALUTAZIONE SU TUTTI I TASK DELL'ORDINE ---
    # Parsiamo la stringa dell'ordine per avere la lista corretta
    order_list = [t.strip() for t in order.replace('[','').replace(']','').split(',')]
    test_results = {}
    
    print("Avvio valutazione sui test set...")
    for test_task in order_list:
        test_dataset = RealSynthethicDataloader(IMAGE_DIR['real'], IMAGE_DIR[test_task], split='test_set', num_training_samples=num_train_samples)
        # Importante: shuffle=False per test
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        _, acc, _, _ = evaluate3(classifier, test_loader, criterion, device, 
                                 test_name=test_task, save_dir=logs_dir, 
                                 task_name=fine_tuning_on, fake_type=test_task)
        test_results[test_task] = acc

    # --- 5. SALVATAGGIO CSV SPECIFICO PER QUESTO ORDINE ---
    # Creiamo un nome file univoco basato sull'ordine
    csv_filename = f"results_{order.replace(',', '_')}.csv"
    eval_csv_path = os.path.join(logs_dir, csv_filename)
    
    # Intestazione colonne: Task Corrente + Lista Task Test
    csv_headers = ["fine_tuning_on"] + order_list
    
    file_exists = os.path.isfile(eval_csv_path)
    with open(eval_csv_path, 'a') as f:
        # Scrivi header solo se il file è nuovo o vuoto
        if not file_exists or os.path.getsize(eval_csv_path) == 0:
            f.write(','.join(csv_headers) + '\n')

        # Costruisci la riga dati rispettando l'ordine delle colonne
        row = [fine_tuning_on]
        for task_col in order_list:
            acc_val = test_results.get(task_col, 0.0)
            row.append(f"{acc_val:.5f}")
        f.write(','.join(row) + '\n')

    

    return test_results

# ---------------------------------------------
# Main CLI
# ---------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=5) # Default ragionevole
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_samples', type=int, default=None)
    parser.add_argument('--lambda_new', type=float, default=1.0)
    parser.add_argument('--lambda_dist', type=float, default=0.5)
    args = parser.parse_args()

    # LISTA DEGLI ORDINI DA ESEGUIRE
    orders_to_run = [
        #"stylegan1,stylegan2,sdv1_4,stylegan3,stylegan_xl,sdv2_1",
        #"stylegan1,stylegan2,stylegan3,stylegan_xl,sdv1_4,sdv2_1",
        "sdv1_4,sdv2_1,stylegan1,stylegan2,stylegan3,stylegan_xl",
        "stylegan2,stylegan3,sdv2_1,stylegan1,stylegan_xl,sdv1_4"
    ]
    for current_order_str in orders_to_run:
        print("\n" + "="*60)
        print(f"NUOVA SEQUENZA DI TRAINING: {current_order_str}")
        print("="*60)

        tasks = [t.strip() for t in current_order_str.split(",") if t.strip()]
        
        # 1. Definizione Backbone: È sempre il PRIMO task dell'ordine corrente
        current_backbone = tasks[0]
        
        # 2. Gestione Checkpoint Sequenziale
        # Creiamo un percorso univoco per questa sequenza per evitare conflitti
        seq_id = current_order_str.replace(",", "_")
        checkpoint_path = f"checkpoint/ckpt_ucir_{seq_id}.pth"
        
        # RESET: Se esiste un file vecchio per questo ordine, lo cancelliamo 
        # per assicurarci di partire da zero (dal backbone puro)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"Checkpoint precedente rimosso. Inizio pulito.")

        # 3. Loop sui Task
        for i, task in enumerate(tasks):
            print(f"\n--- Step {i+1}/{len(tasks)}: Fine-tuning su {task} ---")
            
            # Chiamata principale
            # Nota: 'checkpoint_file' viene passato sia per caricare (se esiste) che per salvare
            fine_tune(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
                num_train_samples=args.num_train_samples,
                
                fine_tuning_on=task,           # Task attuale su cui addestrare
                backbone=current_backbone,     # Architettura base (definita dal primo task)
                checkpoint_file=checkpoint_path, # File "staffetta" per i pesi
                order=current_order_str,       # Ordine completo per il CSV
                
                lambda_new=args.lambda_new,
                lambda_dist=args.lambda_dist
            )

    print("\n\nTUTTI GLI ORDINI COMPLETATI.")