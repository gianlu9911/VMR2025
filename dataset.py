import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


def get_dataset_paths(dataset_group: str) -> tuple[list, dict]:
    """
    Estrae i path corretti dal file config in base al gruppo selezionato.
    
    Ritorna:
    - real_dirs (list): lista di tutte le cartelle che contengono immagini vere.
    - fake_tasks (dict): dizionario { 'nome_task': 'path_fake' }.
    """
    real_dirs = []
    fake_tasks = {}

    if dataset_group == 'ffhq':
        real_dirs.append(config.IMAGE_DIR['real'])
        for key, path in config.IMAGE_DIR.items():
            if key != 'real':
                fake_tasks[key] = path

    elif dataset_group == 'dogan_faces':
        for key, path in config.IMAGE_DIR_DOGAN_FACES.items():
            if key.startswith('real_'):
                real_dirs.append(path)
            elif key.startswith('fake_'):
                base_name = key.replace('fake_', '')
                fake_tasks[base_name] = path
                
        # Fake senza prefisso 'fake_' (es. cyclegan_facades)
        for key, path in config.IMAGE_DIR_DOGAN_FACES.items():
            if not key.startswith('real_') and not key.startswith('fake_'):
                fake_tasks[key] = path

    elif dataset_group == 'dogan_vair':
        for key, path in config.IMAGE_DIR_DOGAN_VAIR.items():
            if key.startswith('real_'):
                real_dirs.append(path)
            elif key.startswith('fake_'):
                base_name = key.replace('fake_', '')
                fake_tasks[base_name] = path

    return real_dirs, fake_tasks


def build_balanced_real_pool(real_dirs: list, total_samples: int) -> list:
    """
    Crea un pool bilanciato di immagini reali pescando in egual misura 
    da tutte le cartelle fornite tramite Round-Robin.
    """
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    dir_to_imgs = {}
    
    # 1. Raccoglie e mescola le immagini di ogni cartella
    for d in set(real_dirs):
        if not os.path.exists(d):
            continue
        imgs = [os.path.join(root, file) for root, _, files in os.walk(d) 
                for file in files if file.lower().endswith(valid_ext)]
        if imgs:
            random.shuffle(imgs) # Zero seed, pura entropia!
            dir_to_imgs[d] = imgs
            
    # 2. Estrazione Round-Robin (un po' da ciascuno)
    selected_paths = []
    active_dirs = list(dir_to_imgs.keys())
    idx_map = {d: 0 for d in active_dirs}
    
    while len(selected_paths) < total_samples and active_dirs:
        for d in list(active_dirs):
            if len(selected_paths) >= total_samples: 
                break
            
            if idx_map[d] < len(dir_to_imgs[d]):
                selected_paths.append(dir_to_imgs[d][idx_map[d]])
                idx_map[d] += 1
            else:
                active_dirs.remove(d)
                
    return selected_paths


def prepare_train_test_paths(real_dirs: list, fake_tasks: dict, num_train_samples: int, test_ratio: float = 0.2) -> dict:
    """
    Divide i path in Train e Test in modo assoluto prima di creare i tensori.
    Garantisce zero data leakage.
    """
    print("\n[INFO] Preparazione rigida dei Path (Train/Test Split)...")
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    
    num_test_samples = int(num_train_samples * test_ratio)
    total_samples_needed = num_train_samples + num_test_samples
    
    # 1. SPLIT DEL POOL REALE (Comune per tutti i task)
    global_reals = build_balanced_real_pool(real_dirs, total_samples_needed)
    
    if len(global_reals) < total_samples_needed:
        raise ValueError(f"[ERRORE] Servono {total_samples_needed} reali in totale, "
                         f"ma ne abbiamo solo {len(global_reals)}!")
                         
    real_train_paths = global_reals[:num_train_samples]
    real_test_paths = global_reals[num_train_samples:]
    
    # 2. SPLIT DEI TASK FAKE
    tasks_split_dict = {}
    
    for task_name, fake_dir in fake_tasks.items():
        all_fakes = [os.path.join(root, file) for root, _, files in os.walk(fake_dir) 
                     for file in files if file.lower().endswith(valid_ext)]
        random.shuffle(all_fakes)
        
        if len(all_fakes) < total_samples_needed:
             raise ValueError(f"[ERRORE] Il task {task_name} ha solo {len(all_fakes)} immagini. "
                              f"Ne servono {total_samples_needed} (Train + Test)!")
        
        fake_train_paths = all_fakes[:num_train_samples]
        fake_test_paths = all_fakes[num_train_samples:total_samples_needed]
        
        tasks_split_dict[task_name] = {
            'train_reals': real_train_paths,
            'train_fakes': fake_train_paths,
            'test_reals': real_test_paths,
            'test_fakes': fake_test_paths
        }
        
        print(f"  - {task_name:15} | Train Fakes: {len(fake_train_paths)} | Test Fakes: {len(fake_test_paths)}")
        
    print(f"  - REAL POOL (Fisso) | Train Reals: {len(real_train_paths)} | Test Reals: {len(real_test_paths)}")
    
    return tasks_split_dict


class UnifiedRealFakeDataset(Dataset):
    def __init__(self, real_paths: list, fake_paths: list, transform=None):
        """
        Prende liste di path ESATTE e blindate. Nessun mescolamento interno.
        """
        self.real_paths = real_paths
        self.fake_paths = fake_paths
        
        self.all_paths = self.real_paths + self.fake_paths
        self.targets = [0] * len(self.real_paths) + [1] * len(self.fake_paths)
        
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        img_path = self.all_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[WARNING] Fallita lettura di {img_path}: {e}")
            img = Image.new('RGB', (224, 224))
            
        return self.transform(img), self.targets[idx]