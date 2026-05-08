# strategy.py
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.supervised import Naive, LwF
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

from dataset import UnifiedRealFakeDataset
from plugin import MetricsCollectorPlugin, FeatureDistillationPlugin, PrototypeiCaRLPlugin, UCIRPlugin, CosineLayer, RelativePlugin, RelativeRepresentationLayer

def build_avalanche_scenario_strict(tasks_split_dict: dict, custom_order: list) -> tuple:
    train_datasets = []
    test_datasets = []
    task_order = []

    for task_name in custom_order:
        if task_name not in tasks_split_dict:
            print(f"⚠️ [WARNING] Task {task_name} non trovato!")
            continue
            
        split_paths = tasks_split_dict[task_name]
        train_ds = UnifiedRealFakeDataset(real_paths=split_paths['train_reals'], fake_paths=split_paths['train_fakes'])
        test_ds = UnifiedRealFakeDataset(real_paths=split_paths['test_reals'], fake_paths=split_paths['test_fakes'])
        
        train_datasets.append(train_ds)
        test_datasets.append(test_ds)
        task_order.append(task_name)

    scenario = dataset_benchmark(train_datasets=train_datasets, test_datasets=test_datasets)
    return scenario, task_order

def get_cl_strategy(args, model, optimizer, criterion, device):
    print(f"\n[INFO] Strategia CL: {args.method.upper()}")
    
    # Teniamo solo l'InteractiveLogger per vedere i progressi a terminale
    eval_plugin = EvaluationPlugin( 
        accuracy_metrics(minibatch=False, epoch=False, experience=True, stream=True),
        loggers=[InteractiveLogger()]
    )

    metrics_plugin = MetricsCollectorPlugin() 
    
    base_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'train_mb_size': args.batch_size, 
        'train_epochs': args.epochs,
        'eval_mb_size': args.batch_size,
        'device': device,
        'evaluator': eval_plugin, 
        'plugins': [metrics_plugin],
        'train_num_workers': args.num_workers,
        'eval_num_workers': args.num_workers
    }

    if args.method == 'finetuning':
        strategy = Naive(**base_kwargs)
    elif args.method == 'distillation':
        fd_plugin = FeatureDistillationPlugin(alpha=1.0, layer_name='resnet.fc') 
        base_kwargs['plugins'].append(fd_plugin)
        strategy = Naive(**base_kwargs)
    elif args.method == 'icarl':
        icarl_plugin = PrototypeiCaRLPlugin()
        base_kwargs['plugins'].append(icarl_plugin)
        strategy = LwF(alpha=1.0, temperature=2.0, **base_kwargs)
    elif args.method == 'ucir':
        # UCIR Memory-Free: Modifica dell'architettura + Less-Forget Constraint
        print("\n[INFO] UCIR: Sostituzione dell'ultimo layer con CosineLayer...")
        in_features = model.resnet.fc.in_features
        # Assumiamo 2 classi di output (Real / Fake) per il nostro dataset
        model.resnet.fc = CosineLayer(in_features, out_features=2) 
        
        ucir_plugin = UCIRPlugin(alpha=5.0, layer_name='resnet.fc')
        base_kwargs['plugins'].append(ucir_plugin)
        strategy = Naive(**base_kwargs)

    elif args.method == 'relative':
        print("\n[INFO] RELATIVE: Sostituzione dell'ultimo layer con RelativeRepresentationLayer...")
        
        # 1. Prendiamo la dimensione delle feature in uscita dalla tua resnet
        in_features = model.resnet.fc.in_features
        
        # 2. Sostituiamo il layer lineare con il nostro Layer Relativo
        model.resnet.fc = RelativeRepresentationLayer(in_features, num_anchors=args.num_anchors, out_features=2)
        model = model.to(device) # Rispingiamo il modello aggiornato su GPU
        
        # 3. Aggiungiamo il plugin che estrarrà le ancore e congelerà la rete
        # (NOTA: assicurati che real_label_id sia corretto. Di solito Fake=1, Real=0)
        rel_plugin = RelativePlugin(num_anchors=args.num_anchors, real_label_id=0) 
        base_kwargs['plugins'].append(rel_plugin)
        
        # 4. Usiamo la strategia Naive (Finetuning), perché il "trucco" 
        # anti-dimenticanza è già nell'architettura congelata!
        strategy = Naive(**base_kwargs)

    else:
        raise NotImplementedError(f"Il metodo {args.method} non è ancora implementato!")
    
    return strategy, metrics_plugin