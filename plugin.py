# plugin.py
from avalanche.training.plugins import SupervisedPlugin
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import numpy as np
import copy
import torch
import numpy as np
import torch.nn as nn
import math

class MetricsCollectorPlugin(SupervisedPlugin):
    """
    Spia Avalanche durante l'eval: colleziona predizioni e probabilità, 
    calcola Accuracy e AUC, e salva i dati grezzi per l'esportazione finale.
    """
    def __init__(self):
        super().__init__()
        # Matrici T x T
        self.auc_matrix = []
        self.acc_matrix = []
        self.current_auc_row = []
        self.current_acc_row = []

        # Variabili che mancavano per esportare le Matrici di Confusione!
        self.last_eval_targets = []
        self.last_eval_preds = []
        
        # --- NUOVO --- 
        # L'attributo che main.py sta cercando disperatamente
        self.last_eval_probas = [] 

        # Contenitori temporanei del singolo batch/task
        self.probs = []        # Serve per AUC (solo probabilità classe 1)
        self.full_probs = []   # --- NUOVO --- Serve per il CSV (probabilità classe 0 e 1)
        self.preds = []
        self.targets = []

    def before_eval(self, strategy, **kwargs):
        # Inizia il mega-loop di valutazione globale
        self.current_auc_row = []
        self.current_acc_row = []
        self.last_eval_targets = []
        self.last_eval_preds = []
        self.last_eval_probas = [] # --- NUOVO --- Resettiamo anche le probabilità

    def before_eval_exp(self, strategy, **kwargs):
        # Inizia il test su un singolo task
        self.probs = []
        self.full_probs = []       # --- NUOVO ---
        self.preds = []
        self.targets = []

    def after_eval_iteration(self, strategy, **kwargs):
        # Accumula i dati batch per batch
        
        # --- MODIFICATO --- 
        # Prima calcoliamo TUTTE le probabilità e le salviamo per il CSV
        all_batch_probs = F.softmax(strategy.mb_output, dim=1).detach().cpu().numpy()
        self.full_probs.extend(all_batch_probs)
        
        # Poi estraiamo solo la colonna 1 (Fake) per calcolare l'AUC come facevi prima
        batch_probs = all_batch_probs[:, 1]
        
        batch_preds = strategy.mb_output.argmax(dim=1).detach().cpu().numpy()
        batch_targets = strategy.mb_y.detach().cpu().numpy()

        self.probs.extend(batch_probs)
        self.preds.extend(batch_preds)
        self.targets.extend(batch_targets)

    def after_eval_exp(self, strategy, **kwargs):
        # Finita la valutazione di un task, calcoliamo i numeretti
        try:
            auc = roc_auc_score(self.targets, self.probs)
        except ValueError:
            auc = 0.5
        acc = np.mean(np.array(self.preds) == np.array(self.targets))

        self.current_auc_row.append(auc)
        self.current_acc_row.append(acc)

        # SALVIAMO target, predizioni e probabilità integrali da passare al main.py
        self.last_eval_targets.append(np.array(self.targets))
        self.last_eval_preds.append(np.array(self.preds))
        self.last_eval_probas.append(np.array(self.full_probs)) # --- NUOVO ---

    def after_eval(self, strategy, **kwargs):
        # IL BUTTAFUORI: Accettiamo solo valutazioni COMPLETE
        
        # Se è la primissima valutazione (Zero-Shot), la salviamo e la usiamo come "metro di misura"
        if len(self.acc_matrix) == 0:
            if len(self.current_acc_row) > 0:
                self.auc_matrix.append(self.current_auc_row)
                self.acc_matrix.append(self.current_acc_row)
        else:
            # Per tutte le altre, controlliamo che abbiano la stessa lunghezza della prima riga
            expected_length = len(self.acc_matrix[0])
            
            if len(self.current_acc_row) == expected_length:
                self.auc_matrix.append(self.current_auc_row)
                self.acc_matrix.append(self.current_acc_row)
            elif len(self.current_acc_row) > 0:
                # Se arriva una valutazione parziale, la ignoriamo stampando un avviso a schermo
                print(f"\n⚠️ [PLUGIN] Ignorata valutazione fantasma/parziale di Avalanche (Colonne: {len(self.current_acc_row)} invece di {expected_length})")


class FeatureDistillationPlugin(SupervisedPlugin):
    """
    Plugin per Feature Distillation Memory-Free.
    Usa la distanza L2 (MSE) tra le feature estratte dal modello vecchio e nuovo.
    """
    def __init__(self, alpha=1.0, layer_name='fc'):
        """
        :param alpha: Peso della loss di distillazione (iperparametro da tunare).
        :param layer_name: Il nome dell'ultimo layer (classificatore). 
                           Nelle ResNet standard di PyTorch si chiama 'fc'.
        """
        super().__init__()
        self.alpha = alpha
        self.layer_name = layer_name
        self.old_model = None
        self.current_features = None

    def _get_layer(self, model, name):
        """Naviga nei sottomoduli di PyTorch usando i punti (es. resnet.fc)"""
        layer = model
        for part in name.split('.'):
            layer = getattr(layer, part)
        return layer

    def before_training(self, strategy, **kwargs):
        # 1. Registriamo un hook permanente sul modello in addestramento
        # Questo intercetterà l'input del layer 'fc' (le feature!) ad ogni forward pass
        target_layer = self._get_layer(strategy.model, self.layer_name)
        def hook(module, input, output):
            self.current_features = input[0]
        target_layer.register_forward_hook(hook)

    def before_training_exp(self, strategy, **kwargs):
        # 2. Dal secondo task in poi, congeliamo una copia del modello passato
        if strategy.clock.train_exp_counter > 0:
            print(f"\n[PLUGIN] Setup Feature Distillation Attivato (Alpha: {self.alpha})")
            self.old_model = copy.deepcopy(strategy.model)
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False

    def before_backward(self, strategy, **kwargs):
        # 3. Al primo task non c'è nulla da distillare
        if self.old_model is None:
            return

        # Estraiamo le vecchie feature con un hook "al volo" sul modello congelato
        old_features = []
        def old_hook(module, input, output):
            old_features.append(input[0])
            
        # --- ECCO LA RIGA CORRETTA ---
        # Usiamo la nostra funzione invece di getattr
        old_target_layer = self._get_layer(self.old_model, self.layer_name)
        handle = old_target_layer.register_forward_hook(old_hook)
        
        # Facciamo scorrere le X correnti nel modello vecchio (senza calcolare gradienti)
        with torch.no_grad():
            self.old_model(strategy.mb_x)
            
        handle.remove() # Puliamo l'hook volante

        # 4. Calcolo della Loss L2 (Mean Squared Error)
        l2_loss = F.mse_loss(self.current_features, old_features[0])
        
        # 5. Aggiungiamo la distillazione alla loss principale (CrossEntropy)
        strategy.loss += self.alpha * l2_loss


class PrototypeiCaRLPlugin(SupervisedPlugin):
    """
    iCaRL in modalità Memory-Free. 
    Sostituisce gli esemplari con i Centroidi (Prototipi) delle classi.
    """
    def __init__(self, layer_name='resnet.fc'):
        super().__init__()
        self.layer_name = layer_name
        self.prototypes = {} # Dizionario: {class_id: tensor_centroide}
        self.current_features = None
        self._hook_handle = None

    def _get_layer(self, model, name):
        """Naviga nei sottomoduli di PyTorch usando i punti (es. resnet.fc)"""
        layer = model
        for part in name.split('.'):
            layer = getattr(layer, part)
        return layer

    def before_training(self, strategy, **kwargs):
        # Evitiamo di registrare l'hook due volte se facciamo più epoche/task
        if self._hook_handle is None:
            target_layer = self._get_layer(strategy.model, self.layer_name)
            def hook(module, input, output):
                self.current_features = input[0]
            self._hook_handle = target_layer.register_forward_hook(hook)

    def after_training_exp(self, strategy, **kwargs):
        """Dopo il training di un task, calcoliamo i prototipi per le nuove classi."""
        strategy.model.eval()
        print(f"\n[PLUGIN] Calcolo Prototipi per le classi del task corrente...")
        
        new_classes = strategy.experience.classes_in_this_experience
        class_features = {c: [] for c in new_classes}

        with torch.no_grad():
            # Usiamo il dataloader del training appena concluso per calcolare le medie
            for mb_x, mb_y, _ in strategy.dataloader:
                mb_x = mb_x.to(strategy.device)
                strategy.model(mb_x) # Triggera l'hook per popolare self.current_features
                features = self.current_features.detach().cpu()
                
                for i, label in enumerate(mb_y):
                    label_item = label.item()
                    if label_item in new_classes:
                        class_features[label_item].append(features[i])

        # Calcoliamo la media per ogni classe
        for c in new_classes:
            if len(class_features[c]) > 0:
                all_f = torch.stack(class_features[c])
                self.prototypes[c] = torch.mean(all_f, dim=0)
        
        print(f"[PLUGIN] Prototipi aggiornati. Totale classi memorizzate: {len(self.prototypes)}")

    def after_eval_iteration(self, strategy, **kwargs):
        """
        SOVRASCRITTURA PREDIZIONE:
        Invece di usare l'output del modello, usiamo la distanza dai prototipi.
        """
        if len(self.prototypes) == 0 or self.current_features is None:
            return

        # Feature correnti estratte dall'hook
        features = self.current_features.detach().cpu() # (Batch, Dim)
        
        # Prepariamo i prototipi come matrice (NumClassi, Dim)
        proto_classes = sorted(self.prototypes.keys())
        proto_matrix = torch.stack([self.prototypes[c] for c in proto_classes])
        
        # Calcoliamo la distanza Euclidea tra ogni sample e ogni prototipo
        # (Batch, 1, Dim) - (1, NumClassi, Dim) -> (Batch, NumClassi)
        dist = torch.cdist(features.unsqueeze(1), proto_matrix.unsqueeze(0)).squeeze(1)
        
        # La predizione è la classe con la distanza MINIMA
        preds_indices = torch.argmin(dist, dim=1)
        preds_labels = torch.tensor([proto_classes[i] for i in preds_indices])
        
        # Sovrascriviamo le predizioni della strategia per le metriche
        strategy.mb_output = torch.zeros((features.shape[0], max(proto_classes)+1)).to(strategy.device)
        for i, label in enumerate(preds_labels):
            strategy.mb_output[i, label] = 1.0 # One-hot fittizio per far funzionare le metriche



class CosineLayer(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.sigma is not None:
            nn.init.constant_(self.sigma, 1)

    def forward(self, input):
        # Normalizziamo sia l'input (feature) che i pesi (classi)
        out = F.linear(F.normalize(input, p=2, dim=1), 
                       F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class UCIRPlugin(SupervisedPlugin):
    """
    Implementazione UCIR Memory-Free. 
    Preserva la geometria dello spazio latente tramite Less-Forget Constraint.
    """
    def __init__(self, alpha=5.0, layer_name='resnet.fc'):
        super().__init__()
        self.alpha = alpha
        self.layer_name = layer_name
        self.old_model = None
        self.current_features = None
        self._hook_handle = None

    def _get_layer(self, model, name):
        """Naviga nei sottomoduli di PyTorch usando i punti (es. resnet.fc)"""
        layer = model
        for part in name.split('.'):
            layer = getattr(layer, part)
        return layer

    def before_training(self, strategy, **kwargs):
        # 1. Registriamo l'hook per catturare le feature correnti (come in FD)
        if self._hook_handle is None:
            target_layer = self._get_layer(strategy.model, self.layer_name)
            def hook(module, input, output):
                self.current_features = input[0]
            self._hook_handle = target_layer.register_forward_hook(hook)

    def before_training_exp(self, strategy, **kwargs):
        # 2. Dal secondo task in poi, congeliamo il modello vecchio
        if strategy.clock.train_exp_counter > 0:
            print(f"\n[PLUGIN] Setup UCIR Attivato (Alpha: {self.alpha})")
            self.old_model = copy.deepcopy(strategy.model)
            self.old_model.eval()
            for p in self.old_model.parameters():
                p.requires_grad = False

    def before_backward(self, strategy, **kwargs):
        # 3. Less-Forget Constraint: Distillazione geometrica
        if self.old_model is None or self.current_features is None:
            return

        # Estraiamo le feature dal modello vecchio tramite hook volante
        old_features = []
        def old_hook(module, input, output):
            old_features.append(input[0])
            
        old_target_layer = self._get_layer(self.old_model, self.layer_name)
        handle = old_target_layer.register_forward_hook(old_hook)
        
        with torch.no_grad():
            self.old_model(strategy.mb_x)
            
        handle.remove() # Puliamo

        # Calcoliamo la similarità coseno tra le feature attuali e passate
        # Vogliamo che l'angolo tra i vettori rimanga identico (dist_loss -> 0)
        dist_loss = 1.0 - F.cosine_similarity(self.current_features, old_features[0]).mean()
        
        # Aggiungiamo alla loss di base
        strategy.loss += self.alpha * dist_loss


class RelativeRepresentationLayer(nn.Module):
    def __init__(self, in_features, num_anchors, out_features=2):
        super().__init__()
        self.num_anchors = num_anchors
        
        # register_buffer salva il tensore nel modello senza che richieda gradienti
        self.register_buffer('anchors', torch.zeros(num_anchors, in_features))
        self.anchors_ready = False
        
        # Il classificatore finale (l'unica cosa che si addestrerà)
        self.classifier = nn.Linear(num_anchors, out_features)

    def set_anchors(self, anchor_features):
        """Salva le ancore estratte e attiva il layer"""
        assert anchor_features.shape[0] == self.num_anchors, "Numero di ancore errato!"
        self.anchors.copy_(anchor_features.detach())
        self.anchors_ready = True

    def forward(self, x):
        if not self.anchors_ready:
            # Ritorno dummy di sicurezza prima dell'inizializzazione
            return self.classifier(torch.zeros(x.size(0), self.num_anchors, device=x.device))
            
        # Normalizzazione L2 per la similarità coseno
        x_norm = F.normalize(x, p=2, dim=1)
        anchors_norm = F.normalize(self.anchors, p=2, dim=1)
        
        # Prodotto scalare matriciale (Batch, Dim) @ (Dim, M) -> (Batch, M)
        # Il risultato sono le Relative Features!
        relative_features = torch.matmul(x_norm, anchors_norm.t())
        
        # Le passiamo al layer lineare
        return self.classifier(relative_features)
    
class RelativePlugin(SupervisedPlugin):
    """
    Gestisce l'estrazione delle Ancore (Real) e il congelamento della backbone
    per il metodo Relative Representation.
    """
    def __init__(self, num_anchors=100, real_label_id=0):
        super().__init__()
        self.num_anchors = num_anchors
        self.real_label_id = real_label_id # L'ID della classe Real (es. 0 o 1)
        self.is_setup_done = False

    def before_training_exp(self, strategy, **kwargs):
        # Facciamo il setup SOLO prima del primissimo task
        if self.is_setup_done:
            return
            
        print(f"\n[PLUGIN RELATIVE] Congelamento della backbone e ricerca di {self.num_anchors} ancore Real...")
        
        model = strategy.model
        device = strategy.device
        model.eval() # Mettiamo in eval per bloccare BatchNorm/Dropout
        
        # 1. CONGELIAMO TUTTO TRANNE IL CLASSIFICATORE FINALE
        for name, param in model.named_parameters():
            if 'classifier' not in name: # Il classificatore è in RelativeRepresentationLayer
                param.requires_grad = False
                
        # 2. ESTRAIAMO LE ANCORE
        # Sostituiamo temporaneamente il nostro layer con un Identity per ottenere le feature pure
        relative_layer = model.resnet.fc
        model.resnet.fc = nn.Identity()
        
        anchors_list = []
        collected = 0
        
        with torch.no_grad():
            for mb_x, mb_y, _ in strategy.dataloader:
                # Filtriamo solo le immagini Real (assicurati che real_label_id sia corretto per il tuo dataset!)
                real_mask = (mb_y == self.real_label_id)
                real_imgs = mb_x[real_mask].to(device)
                
                if len(real_imgs) > 0:
                    features = model(real_imgs) # Estraiamo
                    anchors_list.append(features)
                    collected += len(features)
                    
                if collected >= self.num_anchors:
                    break
                    
        if collected < self.num_anchors:
            print(f"⚠️ [WARNING] Trovate solo {collected} ancore Real invece di {self.num_anchors}!")
            
        # Uniamo i tensori e prendiamo esattamente il numero richiesto
        all_anchors = torch.cat(anchors_list, dim=0)[:self.num_anchors]
        
        # 3. RIPRISTINIAMO IL LAYER E CARICHIAMO LE ANCORE
        model.resnet.fc = relative_layer
        model.resnet.fc.set_anchors(all_anchors)
        
        self.is_setup_done = True
        print(f"[PLUGIN RELATIVE] Setup Completato! Inizia il training sul layer lineare.")