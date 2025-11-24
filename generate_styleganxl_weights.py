from tqdm import tqdm
from src.net import ResNet50BC
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from config import PRETRAINED_MODELS, IMAGE_DIR
from bin_classifier import  accuracy, nn, IMAGE_DIR, PRETRAINED_MODELS, load_pretrained_model
from src.g_dataloader import RealSynthethicDataloader
from src.utils import train_one_epoch


def evaluate(dataloader, model, criterion, device, print_freq=10):
    # switch to evaluate mode
    model.eval()

    val_loss = 0
    val_acc = 0
    k = 0

    with torch.no_grad():
        with tqdm(total=len(dataloader), unit=' batch', desc='Evaluation on test_set: ') as bar:
            for i, (images, target) in enumerate(dataloader):
                bar.update(1)
                if device is not None:
                    images = images.to(device)
                target = target.to(device)

                # compute output
                output = model(images)
                val_loss_it = criterion(output, target)
                val_acc_it = accuracy(output, target)

                val_acc += val_acc_it * target.size(0)
                val_loss += val_loss_it.item() * target.size(0)

                k += target.size(0)  # no of samples already processed

                if i % print_freq == 0:
                    bar.set_postfix({'batch_test_loss' : round(val_loss_it.item(), 5),
                                     'batch_test_acc' : round(val_acc_it, 5),
                                     'test_loss' : round(val_loss / k, 5),
                                     'test_acc' : round(val_acc / k, 5)
                                     })

            bar.close()

    val_loss /= len(dataloader.sampler)
    val_acc /= len(dataloader.sampler)

    return val_loss, val_acc

def evaluate1(dataloader, model, criterion, device, print_freq=10):
    model.eval()

    val_loss = 0
    val_acc = 0
    k = 0

    with torch.no_grad():
        with tqdm(total=len(dataloader), unit=' batch', desc='Evaluation on test_set: ') as bar:
            for i, (images, target) in enumerate(dataloader):
                bar.update(1)

                images = images.to(device)
                target = target.to(device)

                output = model(images)
                loss_it = criterion(output, target)

                acc_it = accuracy(output, target).item()   # FIX HERE

                val_loss += loss_it.item() * target.size(0)
                val_acc += acc_it * target.size(0)
                k += target.size(0)

                if i % print_freq == 0:
                    bar.set_postfix({
                        'batch_test_loss': round(loss_it.item(), 5),
                        'batch_test_acc': round(acc_it, 5),
                        'test_loss': round(val_loss / k, 5),
                        'test_acc': round(val_acc / k, 5)
                    })

    dataset_size = len(dataloader.dataset)
    val_loss /= dataset_size
    val_acc /= dataset_size

    return val_loss, val_acc

########### MAIN ###########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = ResNet50BC().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 
criterion = torch.nn.CrossEntropyLoss()

train_dataset = RealSynthethicDataloader(real_dir=IMAGE_DIR['real'], fake_dir1=IMAGE_DIR['stylegan_xl'], split='train_set')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
print("Starting training...")
#for epoch in range(5):
    #train_one_epoch(model, train_loader, criterion, optimizer, device)
print("Evaluating model...")
checkpoint_path = 'checkpoint/stylegan_xl.pth'
#torch.save({'state_dict': model.state_dict()}, checkpoint_path)
model = load_pretrained_model(checkpoint_path).to(device)
test_dataset = RealSynthethicDataloader(real_dir=IMAGE_DIR['real'], fake_dir1=IMAGE_DIR['stylegan_xl'], split='test_set')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
loss, acc = evaluate(test_loader,model, criterion, device)
print(f"Test Loss: {loss}, Test Accuracy: {acc}")