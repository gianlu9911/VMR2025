from torchvision.models import resnet50
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from bin_classifier import accuracy, RealSynthethicDataloader, nn, IMAGE_DIR, DataLoader


########### MAIN ###########
def main(epochs=10):
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    ############################################################
    # 1. CREA RESNET50 NON PREADDDESTRATA
    ############################################################
    print("Creating ResNet50 (non pre-trained)")
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)   # real / fake
    model.to(device)

    ############################################################
    # 2. DATI DI TRAINING
    ############################################################
    real_dir = IMAGE_DIR['real']
    fake_dir = IMAGE_DIR['stylegan_xl']

    train_ds = RealSynthethicDataloader(real_dir, fake_dir)
    train_loader = DataLoader(train_ds, batch_size=64,
                              num_workers=8, shuffle=True)

    ############################################################
    # 3. LOSS + OPTIMIZER
    ############################################################
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ############################################################
    # 4. TRAINING LOOP
    ############################################################
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        running_acc = 0

        print(f"\n--- Training epoch {epoch+1}/{epochs} ---")

        for batch_idx, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, target)
            acc = accuracy(output, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * target.size(0)
            running_acc += acc * target.size(0)

            # Print batch info every 20 batches
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} "
                      f"- loss: {loss.item():.4f}, acc: {acc:.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)

        print(f"[Epoch {epoch+1}] loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

    ############################################################
    # 5. SALVA MODELLO
    ############################################################
    os.makedirs("checkpoint", exist_ok=True)
    torch.save(model.state_dict(), "checkpoint/resnet50_styleganxl.pth")
    print("Modello salvato come checkpoint/resnet50_styleganxl.pth")

main(1)
