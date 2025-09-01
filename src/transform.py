import torchvision.transforms as T

data_transforms = {
    'image': T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}