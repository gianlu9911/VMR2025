import torch
import torchvision.models
from torch import nn

class ResNet50BC(torch.nn.Module):
    def __init__(self):
        super(ResNet50BC, self).__init__()
        self.name = 'resnet50'
        self.image_size = 224

        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

        num_ftrs = self.resnet.fc.in_features # 2048
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.resnet(x)

def predict(model, x, labels=None):
    # switch to evaluate mode
    model.eval()

    output = model(x)

    with torch.no_grad():
        batch_size = x.size(0)

        _, predicted = torch.max(output.data, 1)
        if labels is not None:
            acc = (predicted == labels).sum().item() / batch_size
            # res = ['Real' if predicted[i] == 0 else 'Fake' for i in range(batch_size)]
            res = ['Real' if p == 0 else 'Fake' for p in predicted]
            return acc, res
        else:
            res = ['Real' if p == 0 else 'Fake' for p in predicted]
            return res

def load_pretrained_model(model_pth):
    classifier = ResNet50BC()
    print('Loading model {}'.format(model_pth))
    checkpoint = torch.load(model_pth, weights_only=True)
    classifier.load_state_dict(checkpoint['state_dict'])
    print('Done.\n')
    return classifier



