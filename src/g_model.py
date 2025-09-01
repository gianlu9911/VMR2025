import argparse
import torch
import os

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import csv
from glob import glob
from PIL import Image
import random

from config import PRETRAINED_MODELS, IMAGE_DIR
from .net import load_pretrained_model
from .transform import data_transforms


class TwoBackbonesClassifier(nn.Module):
    def __init__(self, model1, model2, num_classes=2):
        super().__init__()
        for m in (model1, model2):
            for p in m.parameters():
                p.requires_grad = False

        self.backbone1 = nn.Sequential(*list(model1.children())[:-1])
        self.backbone2 = nn.Sequential(*list(model2.children())[:-1])

        dim1 = 2048 # model1.fc.in_features
        dim2 = 2048 # model2.fc.in_features

        self.classifier = nn.Linear(dim1 + dim2, num_classes)

    def forward(self, x):
        f1 = self.backbone1(x).flatten(1)
        f2 = self.backbone2(x).flatten(1)
        f = torch.cat([f1, f2], dim=1)
        return self.classifier(f)