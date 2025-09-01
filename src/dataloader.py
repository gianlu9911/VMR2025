import os

from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from .transform import data_transforms

class RealSynthethicDataloader(Dataset):
    def __init__(self, real_dir, fake_dir, split='train_set'):
        rgb_real = sorted(glob(os.path.join(real_dir, split, '*.png')))
        rgb_fake = sorted(glob(os.path.join(fake_dir, split, '*.png')))
        if len(rgb_fake) == 0: # SDv1.4
            rgb_fake = sorted(glob(os.path.join(fake_dir, split, '*.jpg')))

        assert (len(rgb_real) == len(rgb_fake))
        #
        self.images = rgb_real + rgb_fake
        self.len = len(self.images)
        #
        self.preprocess_rgb = data_transforms['image']
        self.labels = dict(zip(self.images, ([0] * len(rgb_real) + [1] * len(rgb_fake)))) # Real = 0, AI-Generated=1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.preprocess_rgb(Image.open(self.images[idx]).convert('RGB'))
        y = self.labels[self.images[idx]]
        return x, y

