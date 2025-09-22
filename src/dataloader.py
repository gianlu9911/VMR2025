import os

from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from .transform import data_transforms

class RealSynthethicDataloader(Dataset):
    def __init__(self, real_dir, fake_dir, num_points = None, split='train_set'):
        rgb_real = sorted(glob(os.path.join(real_dir, split, '*.png')))
        rgb_fake = sorted(glob(os.path.join(fake_dir, split, '*.png')))
        print(f'Number of real images: {len(rgb_real)}')
        if len(rgb_fake) == 0: # SDv1.4
            rgb_fake = sorted(glob(os.path.join(fake_dir, split, '*.jpg')))
        print(f'Number of fake images: {len(rgb_fake)}')
        if num_points is not None:
            rgb_real = rgb_real[:num_points]
            rgb_fake = rgb_fake[:num_points]
            print(f'Using {len(rgb_real)} real and {len(rgb_fake)} fake images')

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

