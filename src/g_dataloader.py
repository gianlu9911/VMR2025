from torch.utils.data import Dataset
import os
import random
from glob import glob
from PIL import Image
from src.transform import data_transforms

class RealSynthethicDataloader(Dataset):
    def __init__(self, real_dir=None, fake_dir1=None, fake_dir2=None, 
                 split='train_set', balance_fake_to_real=False, seed=42):

        random.seed(seed)
        self.images = []
        self.labels = {}
        self.source_counts = {}

        # Real images
        if real_dir:
            rgb_real = sorted(glob(os.path.join(real_dir, split, '*.png')))
            rgb_real += sorted(glob(os.path.join(real_dir, split, '*.jpg')))
            self.source_counts['real'] = len(rgb_real)
            self.images += rgb_real
            self.labels.update({img: 0 for img in rgb_real})  # Real = 0
        else:
            self.source_counts['real'] = 0

        # Fake 1
        fake1 = []
        if fake_dir1:
            fake1 += sorted(glob(os.path.join(fake_dir1, split, '*.png')))
            fake1 += sorted(glob(os.path.join(fake_dir1, split, '*.jpg')))
            fake1 = list(set(fake1))  # Deduplicate
        self.source_counts['fake1'] = len(fake1)

        # Fake 2
        fake2 = []
        if fake_dir2:
            fake2 += sorted(glob(os.path.join(fake_dir2, split, '*.png')))
            fake2 += sorted(glob(os.path.join(fake_dir2, split, '*.jpg')))
            fake2 = list(set(fake2))
        self.source_counts['fake2'] = len(fake2)

        # Balancing if needed
        if balance_fake_to_real and real_dir:
            max_fake = self.source_counts['real']
            total_fake = len(fake1) + len(fake2)
            if total_fake > 0:
                portion1 = int((len(fake1) / total_fake) * max_fake)
                portion2 = max_fake - portion1
                fake1 = random.sample(fake1, min(len(fake1), portion1))
                fake2 = random.sample(fake2, min(len(fake2), portion2))

        rgb_fake = fake1 + fake2
        self.source_counts['used_fake'] = len(rgb_fake)

        self.images += rgb_fake
        self.labels.update({img: 1 for img in rgb_fake})  # Fake = 1

        self.len = len(self.images)
        self.preprocess_rgb = data_transforms['image']

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = self.images[idx]
        x = self.preprocess_rgb(Image.open(img_path).convert('RGB'))
        y = self.labels[img_path]
        return x, y
