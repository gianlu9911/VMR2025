import random
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.real_indices = [i for i, path in enumerate(dataset.images) if dataset.labels[path] == 0]
        self.fake_indices = [i for i, path in enumerate(dataset.images) if dataset.labels[path] == 1]

        assert self.batch_size % 2 == 0, "Batch size must be even for balanced sampling."

    def __iter__(self):
        real_perm = np.random.permutation(self.real_indices)
        fake_perm = np.random.permutation(self.fake_indices)
        min_len = min(len(real_perm), len(fake_perm))

        for i in range(0, min_len, self.batch_size // 2):
            real_batch = real_perm[i:i + self.batch_size // 2]
            fake_batch = fake_perm[i:i + self.batch_size // 2]
            if len(real_batch) < self.batch_size // 2 or len(fake_batch) < self.batch_size // 2:
                continue  # skip incomplete batch
            batch = np.concatenate([real_batch, fake_batch])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return min(len(self.real_indices), len(self.fake_indices)) // (self.batch_size // 2)