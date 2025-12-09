import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp

root_folder = "logs/logits"
a = np.load(osp.join(root_folder, "real_step_sdv1_4.npy"))
b = np.load(osp.join(root_folder, "real_step_stylegan1.npy"))

# check if the features are the same
print("Are the features equal?", np.array_equal(a, b))
# print distance between the features
print("Distance between real logits:", np.linalg.norm(a - b))
print("shape a:", a.shape)

c = np.load("logs/relative/fake_step_stylegan1_faketype_stylegan1.npy")
d = np.load("logs/relative/fake_step_sdv1_4_faketype_stylegan1.npy")
#e = np.load("saved_numpy_features/step_sdv21_fake_stylegan2.npy")
# check if the features are the same
print("Are the fake features equal?", np.array_equal(c, d))
# print distance between the features
print("Distance between fake features:", np.linalg.norm(c - d))
# check distance between sdv21 and stylegan2 features
#print("Distance between sdv21 and stylegan2 fake features:", np.linalg.norm(c - e))