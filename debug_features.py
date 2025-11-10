import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp

root_folder = "saved_numpy_features"
a = np.load(osp.join(root_folder, "step_sdv1_4_anchors.npy"))
b = np.load(osp.join(root_folder, "step_stylegan1_anchors.npy"))

# check if the features are the same
print("Are the features equal?", np.array_equal(a, b))
# print distance between the features
print("Distance between features:", np.linalg.norm(a - b))


c = np.load("saved_numpy_features/step_stylegan_xl_fake_stylegan1.npy")
d = np.load("saved_numpy_features/step_stylegan1_fake_stylegan1.npy")
e = np.load("saved_numpy_features/step_sdv21_fake_stylegan2.npy")
# check if the features are the same
print("Are the fake features equal?", np.array_equal(c, d))
# print distance between the features
print("Distance between fake features:", np.linalg.norm(c - d))
# check distance between sdv21 and stylegan2 features
print("Distance between sdv21 and stylegan2 fake features:", np.linalg.norm(c - e))