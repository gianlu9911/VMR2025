import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp

root_folder = "anchros_logits"
a = np.load(osp.join(root_folder, "step_stylegan2_1000.npy"))
b = np.load(osp.join(root_folder, "step_stylegan_xl_1000.npy"))

# check if the features are the same
print("Are the features equal?", np.array_equal(a, b))
# print distance between the features
print("Distance between real logits:", np.mean(a - b))
print("shape a:", a.shape)

c = np.load("anchros_logits/step_stylegan2_1000.npy")
d = np.load("anchros_logits/step_stylegan1_1000.npy")
#e = np.load("saved_numpy_features/step_sdv21_fake_stylegan2.npy")
# check if the features are the same
print("Are the fake features equal?", np.array_equal(c, d))
# print distance between the features
print("Distance between fake features:", np.mean(c - d))
# check distance between sdv21 and stylegan2 features
#print("Distance between sdv21 and stylegan2 fake features:", np.linalg.norm(c - e))

#print the first value
print(a[:5])