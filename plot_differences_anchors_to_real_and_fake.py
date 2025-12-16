import numpy as np

def mean_pairwise_l2(A, B):
    """
    A: (Na, D) anchors
    B: (Nb, D) logits (real or fake)
    returns: scalar mean pairwise L2 distance
    """
    diff = A[:, None, :] - B[None, :, :]  # (Na, Nb, D)
    dist = np.linalg.norm(diff, axis=-1)  # (Na, Nb)
    return dist.mean()
import numpy as np
import matplotlib.pyplot as plt
import os

steps = [
    'stylegan1',
    'stylegan2',
    'sdv1_4',
    'stylegan3',
    'stylegan_xl',
    'sdv2_1'
]

train_logits_dir = "logs/train/logits"

# choose the target step
target_step = "sdv2_1"

# load anchors of target step
anchors_path = f"anchros_logits/step_{target_step}_1000.npy"
anchors = np.load(anchors_path)[:50]  # keep consistent with plots

def load_logits(kind, step):
    return np.load(
        os.path.join(train_logits_dir, f"{kind}_step_{step}.npy")
    )[:50]

def mean_pairwise_l2(A, B):
    diff = A[:, None, :] - B[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist.mean()

# ---- compute distances
distances_fake = []
distance_real = None

for step in steps:
    fake_logits = load_logits("fake", step)
    d_fake = mean_pairwise_l2(anchors, fake_logits)
    distances_fake.append(d_fake)

    if step == target_step:
        real_logits = load_logits("real", step)
        distance_real = mean_pairwise_l2(anchors, real_logits)

# ---- plot
plt.figure()
plt.plot(steps, distances_fake, marker="o", label="Fake")
plt.axhline(
    distance_real,
    linestyle="--",
    label="Real (target step)"
)

plt.title(f"Anchor distance curve (target = {target_step})")
plt.xlabel("Step")
plt.ylabel("Mean L2 distance")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.savefig(f"anchor_distance_curve_to_{target_step}.pdf")
print(f"Saved plot: anchor_distance_curve_to_{target_step}.pdf")
