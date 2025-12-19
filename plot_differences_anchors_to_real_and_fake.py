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

train_logits_dir = "logs/logits"
anchors_dir = "anchros_logits"

def load_logits(kind, step):
    if kind == "real":
        return np.load(os.path.join(train_logits_dir, f"{kind}_step_{step}.npy"))[:50]
    elif kind == "fake":
        return np.load(os.path.join(train_logits_dir, f"{kind}_step_{step}_faketype_{step}.npy"))[:50]

def mean_pairwise_l2(A, B):
    diff = A[:, None, :] - B[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist.mean()

# ---- compute distances per step
distances_fake = []
distances_real = []

for step in steps:
    # load anchors for this step
    anchors_path = os.path.join(anchors_dir, f"step_{step}_1000.npy")
    anchors = np.load(anchors_path)[:50]

    # load fake and real logits
    fake_logits = load_logits("fake", step)
    real_logits = load_logits("real", step)

    # compute mean distances
    distances_fake.append(mean_pairwise_l2(anchors, fake_logits))
    distances_real.append(mean_pairwise_l2(anchors, real_logits))

# ---- plot
plt.figure()
plt.plot(steps, distances_fake, marker="o", label="Fake", color="red")
plt.plot(steps, distances_real, marker="^", label="Real", color="green")
plt.title("Anchor distance curve (per step)")
plt.xlabel("Step")
plt.ylabel("Mean L2 distance")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.savefig("anchor_distance_curve_per_step.pdf")
print("Saved plot: anchor_distance_curve_per_step.pdf")
