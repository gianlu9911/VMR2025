import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
anchors_path = "anchros_logits/step_sdv1_4_1000.npy"

anchors = np.load(anchors_path)[:50]

def load_logits(kind, step):
    return np.load(os.path.join(
        train_logits_dir, f"{kind}_step_{step}.npy"
    ))

fake_logits_cache = {}

# colormap for fake logits
reds = cm.Reds
n_steps = len(steps)

for i, step in enumerate(steps):
    fake_logits = load_logits("fake", step)[:50]
    real_logits = load_logits("real", step)[:50]
    fake_logits_cache[step] = fake_logits

    # ---------------------------
    # Plot 1: current step only
    # ---------------------------
    plt.figure()
    plt.scatter(
        anchors[:, 0], anchors[:, 1],
        marker="*", c="black", label="Anchors (50)"
    )

    fake_color = reds((i + 1) / n_steps)
    plt.scatter(
        fake_logits[:, 0], fake_logits[:, 1],
        c=[fake_color], label=f"Fake {step}"
    )

    plt.scatter(
        real_logits[:, 0], real_logits[:, 1],
        marker="^", c="green", label="Real"
    )

    plt.title(f"{step} : current logits")
    plt.xlabel("logit dim 1")
    plt.ylabel("logit dim 2")
    plt.legend()
    plt.show()

    # -------------------------------------------
    # Plot 2: fake logits from all past steps
    # -------------------------------------------
    plt.figure()
    plt.scatter(
        anchors[:, 0], anchors[:, 1],
        marker="*", c="black", label="Anchors (50)"
    )

    for j, past_step in enumerate(steps[:i + 1]):
        past_fake = fake_logits_cache[past_step]
        past_color = reds((j + 1) / n_steps)

        plt.scatter(
            past_fake[:, 0],
            past_fake[:, 1],
            c=[past_color],
            alpha=0.7,
            label=f"Fake {past_step}"
        )

    plt.scatter(
        real_logits[:, 0], real_logits[:, 1],
        marker="^", c="green", label="Real (current)"
    )

    plt.title(f"{step} : fake logits from past steps")
    plt.xlabel("Class 0")
    plt.ylabel("Class 1")
    plt.legend()
    plt.savefig(f"fake_logits_past_steps_{step}.pdf")
    print(f"Saved plot: {step} in fake_logits_past_steps_{step}.pdf")
 