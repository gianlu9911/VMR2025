import numpy as np
import matplotlib.pyplot as plt
import os

steps = ['stylegan1', 'stylegan2', 'sdv1_4', 'stylegan3', 'stylegan_xl', 'sdv2_1']
base_dir = "logs/logits"
target_step = "sdv2_1"
target_idx = steps.index(target_step)
previous_steps = steps[:target_idx]   # all before sdv2_1

def load_feats(path):
    return np.load(path).astype(np.float32)

def mse(A, B):
    # match dimensionality
    d = min(A.shape[1], B.shape[1])
    #A = A[:, :d]
    #B = B[:, :d]
    return ((A - B) ** 2).mean()

# ------------------------------
#   REAL FEATURES
# ------------------------------
real_target_path = os.path.join(base_dir, f"real_step_{target_step}.npy")
real_target = load_feats(real_target_path)

real_mse_list = []

print("=== REAL FEATURES ===")
for step in previous_steps:
    path_prev = os.path.join(base_dir, f"real_step_{step}.npy")
    if not os.path.exists(path_prev):
        print(f"Missing real {step}, skipping…")
        real_mse_list.append(None)
        continue

    real_prev = load_feats(path_prev)
    d = mse(real_target, real_prev)
    real_mse_list.append(d)
    print(f"REAL: sdv2_1 vs {step} → MSE = {d:.6f}")

# ------------------------------
#   FAKE FEATURES
# ------------------------------
fake_subtypes = ['stylegan1', 'stylegan2', 'sdv1_4', 'stylegan3', 'stylegan_xl', 'sdv2_1']
fake_curves = {ft: [] for ft in fake_subtypes}

print("\n=== FAKE FEATURES ===")
for fake_type in fake_subtypes:

    target_fake_path = os.path.join(
        base_dir, f"fake_step_{target_step}_faketype_{fake_type}.npy"
    )
    if not os.path.exists(target_fake_path):
        print(f"Missing target fake for subtype {fake_type}, skipping subtype.")
        continue

    fake_target = load_feats(target_fake_path)

    for step in previous_steps:
        prev_fake_path = os.path.join(
            base_dir, f"fake_step_{step}_faketype_{fake_type}.npy"
        )

        if not os.path.exists(prev_fake_path):
            print(f"Missing fake {fake_type} for step {step}, skipping…")
            fake_curves[fake_type].append(None)
            continue

        fake_prev = load_feats(prev_fake_path)
        d = mse(fake_target, fake_prev)
        fake_curves[fake_type].append(d)
        print(f"FAKE {fake_type}: sdv2_1 vs {step} → MSE = {d:.6f}")

# ------------------------------
#   PLOT
# ------------------------------
plt.figure(figsize=(10, 6))

x = np.arange(len(previous_steps))

# REAL curve
plt.plot(x, real_mse_list, marker='o', linewidth=2, label="REAL")

# FAKE curves
for fake_type in fake_subtypes:
    if all(v is None for v in fake_curves[fake_type]):
        continue
    plt.plot(x, fake_curves[fake_type], marker='o', linewidth=2, label=f"FAKE {fake_type}")

plt.xticks(x, previous_steps, rotation=45)
plt.ylabel("MSE distance vs sdv2_1")
plt.title("Feature MSE between sdv2_1 and previous steps")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig("mse_plot.pdf")

plt.close()
