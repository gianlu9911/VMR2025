import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
steps = ['stylegan1', 'stylegan2', 'sdv1_4', 'stylegan3', 'stylegan_xl', 'sdv2_1']
base_dir = "saved_numpy_features"
target_step = 'sdv2_1'  # step 6
target_idx = steps.index(target_step)
previous_steps = steps[:target_idx]

out_dir = "difference_matrices"
os.makedirs(out_dir, exist_ok=True)

# --- Utility functions ---
def pairwise_distances(A, B):
    d = min(A.shape[1], B.shape[1])
    A, B = A[:, :d], B[:, :d]
    a2 = (A**2).sum(1, keepdims=True)
    b2 = (B**2).sum(1, keepdims=True).T
    D2 = a2 + b2 - 2 * (A @ B.T)
    np.maximum(D2, 0, out=D2)
    return np.sqrt(D2)

def safe_load(path):
    return np.load(path).astype(np.float32)

def compute_and_plot(A, B, type_label, step_a, step_b):
    """Compute and visualize distance matrix."""
    D = pairwise_distances(A, B)
    mean_dist = D.mean()
    print(f"[{type_label}] {step_a} vs {step_b} | shape={D.shape}, mean={mean_dist:.4f}")

    # --- Heatmap ---
    plt.figure(figsize=(7,5))
    im = plt.imshow(D, cmap='viridis', aspect='auto')
    plt.title(f"{type_label.capitalize()} distances: {step_a} vs {step_b}\n(mean={mean_dist:.4f})")
    plt.xlabel(f"{step_b} {type_label}")
    plt.ylabel(f"{step_a} {type_label}")
    plt.colorbar(im, label="Euclidean distance")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{type_label}_{step_a}_vs_{step_b}.png", dpi=200)
    plt.close()

    # --- Diagonal check ---
    if D.shape[0] == D.shape[1]:
        diag_vals = np.diag(D)
        diag_mean = diag_vals.mean()
        print(f"→ Diagonal mean={diag_mean:.6f}, min={diag_vals.min():.6f}, max={diag_vals.max():.6f}")
        plt.figure(figsize=(6,4))
        plt.bar(np.arange(len(diag_vals)), diag_vals, color='steelblue')
        plt.title(f"{type_label.capitalize()} diagonal ({step_a} vs {step_b})\nmean={diag_mean:.4f}")
        plt.xlabel("Index")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{type_label}_{step_a}_vs_{step_b}_diagonal.png", dpi=200)
        plt.close()

# --- Main loop ---
print("=== Processing FAKE features following step-matching rule ===")

for prev in reversed(previous_steps):
    fake_source = prev  # the smaller step determines the fake subset
    path_A = os.path.join(base_dir, f"step_{target_step}_relative_fake_{fake_source}.npy")
    path_B = os.path.join(base_dir, f"step_{prev}_relative_fake_{fake_source}.npy")

    if not (os.path.exists(path_A) and os.path.exists(path_B)):
        print(f"⚠️ Missing fake features for {fake_source} at {target_step} or {prev}, skipping...")
        continue

    print(f"\nComparing FAKE_{fake_source}: {target_step} vs {prev}")
    A = safe_load(path_A)
    B = safe_load(path_B)
    compute_and_plot(A, B, f"fake_{fake_source}", target_step, prev)

print("\n✅ All fake comparisons completed and saved in 'difference_matrices/'")
