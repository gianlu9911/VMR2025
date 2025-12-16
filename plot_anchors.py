import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
# ---- CONFIG ----
OUTPUT_DIR = "logs/plots_pdf_train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STEPS_ORDER = [
    "stylegan1",
    "stylegan2",
    "sdv1_4",
    "stylegan3",
    "stylegan_xl",
    "sdv2_1",
]

how_many = 100  # quante ancore visualizzare per step

# colormap discreta
cmap = plt.get_cmap("tab10")
step_to_color = {step: cmap(i) for i, step in enumerate(STEPS_ORDER)}

# ---- Plot cumulativo ----
plt.figure(figsize=(8, 6))
alpha_val = 0.6

for step in STEPS_ORDER:
    if step not in logits_by_step:
        print(f"[WARN] step '{step}' non presente")
        continue
    if "anchors" not in logits_by_step[step]:
        print(f"[WARN] nessuna anchor per step '{step}'")
        continue

    anchors_list = logits_by_step[step]["anchors"]
    if not anchors_list:
        continue

    anchors = np.concatenate(anchors_list, axis=0)[:how_many]

    plt.scatter(
        anchors[:, 0],
        anchors[:, 1],
        s=80,
        marker='*',
        color=step_to_color[step],
        alpha=alpha_val,
        label=step,
        zorder=3
    )

plt.xlabel("Class 0")
plt.ylabel("Class 1")
plt.title("Anchor stability across steps\n(anchors do not drift)")
plt.legend(title="Step", markerscale=1.2)
plt.grid(True)
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "anchors_stability_across_steps.pdf")
plt.savefig(out_path)
plt.close()

print(f"Saved anchor stability plot: {out_path}")
