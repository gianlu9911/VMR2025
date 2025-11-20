import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

LOGITS_DIR = "logs/logits"
OUTPUT_DIR = "logs/plots_pdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_filename(fname):
    base = fname.replace(".npy", "")
    parts = base.split("_")

    is_real = parts[0] == "real"
    is_fake = parts[0] == "fake"

    step = None
    if "step" in parts:
        step_idx = parts.index("step")
        step = parts[step_idx + 1]

    faketype = None
    if is_fake and "faketype" in parts:
        fidx = parts.index("faketype")
        faketype = parts[fidx + 1]

    return is_real, is_fake, step, faketype


# ---- Load logits ----
logits_by_step = defaultdict(lambda: defaultdict(list))

for fname in os.listdir(LOGITS_DIR):
    if not fname.endswith(".npy"):
        continue

    is_real, is_fake, step, faketype = parse_filename(fname)

    # Take only first 100 logits
    logits = np.load(os.path.join(LOGITS_DIR, fname)).reshape(-1)[:50]

    if is_real:
        logits_by_step[step]["real"].append(logits)
    elif is_fake:
        logits_by_step[step].setdefault("fake", defaultdict(list))
        logits_by_step[step]["fake"][faketype].append(logits)


# ---- Scatterplot + SAVE PDF ----
for step, groups in logits_by_step.items():
    plt.figure(figsize=(12, 6))

    # Real (stelline)
    if "real" in groups:
        all_real = np.concatenate(groups["real"])
        x_real = np.arange(len(all_real))
        plt.scatter(
            x_real, all_real,
            s=60,               # punti grandi
            alpha=0.8,
            marker='*',         # stelline
            label="real"
        )

    # Fake (cerchi)
    if "fake" in groups:
        for faketype, llist in groups["fake"].items():
            all_fake = np.concatenate(llist)
            x_fake = np.arange(len(all_fake))
            plt.scatter(
                x_fake, all_fake,
                s=60,               # punti grandi
                alpha=0.8,
                marker='o',         # cerchi
                label=f"fake_{faketype}"
            )

    plt.title(f"Scatterplot logits – step {step}")
    plt.xlabel("index")
    plt.ylabel("logit value")
    plt.legend(markerscale=1.5)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"logits_scatter_step_{step}.pdf")
    plt.savefig(out_path)
    plt.close()

    print(f"Saved PDF: {out_path}")
