import numpy as np
d = np.load("data/reports/nn_dataset/windows_train.npz", allow_pickle=True)
dist = np.bincount(d["y_bin"], minlength=37)
print("Bin distribution:")
for b, c in enumerate(dist):
    label = "CENSORED" if b == 36 else f"[{b*5+1}-{b*5+5}]d"
    print(f"  {b:2d} ({label:>12}): {c:5d} ({c/len(d['y_bin']):.1%})")