import numpy as np
d = np.load("data/reports/nn_dataset/windows_train.npz", allow_pickle=True)
print("X shape:", d["X"].shape)       # (N, 200, 11)
print("y_bin shape:", d["y_bin"].shape)  # (N,)
print("mask mean:", d["mask"].mean())    # доля не-цензурированных
print("unique y_bin:", np.unique(d["y_bin"])[:10], "...")