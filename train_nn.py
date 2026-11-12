"""Train TCN model to predict time-to-next-CP (regression over 37 bins).

Архитектура (v2 — с "фильтрующими" слоями):
  GLU feature gating → SpatialDropout → TCN(dil 1/2/4/8)
  → SE block → GlobalAvgPool → Dropout → Linear(37)

Loss: Focal Loss (γ=2) с весами классов:
  - non-censored: 3.0 (нам важнее ловить CP)
  - censored:     1.0

Метрики (на non-censored):
  - Hit@5:       |τ_pred - τ_true| <= 5 дней
  - MedAE:       медианная абсолютная ошибка в днях
  - Precision@k: доля правильных среди k самых уверенных предсказаний
  - Censored_acc: доля правильных предсказаний "censored"
  - Coverage@80:  доля попаданий τ_true в 10–90 перцентили

Usage:
    python train_nn.py --epochs 30 --batch-size 64
    python train_nn.py --epochs 50 --device cpu --gamma 2.0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config

# --- Константы должны совпадать с build_nn_dataset.py ---
WINDOW = 200
BIN_SIZE = 5
N_BINS = 36
CENSORED_BIN = 36
N_CLASSES = 37
N_FEATURES = 11


# --------------------------------------------------------------------------- #
# Dataset (memmap, без загрузки X в RAM)
# --------------------------------------------------------------------------- #
class WindowDataset(Dataset):
    """Читает X.npy через mmap, не держит 4 GB в RAM."""

    def __init__(self, split_dir: Path):
        self.X = np.load(split_dir / "X.npy", mmap_mode="r")
        self.y_bin = np.load(split_dir / "y_bin.npy")
        self.y_tau = np.load(split_dir / "y_tau.npy")
        self.mask = np.load(split_dir / "mask.npy")
        self.n = len(self.y_bin)
        assert self.X.shape[0] == self.n, "X и y разной длины"

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        x = np.asarray(self.X[i], dtype=np.float32)   # [200, 11]
        x = np.transpose(x, (1, 0))                    # [11, 200] — для Conv1d
        return {
            "x": torch.from_numpy(x),
            "y_bin": int(self.y_bin[i]),
            "y_tau": int(self.y_tau[i]),
            "mask": int(self.mask[i]),
        }


# --------------------------------------------------------------------------- #
# Модель
# --------------------------------------------------------------------------- #
class GLU(nn.Module):
    """Feature gating: учит веса входных признаков."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class CausalConv1d(nn.Module):
    """Conv1d с causal padding (без утечки будущего)."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class TCNBlock(nn.Module):
    """Residual block TCN с двумя causal conv."""
    def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, 3, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, 3, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.drop(F.relu(self.bn2(self.conv2(out))))
        return out + self.res(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: учит веса временных каналов."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        s = x.mean(dim=-1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s.unsqueeze(-1)


class CPModel(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_CLASSES,
                 channels: tuple[int, ...] = (32, 64, 64, 64)):
        super().__init__()
        self.glu = GLU(n_features, channels[0])
        self.spatial_drop = nn.Dropout1d(0.3)
        self.blocks = nn.ModuleList()
        prev = channels[0]
        for i, ch in enumerate(channels):
            self.blocks.append(TCNBlock(prev, ch, dilation=2 ** i))
            prev = ch
        self.se = SEBlock(channels[-1])
        self.drop = nn.Dropout(0.4)
        self.head = nn.Linear(channels[-1], n_classes)

    def forward(self, x):
        # x: [B, F, T] → [B, T, F] для GLU
        x = x.transpose(1, 2)
        x = self.glu(x)                # [B, T, C0]
        x = x.transpose(1, 2)          # [B, C0, T]
        x = self.spatial_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.se(x)
        x = x.mean(dim=-1)             # global avg pool → [B, C_last]
        x = self.drop(x)
        return self.head(x)            # [B, N_CLASSES]


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    """Focal loss: снижает вклад "лёгких" примеров, фокусируется на сложных."""
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _tau_center(bin_idx: np.ndarray) -> np.ndarray:
    """Центр бина в днях. Например, bin=0 → 3 дня (центр [1..5])."""
    return bin_idx.astype(np.float64) * BIN_SIZE + BIN_SIZE / 2.0


def evaluate(model, loader, device: str):
    model.eval()
    proba_all, y_bin_all, y_tau_all, mask_all = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            logits = model(x)
            proba_all.append(F.softmax(logits, dim=-1).cpu().numpy())
            y_bin_all.append(batch["y_bin"].numpy())
            y_tau_all.append(batch["y_tau"].numpy())
            mask_all.append(batch["mask"].numpy())

    proba = np.concatenate(proba_all)
    y_bin = np.concatenate(y_bin_all)
    y_tau = np.concatenate(y_tau_all)
    mask = np.concatenate(mask_all)

    pred_bin = proba.argmax(axis=1)
    confidence = proba.max(axis=1)

    nc = (mask == 1)
    metrics: dict = {
        "n_windows": int(len(y_bin)),
        "n_non_censored": int(nc.sum()),
        "censored_fraction": float(1 - nc.mean()) if len(nc) else 0.0,
    }

    # Основные метрики — только на non-censored
    if nc.sum() > 0:
        tau_pred = _tau_center(pred_bin[nc])
        tau_true = y_tau[nc].astype(np.float64)
        err = np.abs(tau_pred - tau_true)
        metrics["hit5"] = float((err <= 5).mean())
        metrics["hit10"] = float((err <= 10).mean())
        metrics["medae"] = float(np.median(err))
        metrics["mae"] = float(np.mean(err))
    else:
        metrics.update({"hit5": 0.0, "hit10": 0.0, "medae": 0.0, "mae": 0.0})

    # Censored accuracy
    censored = (y_bin == CENSORED_BIN)
    metrics["censored_accuracy"] = (
        float((pred_bin[censored] == CENSORED_BIN).mean())
        if censored.sum() else 0.0
    )

    # Precision@k среди всех окон (по уверенности)
    order = np.argsort(-confidence)
    for k in (4, 8, 12, 50, 100):
        idx = order[:k]
        correct = 0
        for i in idx:
            if mask[i] == 1 and abs(_tau_center(pred_bin[i]) - y_tau[i]) <= 5:
                correct += 1
        metrics[f"precision@{k}"] = correct / len(idx) if len(idx) else 0.0

    # Coverage@80: true τ попадает в [p10, p90]?
    if nc.sum() > 0:
        cum = np.cumsum(proba[nc], axis=1)
        p10 = np.argmax(cum >= 0.10, axis=1)
        p90 = np.argmax(cum >= 0.90, axis=1)
        y_tau_bin = ((y_tau[nc] - 1) // BIN_SIZE).astype(int)
        covered = (y_tau_bin >= p10) & (y_tau_bin <= p90)
        metrics["coverage@80"] = float(covered.mean())
    else:
        metrics["coverage@80"] = 0.0

    return metrics


# --------------------------------------------------------------------------- #
# Обучение
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=config.REPORTS_DIR / "nn_dataset")
    parser.add_argument("--out-dir", type=Path,
                        default=config.REPORTS_DIR / "nn_training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10,
                        help="early stopping по val MedAE")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--censored-weight", type=float, default=1.0)
    parser.add_argument("--non-censored-weight", type=float, default=3.0)
    parser.add_argument("--device", default="auto",
                        choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Output: {args.out_dir}")

    # --- Данные ---
    train_ds = WindowDataset(args.data_dir / "train")
    val_ds = WindowDataset(args.data_dir / "val")
    test_ds = WindowDataset(args.data_dir / "test")
    print(f"Train: {len(train_ds):>7,} | Val: {len(val_ds):>6,} | Test: {len(test_ds):>6,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 4,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 4,
                             shuffle=False, num_workers=0)

    # --- Baseline: константное предсказание медианы τ ---
    y_tau_train = train_ds.y_tau[train_ds.mask == 1]
    median_tau = float(np.median(y_tau_train)) if len(y_tau_train) else 90.0
    print(f"\nBaseline (constant τ={median_tau:.1f}d):")
    baseline_hits = 0
    baseline_n = 0
    for i in range(len(test_ds)):
        if test_ds.mask[i] == 1:
            baseline_n += 1
            if abs(median_tau - test_ds.y_tau[i]) <= 5:
                baseline_hits += 1
    baseline_hit5 = baseline_hits / baseline_n if baseline_n else 0.0
    print(f"  Test Hit@5 = {baseline_hit5:.3f} (naive baseline)")

    # --- Модель ---
    model = CPModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {n_params:,}")

    class_weights = np.ones(N_CLASSES, dtype=np.float32)
    class_weights[:CENSORED_BIN] = args.non_censored_weight
    class_weights[CENSORED_BIN] = args.censored_weight
    class_weights = torch.from_numpy(class_weights).to(device)

    loss_fn = FocalLoss(alpha=class_weights, gamma=args.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Обучение ---
    print("\n=== Training ===")
    history = []
    best_val_medae = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y_bin"].to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:2d} | loss={train_loss:.4f} | "
              f"val Hit@5={val_metrics['hit5']:.3f} "
              f"MedAE={val_metrics['medae']:5.1f}d "
              f"cens_acc={val_metrics['censored_accuracy']:.3f} "
              f"P@50={val_metrics['precision@50']:.3f} "
              f"| {elapsed:4.0f}s")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val": val_metrics, "elapsed_s": elapsed})

        # Early stopping по MedAE
        current = val_metrics["medae"] if val_metrics["medae"] > 0 else 1e9
        if current < best_val_medae - 0.1:
            best_val_medae = current
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "median_tau_baseline": median_tau,
            }, args.out_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no MedAE improvement for {args.patience} epochs)")
                break

    # --- Test ---
    print("\n=== Test (best model) ===")
    ckpt = torch.load(args.out_dir / "best_model.pt", map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device)

    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:22s}: {v:.4f}")
        else:
            print(f"  {k:22s}: {v}")

    test_metrics["baseline_hit5"] = baseline_hit5
    test_metrics["baseline_median_tau"] = median_tau
    test_metrics["best_epoch"] = ckpt["epoch"]

    # --- Сохранение ---
    (args.out_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nBest model: {args.out_dir / 'best_model.pt'}")
    print(f"History:    {args.out_dir / 'history.json'}")
    print(f"Metrics:    {args.out_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()