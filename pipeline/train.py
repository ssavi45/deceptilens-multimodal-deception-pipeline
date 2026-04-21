"""
train.py — Training loop with early stopping, checkpointing, and plotting.

Supports:
  - Full training with CosineAnnealingLR + label smoothing
  - Early stopping on validation F1
  - Gradient clipping (max_norm=1.0)
  - WeightedRandomSampler for class imbalance
  - Few-shot fine-tuning (freeze encoders, train classifier only)
  - Training curve visualization (loss, accuracy, F1/AUC)
"""

import os
import copy
import json
import shutil
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.config import (
    MODEL_PATH, MODEL_DIR, XAI_DIR, DEVICE,
    VIDEO_INPUT_DIM, AUDIO_INPUT_DIM, FUSION_DIM, HIDDEN_DIM,
    NUM_CLASSES, DROPOUT,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE,
    RANDOM_SEED, LABEL_SMOOTH,
)
from pipeline.dataset import (
    load_features, split_and_normalize, make_loader,
)
from pipeline.model import DeceptionDetector, count_parameters

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
#  Single epoch helpers
# ────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: str,
) -> dict:
    """Run one training epoch. Returns {loss, acc, f1}."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for video, audio, labels in loader:
        video  = video.to(device)
        audio  = audio.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _, _ = model(video, audio)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / n
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"loss": avg_loss, "acc": acc, "f1": f1}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: str,
) -> dict:
    """Evaluate model on a DataLoader. Returns {loss, acc, f1, auc}."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for video, audio, labels in loader:
        video  = video.to(device)
        audio  = audio.to(device)
        labels = labels.to(device)

        logits, _, _ = model(video, audio)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / n
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # only one class present

    return {"loss": avg_loss, "acc": acc, "f1": f1, "auc": auc}


# ────────────────────────────────────────────────────────────────────────
#  Full training loop
# ────────────────────────────────────────────────────────────────────────

def train(
    feat_csv: str,
    run_name: str = "baseline",
    device_str: str = DEVICE,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    balance: bool = True,
    label_smooth: float = LABEL_SMOOTH,
    model_save_path: str | None = None,
    scaler_save_path: str | None = None,
) -> dict:
    """
    Full training pipeline:
      1. Load features from CSV
      2. Split 70/15/15 stratified
      3. Fit StandardScaler on train
      4. Build DataLoaders (WeightedRandomSampler on train)
      5. Train with CosineAnnealingLR + early stopping on val F1
      6. Save best checkpoint + training curves

    Returns dict with test metrics + history.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    save_path = model_save_path or MODEL_PATH

    print(f"\n{'='*60}")
    print(f"  Training: {run_name}")
    print(f"  Device: {device_str}")
    print(f"{'='*60}")

    # ── Load & split ────────────────────────────────────────────────
    X, y = load_features(feat_csv)
    data = split_and_normalize(X, y, fit_scaler=True, scaler_save_path=scaler_save_path)

    print(f"  Train: {len(data['y_train'])}  Val: {len(data['y_val'])}  "
          f"Test: {len(data['y_test'])}")
    print(f"  Class distribution (train): "
          f"0={np.sum(data['y_train']==0)}, 1={np.sum(data['y_train']==1)}")

    train_loader = make_loader(data["X_train"], data["y_train"],
                               BATCH_SIZE, shuffle=True, balance=balance)
    val_loader   = make_loader(data["X_val"], data["y_val"],
                               BATCH_SIZE, shuffle=False, balance=False)
    test_loader  = make_loader(data["X_test"], data["y_test"],
                               BATCH_SIZE, shuffle=False, balance=False)

    # ── Model, optimizer, scheduler ─────────────────────────────────
    device = torch.device(device_str)
    model = DeceptionDetector().to(device)
    print(f"  Parameters: {count_parameters(model):,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # ── Training loop ───────────────────────────────────────────────
    best_val_f1 = 0.0
    best_state  = None
    wait = 0

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "val_auc": [], "lr": [],
    }

    pbar = tqdm(range(1, epochs + 1), desc=f"{run_name} training")
    for epoch in pbar:
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device_str)
        val_metrics   = evaluate(model, val_loader, criterion, device_str)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])
        history["lr"].append(current_lr)

        pbar.set_postfix({
            "t_loss": f"{train_metrics['loss']:.3f}",
            "v_f1":   f"{val_metrics['f1']:.3f}",
            "v_auc":  f"{val_metrics['auc']:.3f}",
            "lr":     f"{current_lr:.1e}",
        })

        # Early stopping on val F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n  ⏹ Early stopping at epoch {epoch} (patience={patience})")
                break

    # ── Restore best & evaluate on test ─────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), save_path)
    print(f"✓ Model saved to {save_path}")

    test_metrics = evaluate(model, test_loader, criterion, device_str)
    print(f"\n  Test Results ({run_name}):")
    print(f"    Accuracy: {test_metrics['acc']:.4f}")
    print(f"    F1 (macro): {test_metrics['f1']:.4f}")
    print(f"    AUC-ROC: {test_metrics['auc']:.4f}")

    # ── Save history ────────────────────────────────────────────────
    history_path = os.path.join(MODEL_DIR, f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Plot curves ─────────────────────────────────────────────────
    plot_training_curves(history, run_name)

    return {
        "run_name": run_name,
        "test_acc": test_metrics["acc"],
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["auc"],
        "best_val_f1": best_val_f1,
        "epochs_run": len(history["train_loss"]),
        "history": history,
        "data": data,
    }


# ────────────────────────────────────────────────────────────────────────
#  Few-shot fine-tuning
# ────────────────────────────────────────────────────────────────────────

def finetune_few_shot(
    model_path: str,
    X_few: np.ndarray,
    y_few: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device_str: str = DEVICE,
    epochs_ft: int = 30,
    lr_ft: float = 1e-4,
) -> dict:
    """
    Load a pre-trained model, freeze video_encoder + audio_encoder,
    fine-tune only the classifier head on the few-shot samples.

    Returns dict with test metrics.
    """
    device = torch.device(device_str)
    model = DeceptionDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Freeze encoders
    for param in model.video_encoder.parameters():
        param.requires_grad = False
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    for param in model.cross_attn_v2a.parameters():
        param.requires_grad = False
    for param in model.cross_attn_a2v.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Few-shot fine-tuning: {trainable:,} trainable params "
          f"(classifier only), {len(y_few)} samples")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_ft, weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    few_loader  = make_loader(X_few, y_few, batch_size=min(16, len(y_few)),
                              shuffle=True, balance=True)
    test_loader = make_loader(X_test, y_test, batch_size=BATCH_SIZE,
                              shuffle=False, balance=False)

    best_loss = float("inf")
    best_state = None

    for epoch in tqdm(range(1, epochs_ft + 1), desc="Few-shot fine-tune"):
        train_metrics = train_epoch(model, few_loader, optimizer, criterion, device_str)
        if train_metrics["loss"] < best_loss:
            best_loss = train_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    save_path = model_path.replace(".pth", "_finetuned.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✓ Fine-tuned model saved to {save_path}")

    test_metrics = evaluate(model, test_loader, criterion, device_str)
    print(f"  Few-shot Test: Acc={test_metrics['acc']:.4f}  "
          f"F1={test_metrics['f1']:.4f}  AUC={test_metrics['auc']:.4f}")

    return test_metrics


# ────────────────────────────────────────────────────────────────────────
#  Training curves
# ────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, run_name: str) -> str:
    """
    Create a 3-panel figure: Loss, Accuracy, F1/AUC.
    Saves to XAI_DIR and returns the path.
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Curves — {run_name}", fontsize=14, fontweight="bold")

    # Panel 1: Loss
    axes[0].plot(epochs_range, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs_range, history["val_loss"], label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Accuracy
    axes[1].plot(epochs_range, history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(epochs_range, history["val_acc"], label="Val", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: F1 + AUC
    axes[2].plot(epochs_range, history["train_f1"], label="Train F1", linewidth=2)
    axes[2].plot(epochs_range, history["val_f1"], label="Val F1", linewidth=2)
    axes[2].plot(epochs_range, history["val_auc"], label="Val AUC",
                 linewidth=2, linestyle="--")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_title("F1 / AUC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(XAI_DIR, f"{run_name}_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Training curves saved to {save_path}")
    return save_path
