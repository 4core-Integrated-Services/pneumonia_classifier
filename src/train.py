"""
Training Script for Pneumonia X-Ray Classifier
=================================================
Handles the full training loop with:
- Mixed precision training (AMP)
- Cosine annealing LR scheduler
- Early stopping on validation AUROC
- Class-weighted BCE loss for imbalanced data
- MLflow experiment tracking
- Reproducibility controls (seeds, deterministic ops)

Usage:
    python -m src.train --lr 1e-4 --epochs 25 --seed 42
"""

import argparse
import os
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
import mlflow
from tqdm import tqdm

from src.dataset import create_dataloaders
from src.model import build_model


# ─── Reproducibility ────────────────────────────────────────
def set_seed(seed: int = 42):
    """
    Set seeds for reproducibility across all libraries.

    NOTE: Even with all seeds set, CUDA operations can introduce
    nondeterminism from atomicAdd in cuDNN convolutions. We enable
    torch.backends.cudnn.deterministic to minimize this, but a small
    source of remaining nondeterminism is the parallel DataLoader
    workers (mitigated via worker_init_fn in dataset.py).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── Training Step ──────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
) -> dict:
    """Train for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()

        # Mixed precision forward pass
        with autocast():
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    # Compute epoch metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
    epoch_auroc = roc_auc_score(all_labels, all_preds)

    return {"loss": epoch_loss, "f1": epoch_f1, "auroc": epoch_auroc}


# ─── Validation Step ────────────────────────────────────────
@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> dict:
    """Evaluate on validation/test set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        with autocast():
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    epoch_loss = running_loss / len(loader.dataset)

    preds_binary = (all_preds > 0.5).astype(int)
    metrics = {
        "loss": epoch_loss,
        "f1": f1_score(all_labels, preds_binary),
        "auroc": roc_auc_score(all_labels, all_preds),
        "precision": precision_score(all_labels, preds_binary),
        "recall": recall_score(all_labels, preds_binary),  # Sensitivity
        "predictions": all_preds,
        "labels": all_labels,
    }

    return metrics


# ─── Main Training Loop ────────────────────────────────────
def train(args):
    """Full training pipeline."""
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Data ──
    loaders = create_dataloaders(
        root_dir=args.data_dir,
        img_size=224,
        batch_size=args.batch_size,
        num_workers=4,
        seed=args.seed,
    )

    # ── Model ──
    model = build_model(
        architecture="resnet18",
        pretrained=True,
        dropout=0.3,
        device=device,
    )

    # ── Loss with class weights ──
    # Positive weight = n_negative / n_positive to handle imbalance
    # Dataset: 1341 normal, 3875 pneumonia → pos_weight = 1341/3875 ≈ 0.346
    # But we WANT to catch pneumonia (high recall), so we use pos_weight > 1
    pos_weight = torch.tensor([1.94]).to(device)  # Upweights normal class
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # ── Scheduler ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Mixed Precision ──
    scaler = GradScaler()

    # ── MLflow Tracking ──
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("pneumonia-resnet18")

    with mlflow.start_run(run_name=f"lr{args.lr}_bs{args.batch_size}_seed{args.seed}"):
        mlflow.log_params({
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealing",
            "architecture": "resnet18",
            "pretrained": True,
            "dropout": 0.3,
            "pos_weight": 1.94,
        })

        # ── Training Loop ──
        best_auroc = 0.0
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            # Train
            train_metrics = train_one_epoch(
                model, loaders["train"], criterion, optimizer, scaler, device
            )

            # Validate
            val_metrics = validate(model, loaders["val"], criterion, device)

            # Step scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            elapsed = time.time() - start_time

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "train_f1": train_metrics["f1"],
                "train_auroc": train_metrics["auroc"],
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["f1"],
                "val_auroc": val_metrics["auroc"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "lr": current_lr,
            }, step=epoch)

            print(
                f"Epoch {epoch:02d}/{args.epochs} ({elapsed:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} F1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1']:.4f} "
                f"AUROC: {val_metrics['auroc']:.4f} Recall: {val_metrics['recall']:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # ── Early Stopping on AUROC ──
            if val_metrics["auroc"] > best_auroc:
                best_auroc = val_metrics["auroc"]
                best_epoch = epoch
                patience_counter = 0

                # Save best checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auroc": best_auroc,
                    "val_f1": val_metrics["f1"],
                }
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = f"checkpoints/best_model_seed{args.seed}.pt"
                torch.save(checkpoint, ckpt_path)
                print(f"  → Saved best model (AUROC: {best_auroc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(best AUROC: {best_auroc:.4f} at epoch {best_epoch})")
                    break

        # ── Final Test Evaluation ──
        print("\n" + "=" * 60)
        print("FINAL TEST EVALUATION")
        print("=" * 60)

        # Load best checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")

        test_metrics = validate(model, loaders["test"], criterion, device)

        # Confusion matrix
        preds_binary = (test_metrics["predictions"] > 0.5).astype(int)
        cm = confusion_matrix(test_metrics["labels"], preds_binary)
        report = classification_report(
            test_metrics["labels"], preds_binary,
            target_names=["NORMAL", "PNEUMONIA"],
        )

        print(f"\nTest Loss:      {test_metrics['loss']:.4f}")
        print(f"Test F1:        {test_metrics['f1']:.4f}")
        print(f"Test AUROC:     {test_metrics['auroc']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall:    {test_metrics['recall']:.4f}")
        print(f"\nConfusion Matrix:\n{cm}")
        print(f"\nClassification Report:\n{report}")

        # Log final test metrics
        mlflow.log_metrics({
            "test_loss": test_metrics["loss"],
            "test_f1": test_metrics["f1"],
            "test_auroc": test_metrics["auroc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
        })

        # Log final validation line for the form
        print(f"\n{'=' * 60}")
        print(f"FINAL VALIDATION LOG LINE:")
        print(f"Epoch {best_epoch} | val_loss={val_metrics['loss']:.4f} | "
              f"val_auroc={best_auroc:.4f} | val_f1={val_metrics['f1']:.4f}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pneumonia Classifier")
    parser.add_argument("--data_dir", type=str, default="./data/chest_xray")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
