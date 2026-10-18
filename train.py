import os
import sys
import time
import datetime
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import config
from tokenizer import CTCLabelConverter
from dataset import PersianOCRDataset, ocr_collate_fn
from model import MobileOCRNet

def compute_accuracy(preds, targets):
    """
    Computes exact match sequence accuracy and character-level accuracy.
    """
    exact_matches = 0
    total_samples = len(preds)

    total_chars = 0
    char_matches = 0

    for pred, gt in zip(preds, targets):
        if pred == gt:
            exact_matches += 1

        min_len = min(len(pred), len(gt))
        for p_c, g_c in zip(pred[:min_len], gt[:min_len]):
            if p_c == g_c:
                char_matches += 1
        total_chars += max(len(pred), len(gt), 1)

    seq_acc = exact_matches / max(total_samples, 1)
    char_acc = char_matches / max(total_chars, 1)
    return seq_acc, char_acc

def evaluate(model, val_loader, converter, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, input_lengths, target_lengths, raw_labels in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            log_probs = model(images)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if not torch.isnan(loss) and not torch.isinf(loss):
                val_loss += loss.item()

            preds_indices = log_probs.permute(1, 0, 2).argmax(dim=2).cpu().numpy()
            decoded_texts = converter.decode_greedy(preds_indices)

            all_preds.extend(decoded_texts)
            all_targets.extend(raw_labels)

    avg_loss = val_loss / max(len(val_loader), 1)
    seq_acc, char_acc = compute_accuracy(all_preds, all_targets)
    return avg_loss, seq_acc, char_acc, all_preds, all_targets

def init_mlflow_tracker(tracking_uri: str, experiment_name: str, run_name: str = None):
    """
    Safely initializes MLflow tracking with the remote server.
    Returns (mlflow_module, active_run) or (None, None) if unavailable.
    """
    try:
        import mlflow
        # Ensure URI starts with http:// if not specified
        if not tracking_uri.startswith("http://") and not tracking_uri.startswith("https://"):
            tracking_uri = f"http://{tracking_uri}"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        if not run_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"mobile_ocr_{timestamp}"

        active_run = mlflow.start_run(run_name=run_name)
        print(f"[+] MLflow Connected Successfully!")
        print(f"    - Tracking Server : {tracking_uri}")
        print(f"    - Experiment Name : {experiment_name}")
        print(f"    - Run Name        : {active_run.info.run_name}")
        print(f"    - Run ID          : {active_run.info.run_id}")
        return mlflow, active_run
    except ImportError:
        print("[!] Warning: 'mlflow' package is not installed. Run 'pip install mlflow' to enable tracking.")
        return None, None
    except Exception as e:
        print(f"[!] Warning: Could not connect to MLflow server at {tracking_uri}: {e}")
        print("    --> Training will continue locally without MLflow logging.")
        return None, None

def train(
    epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    use_mlflow: bool = config.MLFLOW_ENABLE,
    mlflow_uri: str = config.MLFLOW_TRACKING_URI,
    experiment_name: str = config.MLFLOW_EXPERIMENT_NAME,
    run_name: str = None
):
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    device = torch.device(config.DEVICE)
    print("=" * 65)
    print("  Iranian National ID Card OCR - Lightweight Training")
    print("=" * 65)
    print(f"[*] Device: {device}")
    print(f"[*] Hyperparameters: Epochs={epochs}, BatchSize={batch_size}, LR={lr}")

    # Datasets
    train_dataset = PersianOCRDataset(config.TRAIN_LABEL_FILE, is_train=True)
    val_dataset = PersianOCRDataset(config.VAL_LABEL_FILE, is_train=False)

    if len(train_dataset) == 0:
        print("[!] Error: Train dataset is empty. Run prepare_dataset.py first!")
        return

    if len(val_dataset) == 0:
        print("[*] Note: Validation set is empty, using training set for evaluation metrics.")
        val_dataset = train_dataset

    converter = CTCLabelConverter(config.ALL_CHARS)
    print(f"[*] Vocabulary size (including CTC blank): {converter.num_classes}")

    collate = partial(ocr_collate_fn, converter=converter)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0
    )

    # Initialize model
    model = MobileOCRNet(num_classes=converter.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)
    print(f"[*] MobileOCRNet initialized. Trainable parameters: {total_params:,} ({model_size_mb:.2f} MB)")

    criterion = nn.CTCLoss(blank=converter.blank_idx, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Initialize MLflow if requested
    mlflow_module = None
    if use_mlflow:
        mlflow_module, _ = init_mlflow_tracker(
            tracking_uri=mlflow_uri,
            experiment_name=experiment_name,
            run_name=run_name
        )
        if mlflow_module:
            try:
                mlflow_module.log_params({
                    "model_architecture": "MobileOCRNet",
                    "model_params": total_params,
                    "model_size_mb": round(model_size_mb, 2),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "initial_learning_rate": lr,
                    "weight_decay": config.WEIGHT_DECAY,
                    "img_height": config.IMG_HEIGHT,
                    "min_width": config.MIN_WIDTH,
                    "max_width": config.MAX_WIDTH,
                    "device": str(device),
                    "vocab_size": converter.num_classes,
                    "train_samples_count": len(train_dataset),
                    "val_samples_count": len(val_dataset),
                    "optimizer": "AdamW",
                    "scheduler": "CosineAnnealingLR"
                })
            except Exception as e:
                print(f"[!] Warning: Could not log parameters to MLflow: {e}")

    best_val_acc = 0.0
    best_char_acc = 0.0
    best_model_path = os.path.join(config.CHECKPOINTS_DIR, "best_mobile_ocr.pth")
    last_sample_preds = []
    last_sample_gts = []

    print("\n[*] Starting training loop...")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch_idx, (images, targets, input_lengths, target_lengths, _) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            log_probs = model(images)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_loss = train_loss / max(len(train_loader), 1)

        # Validation
        val_loss, seq_acc, char_acc, sample_preds, sample_gts = evaluate(
            model, val_loader, converter, criterion, device
        )
        last_sample_preds = sample_preds
        last_sample_gts = sample_gts
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Seq Acc: {seq_acc * 100:.2f}% | Char Acc: {char_acc * 100:.2f}% | "
              f"LR: {current_lr:.6f}")

        # Sample comparison
        if len(sample_preds) > 0 and (epoch % 5 == 0 or epoch == epochs):
            for s_idx in range(min(3, len(sample_preds))):
                print(f"    Sample #{s_idx + 1} -> GT: '{sample_gts[s_idx]}' | Pred: '{sample_preds[s_idx]}'")

        # MLflow epoch metrics logging
        if mlflow_module:
            try:
                mlflow_module.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_seq_acc": seq_acc,
                    "val_char_acc": char_acc,
                    "learning_rate": current_lr,
                    "epoch_duration_sec": elapsed
                }, step=epoch)
            except Exception as e:
                print(f"[!] Warning: Failed to log epoch {epoch} to MLflow: {e}")

        # Save best model
        if seq_acc >= best_val_acc:
            best_val_acc = seq_acc
            best_char_acc = char_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': seq_acc,
                'char_acc': char_acc,
                'vocab': converter.character
            }, best_model_path)
            print(f"    --> Saved best model checkpoint to {best_model_path}")

    print("-" * 65)
    print(f"[+] Training complete!")
    print(f"    - Best Validation Sequence Accuracy : {best_val_acc * 100:.2f}%")
    print(f"    - Best Validation Character Accuracy: {best_char_acc * 100:.2f}%")

    # Post-training MLflow artifact logging and export
    if mlflow_module:
        try:
            print("\n[*] Uploading model artifacts and metrics to MLflow...")
            mlflow_module.log_metrics({
                "best_val_seq_acc": best_val_acc,
                "best_val_char_acc": best_char_acc
            })

            # 1. Log best PyTorch checkpoint
            if os.path.exists(best_model_path):
                mlflow_module.log_artifact(best_model_path, artifact_path="checkpoints")

            # 2. Export and log mobile models (TorchScript & ONNX)
            try:
                from export_model import export_models
                export_models(best_model_path)
                mobile_pt = os.path.join(config.CHECKPOINTS_DIR, "model_mobile.pt")
                vocab_txt = os.path.join(config.CHECKPOINTS_DIR, "vocab.txt")
                onnx_file = os.path.join(config.CHECKPOINTS_DIR, "model.onnx")

                if os.path.exists(mobile_pt):
                    mlflow_module.log_artifact(mobile_pt, artifact_path="mobile_deployment")
                if os.path.exists(vocab_txt):
                    mlflow_module.log_artifact(vocab_txt, artifact_path="mobile_deployment")
                if os.path.exists(onnx_file):
                    mlflow_module.log_artifact(onnx_file, artifact_path="mobile_deployment")
            except Exception as e:
                print(f"[!] Note: Automatic mobile export encountered: {e}")

            # 3. Log sample predictions report
            if last_sample_preds:
                samples_report = os.path.join(config.CHECKPOINTS_DIR, "sample_predictions.txt")
                with open(samples_report, "w", encoding="utf-8") as f:
                    f.write("Sample Predictions at End of Training\n")
                    f.write("=" * 60 + "\n")
                    for i, (gt, pr) in enumerate(zip(last_sample_gts[:50], last_sample_preds[:50])):
                        status = "EXACT MATCH" if gt == pr else "MISMATCH"
                        f.write(f"[{i+1:02d}] [{status}] GT: '{gt}' | Pred: '{pr}'\n")
                mlflow_module.log_artifact(samples_report, artifact_path="reports")

            mlflow_module.end_run()
            print(f"[+] MLflow run successfully finalized and closed.")
        except Exception as e:
            print(f"[!] Warning: Error finalizing MLflow run: {e}")

    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lightweight Persian OCR model with MLflow tracking")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--mlflow_uri", type=str, default=config.MLFLOW_TRACKING_URI, help="MLflow Tracking Server URI (default: http://172.16.101.42:8082)")
    parser.add_argument("--experiment_name", type=str, default=config.MLFLOW_EXPERIMENT_NAME, help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name (optional)")
    parser.add_argument("--no_mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_mlflow=not args.no_mlflow,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name
    )
