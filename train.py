import os
import sys
import time
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

        # Character level overlap
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

            # Forward pass: log_probs has shape [T, B, C]
            log_probs = model(images)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if not torch.isnan(loss) and not torch.isinf(loss):
                val_loss += loss.item()

            # Decode predictions
            preds_indices = log_probs.permute(1, 0, 2).argmax(dim=2).cpu().numpy()
            decoded_texts = converter.decode_greedy(preds_indices)

            all_preds.extend(decoded_texts)
            all_targets.extend(raw_labels)

    avg_loss = val_loss / max(len(val_loader), 1)
    seq_acc, char_acc = compute_accuracy(all_preds, all_targets)
    return avg_loss, seq_acc, char_acc, all_preds, all_targets

def train():
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    device = torch.device(config.DEVICE)
    print(f"[*] Training on device: {device}")

    converter = CTCLabelConverter(config.ALL_CHARS)
    print(f"[*] Vocabulary size (including CTC blank): {converter.num_classes}")

    collate = partial(ocr_collate_fn, converter=converter)

    # Datasets and Loaders
    train_dataset = PersianOCRDataset(config.TRAIN_LABEL_FILE, is_train=True)
    val_dataset = PersianOCRDataset(config.VAL_LABEL_FILE, is_train=False)

    if len(train_dataset) == 0:
        print("[!] Train dataset is empty. Run prepare_dataset.py first!")
        return

    if len(val_dataset) == 0:
        print("[*] Note: Validation set is empty, using training set for evaluation metrics.")
        val_dataset = train_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=0
    )

    # Initialize lightweight model (~4MB)
    model = MobileOCRNet(num_classes=converter.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] MobileOCRNet initialized. Trainable parameters: {total_params:,} ({total_params*4/(1024*1024):.2f} MB)")

    criterion = nn.CTCLoss(blank=converter.blank_idx, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    best_model_path = os.path.join(config.CHECKPOINTS_DIR, "best_mobile_ocr.pth")

    print("[*] Starting training...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch_idx, (images, targets, input_lengths, target_lengths, _) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            log_probs = model(images)  # [T, B, C]

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss = train_loss / max(len(train_loader), 1)

        # Validation
        val_loss, seq_acc, char_acc, sample_preds, sample_gts = evaluate(
            model, val_loader, converter, criterion, device
        )
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{config.NUM_EPOCHS:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Seq Acc: {seq_acc * 100:.2f}% | Char Acc: {char_acc * 100:.2f}%")

        # Print a sample comparison
        if len(sample_preds) > 0 and epoch % 5 == 0:
            for s_idx in range(min(3, len(sample_preds))):
                print(f"    Sample #{s_idx + 1} -> GT: '{sample_gts[s_idx]}' | Pred: '{sample_preds[s_idx]}'")

        # Save best model based on sequence accuracy
        if seq_acc >= best_val_acc:
            best_val_acc = seq_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': seq_acc,
                'char_acc': char_acc,
                'vocab': converter.character
            }, best_model_path)
            print(f"    --> Saved best model checkpoint to {best_model_path}")

    print(f"\n[+] Training complete! Best validation sequence accuracy: {best_val_acc * 100:.2f}%")

if __name__ == "__main__":
    train()
