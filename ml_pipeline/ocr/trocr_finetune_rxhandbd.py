"""
trocr_finetune_rxhandbd.py — Phase 6

Fine-tune TrOCR decoder on RxHandBD word crops.
- Encoder (BEiT ViT-Large) : FROZEN  — saves ~1.3GB gradient memory
- Decoder (RoBERTa)         : TRAINED — learns BD brand name character patterns
- Device                    : MPS (Apple Silicon)
- Precision                 : float32 (MPS has no float16 backward pass)
- Batch size                : 4  + gradient accumulation 8 = effective batch 32
- LR                        : 5e-5 (small — decoder already pretrained)
- Epochs                    : 15 with early stopping (patience=4)

Expected training time: ~2-3 hours on M-series Mac
Expected outcome: raw CER drops ~35% → ~12-18% on BD names
"""

import sys, json, time
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

ML_DIR   = Path(__file__).resolve().parent.parent          # ml_pipeline/
DATA_DIR = ML_DIR / "data" / "rxhandbd"
SAVE_DIR = ML_DIR / "models"
SAVE_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────
TRAIN_CSV     = DATA_DIR / "Train_Label.csv"
TRAIN_IMG_DIR = DATA_DIR / "Train_Set"
MODEL_ID      = "microsoft/trocr-large-handwritten"
SAVE_PATH     = SAVE_DIR / "trocr_rxhandbd_finetuned"

DEVICE        = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE    = 4
GRAD_ACCUM    = 8       # effective batch = 32
LR            = 5e-5
MAX_EPOCHS    = 15
PATIENCE      = 4
IMG_H         = 384     # TrOCR optimal height

print(f"[Finetune] Device: {DEVICE}", flush=True)
print(f"[Finetune] Effective batch size: {BATCH_SIZE * GRAD_ACCUM}", flush=True)


# ── Dataset ────────────────────────────────────────────────────────────
class RxHandBDWordDataset(Dataset):
    def __init__(self, df, img_dir, processor):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = str(row["label"]).strip()
        path  = self.img_dir / row["image"]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (128, IMG_H), (255, 255, 255))

        # Resize to TrOCR input height preserving aspect ratio
        w, h  = img.size
        new_w = max(int(w * IMG_H / h), 64)
        img   = img.resize((new_w, IMG_H), Image.LANCZOS)

        # Processor encodes image + tokenizes label
        encoding = self.processor(
            images=img,
            text=label,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True,
        )

        pixel_values = encoding["pixel_values"].squeeze(0)   # (3, H, W)
        labels       = encoding["labels"].squeeze(0)          # (seq_len,)

        # Replace padding token id (-100 mask so loss ignores them)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels       = torch.stack([b["labels"]       for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}


# ── CER metric ─────────────────────────────────────────────────────────
def char_error_rate(preds, targets):
    total_err, total_len = 0, 0
    for p, t in zip(preds, targets):
        dp = list(range(len(t) + 1))
        for cp in p:
            ndp = [dp[0] + 1]
            for j, ct in enumerate(t):
                ndp.append(min(dp[j] + (cp != ct), dp[j+1] + 1, ndp[-1] + 1))
            dp = ndp
        total_err += dp[len(t)]
        total_len += max(len(t), 1)
    return total_err / max(total_len, 1)


# ── Train ──────────────────────────────────────────────────────────────
def main():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    # Load model + processor
    print(f"[Finetune] Loading {MODEL_ID}...", flush=True)
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model     = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

    # FREEZE encoder — only decoder gets gradients
    for param in model.encoder.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Finetune] Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)", flush=True)

    model.to(DEVICE)

    # Set required model config for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id

    # Data
    df = pd.read_csv(TRAIN_CSV, header=0, names=["image", "label"])
    df["label"] = df["label"].astype(str).str.strip()

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)
    print(f"[Finetune] Train: {len(train_df)} | Val: {len(val_df)}", flush=True)

    train_ds = RxHandBDWordDataset(train_df, TRAIN_IMG_DIR, processor)
    val_ds   = RxHandBDWordDataset(val_df,   TRAIN_IMG_DIR, processor)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=0, pin_memory=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )

    best_cer, no_improve = float("inf"), 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        epoch_loss, step = 0.0, 0
        t0 = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_dl):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels       = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss    = outputs.loss / GRAD_ACCUM      # scale for accumulation

            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM   # unscale for logging

            # Step optimizer every GRAD_ACCUM batches
            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                step += 1

        # Final step if leftover batches
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss /= len(train_dl)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in val_dl:
                pixel_values = batch["pixel_values"].to(DEVICE)
                labels       = batch["labels"]

                generated = model.generate(
                    pixel_values, max_length=32, num_beams=1
                )
                preds   = processor.batch_decode(generated, skip_special_tokens=True)
                targets = processor.batch_decode(
                    labels.masked_fill(labels == -100, processor.tokenizer.pad_token_id),
                    skip_special_tokens=True,
                )
                all_preds.extend(preds)
                all_targets.extend(targets)

        val_cer = char_error_rate(all_preds, all_targets)
        elapsed = time.time() - t0

        print(
            f"[Finetune] Epoch {epoch:2d}/{MAX_EPOCHS} | "
            f"Loss {epoch_loss:.4f} | Val CER {val_cer:.4f} | {elapsed:.0f}s",
            flush=True
        )

        # Sample predictions
        if epoch % 3 == 0:
            for p, t in zip(all_preds[:3], all_targets[:3]):
                print(f"           pred='{p}'  target='{t}'", flush=True)

        # Save best
        if val_cer < best_cer:
            best_cer, no_improve = val_cer, 0
            model.save_pretrained(str(SAVE_PATH))
            processor.save_pretrained(str(SAVE_PATH))
            print(f"[Finetune] ✓ Saved best model  CER={val_cer:.4f}  → {SAVE_PATH}", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[Finetune] Early stopping at epoch {epoch}", flush=True)
                break

    print(f"\n[Finetune] Done. Best Val CER: {best_cer:.4f}", flush=True)
    print(f"[Finetune] Fine-tuned model saved to: {SAVE_PATH}", flush=True)
    print(f"[Finetune] To use: TrOCRInference(model_path='{SAVE_PATH}')", flush=True)


if __name__ == "__main__":
    main()
