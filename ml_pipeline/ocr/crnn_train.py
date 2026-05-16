"""
CRNN Training on RxHandBD Dataset
Architecture: CNN (5 blocks) → BiLSTM (2 layers, 256 hidden) → CTC Loss
Input:        128×32 grayscale word crops
Dataset:      RxHandBD Train_Set (4,463 images, 90/10 train/val split)
Output:       ml_pipeline/models/crnn_rxhandbd_best.pt + crnn_vocab.json
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import CTCLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────
ML_DIR    = Path(__file__).resolve().parent.parent          # ml_pipeline/
DATA_DIR  = ML_DIR / "data" / "rxhandbd"
TRAIN_CSV = DATA_DIR / "Train_Label.csv"
TRAIN_IMG = DATA_DIR / "Train_Set"
SAVE_DIR  = ML_DIR / "models"
SAVE_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────
IMG_H      = 32
IMG_W      = 128
BATCH_SIZE = 32
MAX_EPOCHS = 50
LR         = 1e-3
PATIENCE   = 7

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[CRNN] Device: {DEVICE}", flush=True)


# ── Vocabulary ─────────────────────────────────────────────────────────
def build_vocab(csv_path: Path):
    df = pd.read_csv(csv_path, header=0, names=["image", "label"])
    df["label"] = df["label"].astype(str).str.strip()
    all_chars = sorted(set("".join(df["label"].tolist())))
    vocab     = ["<blank>"] + all_chars        # index 0 = CTC blank
    char2idx  = {c: i for i, c in enumerate(vocab)}
    idx2char  = {i: c for i, c in enumerate(vocab)}
    return vocab, char2idx, idx2char, df


# ── Dataset ────────────────────────────────────────────────────────────
class RxHandBDDataset(Dataset):
    def __init__(self, df, img_dir, char2idx):
        self.df       = df.reset_index(drop=True)
        self.img_dir  = Path(img_dir)
        self.char2idx = char2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = str(row["label"])
        path  = self.img_dir / row["image"]

        try:
            img = Image.open(path).convert("L")
        except Exception:
            img = Image.new("L", (IMG_W, IMG_H), 255)

        img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5                             # normalize [-1, 1]
        tensor = torch.tensor(arr).unsqueeze(0)             # (1, H, W)

        encoded = torch.tensor([self.char2idx.get(c, 0) for c in label], dtype=torch.long)
        return tensor, encoded, len(encoded)


def collate_fn(batch):
    images, labels, lens = zip(*batch)
    return torch.stack(images), torch.cat(labels), torch.tensor(lens, dtype=torch.long)


# ── Model ──────────────────────────────────────────────────────────────
class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            # B1: 1→32, pool H/2 W/2
            nn.Conv2d(1, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            # B2: 32→64, pool H/2 W/2
            nn.Conv2d(32, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            # B3: 64→128, pool W/2 only
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            # B4: 128→128, pool W/2 only
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            # B5: 128→256, no pool
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        # H: 32→16→8→8→8  W: 128→64→32→16→8
        # adaptive pool collapses H to 1 for any input size
        self.adapt = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=256, hidden_size=256, num_layers=2,
            batch_first=False, bidirectional=True, dropout=0.3,
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):                      # x: (B,1,H,W)
        f = self.cnn(x)                        # (B,256,H',W')
        f = self.adapt(f).squeeze(2)           # (B,256,W')
        f = f.permute(2, 0, 1)                 # (T,B,256)
        out, _ = self.rnn(f)                   # (T,B,512)
        logits = self.fc(out)                  # (T,B,C)
        return torch.nn.functional.log_softmax(logits, dim=2)


# ── Decode / CER ───────────────────────────────────────────────────────
def greedy_decode(log_probs, idx2char):
    pred_idx = log_probs.argmax(dim=2)          # (T,B)
    results = []
    for b in range(pred_idx.shape[1]):
        seq = pred_idx[:, b].tolist()
        collapsed = [seq[0]] + [seq[i] for i in range(1, len(seq)) if seq[i] != seq[i-1]]
        chars = [idx2char[c] for c in collapsed if c != 0]
        results.append("".join(chars))
    return results


def char_error_rate(preds, targets):
    total_err, total_len = 0, 0
    for p, t in zip(preds, targets):
        # simple edit distance
        dp = list(range(len(t) + 1))
        for i, cp in enumerate(p):
            ndp = [i + 1]
            for j, ct in enumerate(t):
                ndp.append(min(dp[j] + (cp != ct), dp[j+1] + 1, ndp[-1] + 1))
            dp = ndp
        total_err += dp[len(t)]
        total_len += max(len(t), 1)
    return total_err / total_len


# ── Train ──────────────────────────────────────────────────────────────
def main():
    vocab, char2idx, idx2char, df = build_vocab(TRAIN_CSV)
    num_classes = len(vocab)
    print(f"[CRNN] Vocab: {num_classes} chars | e.g. {''.join(vocab[1:20])}...", flush=True)

    with open(SAVE_DIR / "crnn_vocab.json", "w") as f:
        json.dump({"vocab": vocab, "char2idx": char2idx, "idx2char": {str(k): v for k, v in idx2char.items()}}, f)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)
    print(f"[CRNN] Train: {len(train_df)} | Val: {len(val_df)}", flush=True)

    train_ds = RxHandBDDataset(train_df, TRAIN_IMG, char2idx)
    val_ds   = RxHandBDDataset(val_df,   TRAIN_IMG, char2idx)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model     = CRNN(num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    ctc_loss  = CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    best_cer, no_improve = float("inf"), 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──
        model.train()
        epoch_loss, t0 = 0.0, time.time()

        for images, labels_cat, label_lens in train_dl:
            images = images.to(DEVICE)
            optimizer.zero_grad()
            log_probs = model(images)              # (T,B,C) on DEVICE
            T, B, _   = log_probs.shape
            input_lens = torch.full((B,), T, dtype=torch.long)
            # CTC loss must run on CPU
            loss = ctc_loss(log_probs.cpu(), labels_cat, input_lens, label_lens)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_dl)

        # ── Validate ──
        model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, labels_cat, label_lens in val_dl:
                images    = images.to(DEVICE)
                log_probs = model(images)
                preds     = greedy_decode(log_probs, idx2char)
                all_preds.extend(preds)
                # decode targets
                offset = 0
                for l in label_lens.tolist():
                    chars = [idx2char[c] for c in labels_cat[offset:offset+l].tolist()]
                    all_targets.append("".join(chars))
                    offset += l

        val_cer = char_error_rate(all_preds, all_targets)
        scheduler.step(val_cer)
        elapsed = time.time() - t0

        print(f"[CRNN] Epoch {epoch:3d}/{MAX_EPOCHS} | Loss {epoch_loss:.4f} | Val CER {val_cer:.4f} | {elapsed:.0f}s", flush=True)

        if epoch % 5 == 0:
            for p, t in zip(all_preds[:3], all_targets[:3]):
                print(f"         pred='{p}'  target='{t}'", flush=True)

        if val_cer < best_cer:
            best_cer, no_improve = val_cer, 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_cer": val_cer, "num_classes": num_classes,
            }, SAVE_DIR / "crnn_rxhandbd_best.pt")
            print(f"[CRNN] ✓ Saved best model  CER={val_cer:.4f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[CRNN] Early stopping at epoch {epoch}", flush=True)
                break

    print(f"[CRNN] Done. Best Val CER: {best_cer:.4f}", flush=True)


if __name__ == "__main__":
    main()
