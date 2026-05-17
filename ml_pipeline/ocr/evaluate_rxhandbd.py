"""
evaluate_rxhandbd.py — Phase 5C

Evaluates TrOCR + RapidFuzz pipeline on RxHandBD Test_Set.
Test images are already word-level crops (exactly what we feed to TrOCR).

Reports:
  - Raw TrOCR accuracy      (before fuzzy correction)
  - Post-correction accuracy (after RapidFuzz vs RxHandBD labels)
  - Top-3 accuracy           (true label in top 3 matches)
  - Confusion examples        (worst mismatches)

Run: python3 ml_pipeline/ocr/evaluate_rxhandbd.py
"""

import sys, time
from pathlib import Path
from PIL import Image
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # ml_pipeline/

from ocr.trocr_inference import TrOCRInference, preprocess_for_trocr, load_medicine_list

try:
    from rapidfuzz import process, fuzz
    HAS_FUZZ = True
except ImportError:
    HAS_FUZZ = False
    print("WARNING: rapidfuzz not found — only raw accuracy will be reported")

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).resolve().parent.parent / "data" / "rxhandbd"
TEST_CSV  = DATA_DIR / "Test_Label.csv"
TEST_IMGS = DATA_DIR / "Test_Set"

# ── Load ───────────────────────────────────────────────────────────────
print("[Eval] Loading TrOCR engine...", flush=True)
engine = TrOCRInference(model_path="ml_pipeline/models/trocr_rxhandbd_finetuned")

print("[Eval] Loading medicine vocabulary...", flush=True)
BD_MEDICINES = load_medicine_list()

test_df = pd.read_csv(TEST_CSV, header=0, names=["image", "label"])
test_df["label"] = test_df["label"].astype(str).str.strip()
print(f"[Eval] Test samples: {len(test_df)}", flush=True)

# ── Evaluate ───────────────────────────────────────────────────────────
raw_correct    = 0   # TrOCR output exactly matches label (case-insensitive)
fuzz_correct   = 0   # after RapidFuzz correction matches label
top3_correct   = 0   # true label in top-3 RapidFuzz matches
total          = 0
errors         = []  # (true_label, raw_pred, fuzz_pred)

t_start = time.time()

for idx, row in test_df.iterrows():
    img_path = TEST_IMGS / row["image"]
    true_label = row["label"]

    if not img_path.exists():
        continue

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        continue

    # Preprocess (same pipeline as predict_prescription)
    processed = preprocess_for_trocr(img)

    # TrOCR on this word crop directly
    raw_pred = engine._infer_line(processed)

    # RapidFuzz correction
    fuzz_pred = raw_pred
    top3 = []
    if HAS_FUZZ and raw_pred:
        results = process.extract(raw_pred, BD_MEDICINES, scorer=fuzz.ratio, limit=3)
        if results and results[0][1] >= 82:
            fuzz_pred = results[0][0]
        top3 = [r[0] for r in results]

    # Score
    raw_match  = raw_pred.lower().strip() == true_label.lower().strip()
    fuzz_match = fuzz_pred.lower().strip() == true_label.lower().strip()
    top3_match = true_label.lower().strip() in [t.lower() for t in top3]

    raw_correct  += raw_match
    fuzz_correct += fuzz_match
    top3_correct += top3_match
    total        += 1

    if not fuzz_match:
        errors.append((true_label, raw_pred, fuzz_pred))

    # Progress every 100 samples
    if (total % 100) == 0:
        elapsed = time.time() - t_start
        print(
            f"[Eval] {total}/{len(test_df)} | "
            f"Raw: {raw_correct/total*100:.1f}% | "
            f"Fuzz: {fuzz_correct/total*100:.1f}% | "
            f"{elapsed:.0f}s elapsed",
            flush=True
        )

# ── Results ────────────────────────────────────────────────────────────
elapsed = time.time() - t_start

print("\n" + "="*55)
print("  RxHandBD Test Set Evaluation Results")
print("="*55)
print(f"  Total samples evaluated : {total}")
print(f"  Time elapsed            : {elapsed:.0f}s")
print(f"  Raw TrOCR accuracy      : {raw_correct/total*100:.2f}%")
print(f"  Post-RapidFuzz accuracy : {fuzz_correct/total*100:.2f}%")
print(f"  Top-3 accuracy          : {top3_correct/total*100:.2f}%")
print("="*55)

print(f"\nWorst mismatches (first 15):")
print(f"{'True Label':<20} {'Raw TrOCR':<20} {'Fuzz Result':<20}")
print("-"*60)
for true, raw, fuzz_res in errors[:15]:
    print(f"{true:<20} {raw:<20} {fuzz_res:<20}")
