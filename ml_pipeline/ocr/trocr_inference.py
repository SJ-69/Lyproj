"""
TrOCR Inference Module — uses microsoft/trocr-large-handwritten.

Pipeline:
  1. CV Preprocessing   (deskew → denoise → adaptive threshold → erosion/dilation)
  2. Header skip        (top 20% = printed clinic header — ignored)
  3. Line splitting     (horizontal projection profile)
  4. Word detection     (connected components per line → individual word crops)
  5. TrOCR on word crop (focused single-word input — better than full line)
  6. Medicine matching  (RapidFuzz ≥ 82% vs 1,440 RxHandBD verified names)
  7. Dosage/freq regex  (on joined line text)

Compatible: Python 3.13, torch 2.8+, transformers 4.35+
Device: CPU (MPS has known bugs with generate() output_scores)
"""

import os
import re
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Tuple
import datetime
import time

# ── Medicine vocabulary (loaded from RxHandBD label file) ─────────────

_RXHANDBD_LABEL_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "rxhandbd" / "rxhandbd_labels.txt"
)

# Words that can NEVER be medicine names — blocks false positives
_NOISE_WORDS = {
    "patient", "name", "date", "doctor", "hospital", "clinic", "address",
    "age", "sex", "male", "female", "with", "take", "after", "before",
    "meal", "water", "daily", "tablet", "capsule", "injection", "syrup",
    "tab", "cap", "inj", "sig", "times", "days", "weeks", "month",
    "morning", "night", "evening", "noon", "once", "twice", "thrice",
    "the", "and", "for", "per", "ref", "cont", "stat",
}


def load_medicine_list() -> List[str]:
    """Load verified BD medicine names from RxHandBD label file."""
    if _RXHANDBD_LABEL_FILE.exists():
        names = [l.strip() for l in _RXHANDBD_LABEL_FILE.read_text().splitlines() if l.strip()]
        print(f"[MedDict] Loaded {len(names)} medicines from RxHandBD labels", flush=True)
        return names
    # Minimal safe fallback — only names ≥ 5 chars to reduce false positives
    print("[MedDict] WARNING: rxhandbd_labels.txt not found, using minimal fallback", flush=True)
    return [
        "Napa", "Beklo", "Montair", "Fenadin", "Esoral", "Fixal", "Maxpro",
        "Azithromycin", "Ciprofloxacin", "Metronidazole", "Omeprazole",
        "Paracetamol", "Ibuprofen", "Amoxicillin", "Cetirizine", "Diclofenac",
        "Pantoprazole", "Doxycycline", "Levofloxacin", "Atorvastatin",
    ]


BD_MEDICINES: List[str] = []   # populated at first engine init


def _is_medicine_candidate(token: str) -> bool:
    """
    Pre-filter: only pass tokens to fuzzy matching if they could plausibly
    be a medicine name. Rejects: short noise, common English, all-digit, etc.
    """
    clean = re.sub(r"[^a-zA-Z\-]", "", token).strip()
    if len(clean) < 3:
        return False
    if clean.lower() in _NOISE_WORDS:
        return False
    if clean.isdigit():
        return False
    return True


# ── Regex patterns ────────────────────────────────────────────────────

DOSAGE_PATTERN = re.compile(r"(\d+\.?\d*)\s*(mg|g|ml|mcg|IU)", re.IGNORECASE)

FREQ_PATTERNS = {
    r"\bOD\b": "OD",  r"\bBD\b": "BD",  r"\bTDS\b": "TDS", r"\bQID\b": "QID",
    r"\bSOS\b": "SOS", r"\bHS\b": "HS",
    r"1\s*-\s*0\s*-\s*1": "1-0-1", r"1\s*-\s*1\s*-\s*1": "1-1-1",
    r"0\s*-\s*0\s*-\s*1": "0-0-1", r"1\s*-\s*0\s*-\s*0": "1-0-0",
    r"\bonce\s+daily\b": "OD", r"\btwice\s+daily\b": "BD", r"\bthrice\s+daily\b": "TDS",
}


def _expand_freq(abbr: str) -> str:
    MAP = {
        "OD": "Once daily", "BD": "Twice daily", "TDS": "Thrice daily",
        "QID": "Four times daily", "HS": "At bedtime", "SOS": "As needed",
        "1-0-1": "Morning and night", "1-1-1": "Three times daily",
        "0-0-1": "At night", "1-0-0": "Morning only",
    }
    return MAP.get(abbr.upper(), abbr)


# ── CV Preprocessing ──────────────────────────────────────────────────

def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    if len(arr.shape) == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def preprocess_for_trocr(image: Image.Image) -> Image.Image:
    """
    Full CV preprocessing pipeline before TrOCR inference:
      1. Denoise      — median + bilateral (handles camera noise)
      2. Deskew       — Hough-transform rotation correction
      3. Binarise     — adaptive Gaussian threshold (handles uneven lighting)
      4. Morph cleanup— erosion + dilation (connects broken strokes)
      5. Resize       — height → 384px (TrOCR optimal)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from preprocessing.denoise  import denoise_image
    from preprocessing.deskew   import deskew_image
    from preprocessing.binarise import binarise_image

    cv_img = _pil_to_cv(image)

    # 1. Denoise
    cv_img = denoise_image(cv_img, method="combined")

    # 2. Deskew (Hough transform — corrects page-level tilt)
    cv_img = deskew_image(cv_img)

    # 3. Adaptive Gaussian threshold → binary (black text on white)
    binary = binarise_image(cv_img, method="adaptive_gaussian", block_size=31, C=10)

    # 4. Morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)   # connect broken strokes
    binary = cv2.erode(binary, kernel, iterations=1)    # separate touching chars

    # 5. Convert to RGB (TrOCR expects 3-channel), resize height = 384
    rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(rgb)
    w, h = img.size
    new_h = 384
    new_w = max(int(w * new_h / h), 64)
    return img.resize((new_w, new_h), Image.LANCZOS)


# ── Line Splitting ────────────────────────────────────────────────────

def split_into_lines(image: Image.Image, min_line_height: int = 20) -> List[Image.Image]:
    """
    Split a (pre-processed) prescription image into horizontal line strips.

    - Skips top 20% of image (printed clinic header / address)
    - Uses horizontal projection profile on the Rx body only
    """
    # Skip printed header (top 20%)
    header_cut = int(image.height * 0.20)
    rx_body    = image.crop((0, header_cut, image.width, image.height))

    gray     = np.array(rx_body.convert("L"))
    inverted = 255 - gray
    row_sums = inverted.sum(axis=1)
    threshold = max(row_sums.max() * 0.05, 1)
    text_rows = row_sums > threshold

    lines, in_band, band_start = [], False, 0

    for i, is_text in enumerate(text_rows):
        if is_text and not in_band:
            in_band, band_start = True, i
        elif not is_text and in_band:
            in_band = False
            if (i - band_start) >= min_line_height:
                y0 = max(0, band_start - 4) + header_cut
                y1 = min(image.height, i + 4 + header_cut)
                lines.append(image.crop((0, y0, image.width, y1)))

    if in_band and (len(text_rows) - band_start) >= min_line_height:
        y0 = max(0, band_start - 4) + header_cut
        lines.append(image.crop((0, y0, image.width, image.height)))

    return lines if lines else [image]


# ── TrOCR Engine ──────────────────────────────────────────────────────

MODEL_ID = "microsoft/trocr-large-handwritten"


class TrOCRInference:
    """
    Wraps microsoft/trocr-large-handwritten for line-level prescription OCR.

    Usage:
        engine = TrOCRInference()
        result = engine.predict_prescription(pil_image)
    """

    _instance = None   # singleton

    def __init__(self, model_path: Optional[str] = None):
        global BD_MEDICINES
        self.model     = None
        self.processor = None
        self.device    = "cpu"   # MPS has known generate() bugs

        # Load medicine vocabulary once
        if not BD_MEDICINES:
            BD_MEDICINES = load_medicine_list()

        self._load_model(model_path or MODEL_ID)

    def _load_model(self, model_id: str):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        try:
            print(f"[TrOCR] Loading {model_id}...", flush=True)
            self.processor = TrOCRProcessor.from_pretrained(model_id)
            self.model     = VisionEncoderDecoderModel.from_pretrained(
                model_id, torch_dtype=torch.float16
            )
            self.model.to(self.device).eval()
            print(f"[TrOCR] Ready on {self.device} (float16)", flush=True)
        except Exception as e:
            raise RuntimeError(f"[TrOCR] Model load failed: {e}")

    def _infer_line(self, line_image: Image.Image) -> str:
        """Run TrOCR on a single pre-processed line crop. Returns raw text string."""
        import torch
        try:
            pixel_values = self.processor(
                line_image, return_tensors="pt"
            ).pixel_values.to(self.device).half()

            with torch.no_grad():
                outputs = self.model.generate(pixel_values, max_length=64, num_beams=1)

            return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        except Exception as e:
            print(f"[TrOCR] Line inference error: {e}", flush=True)
            return ""

    def predict_prescription(self, image: Image.Image) -> Dict:
        """
        Full pipeline:
          preprocess → line split → word crops → TrOCR per word
          → RapidFuzz match vs RxHandBD → structured output
        """
        from ocr.word_detector import crop_words

        # 1. CV preprocessing
        clean = preprocess_for_trocr(image)

        # 2. Line splitting
        lines = split_into_lines(clean)
        print(f"[TrOCR] {len(lines)} lines detected", flush=True)

        # 3. Word-level TrOCR per line
        all_line_texts  = []   # full joined line text (for dosage/freq regex)
        all_word_texts  = []   # individual word OCR outputs (for medicine matching)

        for i, line in enumerate(lines):
            if line.width < 60 or line.height < 10:
                continue

            word_crops = crop_words(line)

            if not word_crops:
                # Fallback: run TrOCR on full line if word detector finds nothing
                t0   = time.time()
                text = self._infer_line(line)
                if text:
                    all_line_texts.append(text)
                    all_word_texts.extend(text.split())
                    print(f"[TrOCR] Line {i+1} (full): '{text}' ({time.time()-t0:.1f}s)", flush=True)
                continue

            line_words = []
            for crop in word_crops:
                if crop.width < 8:
                    continue
                t0        = time.time()
                word_text = self._infer_line(crop)
                if word_text:
                    line_words.append(word_text)

            if line_words:
                joined = " ".join(line_words)
                all_line_texts.append(joined)
                all_word_texts.extend(line_words)
                print(f"[TrOCR] Line {i+1}: {line_words}", flush=True)

        raw_text = "\n".join(all_line_texts)

        # 4. Medicine extraction: match each word against RxHandBD vocabulary
        medicines = self._extract_medicines_from_words(all_word_texts, all_line_texts)

        # 5. Metadata heuristics
        patient_name = "Unknown Patient"
        doctor_name  = "Unknown Doctor"
        date_str     = datetime.datetime.utcnow().strftime("%Y-%m-%d")

        for line in all_line_texts:
            ll = line.lower()
            if "patient" in ll or "name:" in ll:
                patient_name = re.sub(r"(?i)(patient|name)\s*:?\s*", "", line).strip() or patient_name
            if "dr." in ll or "doctor" in ll:
                doctor_name = line.strip()
            m = re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", line)
            if m:
                date_str = m.group()

        avg_conf = sum(m.get("confidence", 0.0) for m in medicines) / max(len(medicines), 1)

        return {
            "raw_text":      raw_text.strip(),
            "confidence":    round(avg_conf, 3),
            "medicines":     medicines,
            "patient_name":  patient_name,
            "doctor_name":   doctor_name,
            "date":          date_str,
            "processed_at":  datetime.datetime.utcnow().isoformat(),
            "status":        "completed",
        }

    def _extract_medicines_from_words(
        self, word_texts: List[str], line_texts: List[str]
    ) -> List[Dict]:
        """
        Match individual word OCR outputs against RxHandBD medicine vocabulary.
        Dosage and frequency extracted from full line texts via regex.

        word_texts: flat list of every word TrOCR read (one word per entry)
        line_texts: list of joined line strings (for dosage/freq regex)
        """
        try:
            from rapidfuzz import process, fuzz
            HAS_FUZZ = True
        except ImportError:
            HAS_FUZZ = False

        medicines, seen = [], set()

        # Build dosage + frequency maps from line texts
        line_dosages   = {}
        line_freqs     = {}
        for idx, line in enumerate(line_texts):
            dm = DOSAGE_PATTERN.search(line)
            line_dosages[idx] = f"{dm.group(1)}{dm.group(2).lower()}" if dm else ""
            freq = ""
            for pat, code in FREQ_PATTERNS.items():
                if re.search(pat, line, re.IGNORECASE):
                    freq = code
                    break
            line_freqs[idx] = freq

        # Track which line each word came from (approximate by position)
        words_per_line = max(len(word_texts) // max(len(line_texts), 1), 1)

        for word_idx, word in enumerate(word_texts):
            word = word.strip()
            if not _is_medicine_candidate(word):
                continue

            clean = re.sub(r"[^a-zA-Z\-]", "", word).strip()
            if len(clean) < 3:
                continue

            if HAS_FUZZ:
                result = process.extractOne(clean, BD_MEDICINES, scorer=fuzz.ratio)
                if not result or result[1] < 82:
                    continue
                matched_name = result[0]
                score        = result[1]
            else:
                matched_name = next(
                    (m for m in BD_MEDICINES if m.lower() == clean.lower()), None
                )
                if not matched_name:
                    continue
                score = 100.0

            if matched_name in seen:
                continue
            seen.add(matched_name)

            # Approximate which line this word belongs to
            line_idx    = min(word_idx // words_per_line, len(line_texts) - 1)
            dosage      = line_dosages.get(line_idx, "")
            frequency   = line_freqs.get(line_idx, "")
            was_corrected = clean.lower() != matched_name.lower()

            medicines.append({
                "name":               matched_name,
                "dosage":             dosage,
                "frequency":          frequency,
                "duration":           "",
                "instructions":       "",
                "confidence":         round(score / 100.0, 3),
                "match_score":        round(score, 1),
                "was_corrected":      was_corrected,
                "original_name":      clean if was_corrected else None,
                "frequency_expanded": _expand_freq(frequency) if frequency else "",
            })

        if not medicines and any(word_texts):
            medicines.append({
                "name": "Unrecognized", "dosage": "", "frequency": "",
                "duration": "", "instructions": "Please review manually",
                "confidence": 0.0, "match_score": 0.0,
                "was_corrected": False, "original_name": None,
                "frequency_expanded": "",
            })

        return medicines

    def _extract_medicines(self, raw_text: str) -> List[Dict]:
        """
        Parse raw OCR text into structured medicine entries.
        - Token filter rejects noise words before fuzzy matching
        - Fuzzy threshold ≥ 82 (was 70 — too loose)
        - No multi-token pair generation (caused false positives)
        - No default frequency (empty string = honest)
        """
        try:
            from rapidfuzz import process, fuzz
            HAS_FUZZ = True
        except ImportError:
            HAS_FUZZ = False

        medicines, seen = [], set()

        for line in raw_text.split("\n"):
            line = line.strip()
            if len(line) < 2:
                continue

            # Dosage
            dosage = ""
            dm = DOSAGE_PATTERN.search(line)
            if dm:
                dosage = f"{dm.group(1)}{dm.group(2).lower()}"

            # Frequency
            frequency = ""
            for pat, code in FREQ_PATTERNS.items():
                if re.search(pat, line, re.IGNORECASE):
                    frequency = code
                    break

            # Medicine name — single tokens only (no pair candidates)
            best_name, best_score, original = None, 0.0, None

            for token in line.split():
                if not _is_medicine_candidate(token):
                    continue
                clean = re.sub(r"[^a-zA-Z\s\-]", "", token).strip()
                if len(clean) < 3:
                    continue

                if HAS_FUZZ:
                    result = process.extractOne(clean, BD_MEDICINES, scorer=fuzz.ratio)
                    if result and result[1] >= 82 and result[1] > best_score:
                        best_score = result[1]
                        best_name  = result[0]
                        original   = clean
                else:
                    for med in BD_MEDICINES:
                        if clean.lower() == med.lower():
                            best_name, best_score, original = med, 100.0, clean

            if best_name and best_name not in seen:
                seen.add(best_name)
                was_corrected = original.lower() != best_name.lower() if original else False
                medicines.append({
                    "name":               best_name,
                    "dosage":             dosage,
                    "frequency":          frequency,
                    "duration":           "",
                    "instructions":       "",
                    "confidence":         round(best_score / 100.0, 3),
                    "match_score":        round(best_score, 1),
                    "was_corrected":      was_corrected,
                    "original_name":      original if was_corrected else None,
                    "frequency_expanded": _expand_freq(frequency) if frequency else "",
                })

        if not medicines and raw_text.strip():
            medicines.append({
                "name": "Unrecognized", "dosage": "", "frequency": "",
                "duration": "", "instructions": "Please review manually",
                "confidence": 0.0, "match_score": 0.0,
                "was_corrected": False, "original_name": None,
                "frequency_expanded": "",
            })

        return medicines
