"""
MedScript Backend — FastAPI REST API
Main application entry point.
"""
import os, uuid
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import io, json

# ─── Pydantic Models ──────────────────────────────────────────────────

class MedicineOut(BaseModel):
    name: str
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    instructions: str = ""
    confidence: float = 0.0
    match_score: float = 100.0
    was_corrected: bool = False
    original_name: Optional[str] = None
    frequency_expanded: str = ""

class PrescriptionOut(BaseModel):
    id: str
    raw_text: str
    confidence: float
    medicines: List[MedicineOut]
    patient_name: str = "Patient"
    doctor_name: str = "Dr. Physician"
    date: str = ""
    processed_at: str = ""
    image_url: Optional[str] = None
    status: str = "completed"

class InteractionOut(BaseModel):
    drug_a: str
    drug_b: str
    severity: str
    severity_color: str
    description: str

class InteractionCheckOut(BaseModel):
    interactions: List[InteractionOut]
    total_count: int
    overall_risk: str
    has_severe: bool = False
    has_moderate: bool = False
    medicines_checked: List[str]

class InteractionCheckRequest(BaseModel):
    medicines: List[str]

class StatsOut(BaseModel):
    total_prescriptions: int
    total_medicines: int
    avg_confidence: float
    interaction_alerts: int
    prescriptions_today: int

# ─── In-memory storage for prototype ──────────────────────────────────

prescriptions_db: List[dict] = []
interactions_log: List[dict] = []

# ─── App Setup ────────────────────────────────────────────────────────

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'ml_pipeline'))

from ocr.trocr_inference import TrOCRInference
from ocr.tesseract_inference import TesseractInference
from ocr.postprocess_ocr import PostProcessor
from drug_interaction.interaction_inference import InteractionChecker

# Setup dynamic paths for Kaggle datasets
MODEL_DIR = os.path.join(BASE_DIR, 'ml_pipeline', 'checkpoints', 'trocr_bd_prescription', 'best_model')
DDI_DATA_DIR = os.path.join(BASE_DIR, 'ml_pipeline', 'data', 'ddi_dataset')

# Find CSV in DDI directory if it exists
ddi_csv_path = None
if os.path.exists(DDI_DATA_DIR):
    import glob
    csv_files = glob.glob(os.path.join(DDI_DATA_DIR, "*.csv"))
    if csv_files:
        ddi_csv_path = csv_files[0]

# Use TrOCR (trocr-large-handwritten) as primary OCR engine.
# Falls back to demo mode automatically if model is not yet downloaded.
print('[Backend] Loading TrOCR engine (this may take a moment on first run)...')
ocr_engine = TrOCRInference()
post_processor = PostProcessor()
interaction_checker = InteractionChecker(dataset_path=ddi_csv_path)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="MedScript API",
    description="AI-Powered Prescription Intelligence System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ─── API Routes ───────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/prescriptions/upload", response_model=PrescriptionOut)
async def upload_prescription(image: UploadFile = File(...)):
    """Upload a prescription image for OCR processing."""
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image (jpg, png, etc.)")
    
    contents = await image.read()
    
    # Save image
    ext = image.filename.split('.')[-1] if image.filename else 'jpg'
    img_id = str(uuid.uuid4())
    img_filename = f"{img_id}.{ext}"
    img_path = os.path.join(UPLOAD_DIR, img_filename)
    with open(img_path, 'wb') as f:
        f.write(contents)
    
    # Run OCR
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(contents))
    result = ocr_engine.predict_prescription(img)
    result['medicines'] = post_processor.process_ocr_output(result.get('medicines', []))
    
    prescription = {
        'id': img_id,
        'raw_text': result.get('raw_text', ''),
        'confidence': result.get('confidence', 0),
        'medicines': result.get('medicines', []),
        'patient_name': result.get('patient_name', 'Patient'),
        'doctor_name': result.get('doctor_name', 'Dr. Physician'),
        'date': datetime.utcnow().strftime('%Y-%m-%d'),
        'processed_at': datetime.utcnow().isoformat(),
        'image_url': f"/uploads/{img_filename}",
        'status': 'completed',
    }
    
    prescriptions_db.insert(0, prescription)
    return prescription

@app.get("/api/prescriptions", response_model=List[PrescriptionOut])
async def list_prescriptions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all processed prescriptions."""
    return prescriptions_db[offset:offset + limit]

@app.get("/api/prescriptions/{prescription_id}", response_model=PrescriptionOut)
async def get_prescription(prescription_id: str):
    """Get a single prescription by ID."""
    for rx in prescriptions_db:
        if rx['id'] == prescription_id:
            return rx
    raise HTTPException(404, "Prescription not found")

@app.post("/api/interactions/check", response_model=InteractionCheckOut)
async def check_interactions(req: InteractionCheckRequest):
    """Check drug interactions for a list of medicine names."""
    if len(req.medicines) < 2:
        return {"interactions": [], "total_count": 0, "overall_risk": "none",
                "has_severe": False, "has_moderate": False, "medicines_checked": req.medicines}
    result = interaction_checker.check(req.medicines)
    interactions_log.append(result)
    return result

@app.get("/api/stats", response_model=StatsOut)
async def get_stats():
    """Get dashboard statistics."""
    total_rx = len(prescriptions_db)
    total_meds = sum(len(rx.get('medicines', [])) for rx in prescriptions_db)
    avg_conf = 0.0
    if total_rx > 0:
        confs = [rx.get('confidence', 0) for rx in prescriptions_db]
        avg_conf = sum(confs) / len(confs)
    today = datetime.utcnow().strftime('%Y-%m-%d')
    today_count = sum(1 for rx in prescriptions_db if rx.get('date','').startswith(today))
    alert_count = len(interactions_log)
    return {
        "total_prescriptions": total_rx,
        "total_medicines": total_meds,
        "avg_confidence": round(avg_conf, 3),
        "interaction_alerts": alert_count,
        "prescriptions_today": today_count,
    }

@app.get("/api/medicines")
async def list_medicines():
    """List all known medicines in the database."""
    from ocr.trocr_inference import BD_MEDICINES
    return {"medicines": BD_MEDICINES, "total": len(BD_MEDICINES)}

@app.get("/api/medicines/{name}/interactions")
async def get_medicine_interactions(name: str):
    """Get all known interactions for a specific medicine."""
    from drug_interaction.build_knowledge_graph import DrugInteractionGraph
    graph = DrugInteractionGraph()
    return graph.get_drug_info(name)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
