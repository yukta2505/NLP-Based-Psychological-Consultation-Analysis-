"""
backend/api.py
FastAPI REST API for the Psychological Consultation NLP System
"""

import os
import json
import sqlite3
import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile

# Import our pipeline
import sys
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import analyze

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="Psychological NLP Analysis API",
    description="NLP-based mental health consultation analysis system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(__file__), "../database/psych_nlp.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ──────────────────────────────────────────────
# Database initialization
# ──────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id   TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            age          INTEGER,
            gender       TEXT,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id           TEXT PRIMARY KEY,
            patient_id           TEXT NOT NULL,
            date                 TEXT NOT NULL,
            raw_text             TEXT,
            emotions             TEXT,
            predicted_disorder   TEXT,
            symptoms             TEXT,
            recommended_therapy  TEXT,
            severity_score       REAL,
            severity_label       TEXT,
            mind_map             TEXT,
            report               TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );
    """)
    conn.commit()
    conn.close()


init_db()


# ──────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str
    patient_name: Optional[str] = "Patient"
    session_id: Optional[str] = None


class AddPatientRequest(BaseModel):
    patient_id: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None


class SessionSaveRequest(BaseModel):
    patient_id: str
    session_data: dict


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────
def save_session_to_db(patient_id: str, session: dict):
    """
    Save session. Auto-creates patient record if not exists.
    patient_id is derived from patient_name (lowercased, spaces→underscores).
    """
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    patient_name = session.get("patient_name", "Anonymous Patient")

    # Always upsert patient so record is always present
    cur.execute("""
        INSERT INTO patients (patient_id, name, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(patient_id) DO UPDATE SET name=excluded.name
    """, (patient_id, patient_name, datetime.datetime.now().isoformat()))

    cur.execute("""
        INSERT OR REPLACE INTO sessions
        (session_id, patient_id, date, raw_text, emotions,
         predicted_disorder, symptoms, recommended_therapy,
         severity_score, severity_label, mind_map, report)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session.get("session_id", f"S-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"),
        patient_id,
        session.get("timestamp") or datetime.datetime.now().isoformat(),
        session.get("original_text", "")[:1000],
        json.dumps(session.get("emotions", [])),
        session.get("predicted_disorder", ""),
        json.dumps(session.get("entities", {}).get("SYMPTOM", [])),
        session.get("recommended_therapy", ""),
        float(session.get("severity_score", 0)),
        session.get("severity_label", ""),
        json.dumps(session.get("mind_map", {})),
        session.get("report", ""),
    ))
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Psychological NLP Analysis API is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}


@app.get("/metrics")
def get_metrics():
    """Return stored model evaluation metrics."""
    results_dir = Path(os.path.dirname(__file__)) / "../results"

    def _load_json(p: Path):
        try:
            if not p.exists():
                return None
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    merged = {}
    loaded_files = []

    base = _load_json(results_dir / "metrics.json")
    if isinstance(base, dict):
        merged.update(base)
        loaded_files.append("metrics.json")

    disorder = _load_json(results_dir / "disorder_metrics.json")
    if isinstance(disorder, dict) and isinstance(disorder.get("disorder_model"), dict):
        merged["disorder_model"] = disorder["disorder_model"]
        loaded_files.append("disorder_metrics.json")

    ner = _load_json(results_dir / "ner_metrics.json")
    if isinstance(ner, dict) and ("ents_p" in ner or "ents_f" in ner or "ents_per_type" in ner):
        merged["ner_model"] = {
            "model": "ner_model (spaCy)",
            "precision": ner.get("ents_p"),
            "recall": ner.get("ents_r"),
            "f1": ner.get("ents_f"),
            "per_class": ner.get("ents_per_type") or {},
        }
        loaded_files.append("ner_metrics.json")

    emotion = _load_json(results_dir / "emotion_metrics.json")
    if isinstance(emotion, dict) and isinstance(emotion.get("emotion_model"), dict):
        merged["emotion_model"] = emotion["emotion_model"]
        loaded_files.append("emotion_metrics.json")

    therapy = _load_json(results_dir / "therapy_metrics.json")
    if isinstance(therapy, dict) and isinstance(therapy.get("therapy_model"), dict):
        merged["therapy_model"] = therapy["therapy_model"]
        loaded_files.append("therapy_metrics.json")

    if merged:
        merged["_meta"] = {"loaded_files": loaded_files}
        return merged

    return {"message": "No metrics found. Train models first."}


@app.post("/analyze")
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze psychological consultation text.
    Returns: emotions, severity, disorder, NER entities, therapy, mind map, report.
    """
    import traceback
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=422, detail="Text field cannot be empty.")
        result = analyze(
            text=request.text,
            patient_name=request.patient_name or "Patient",
            session_id=request.session_id,
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/analyze-pdf")
async def analyze_pdf(
    file: UploadFile = File(...),
    patient_name: str = Form("Patient"),
):
    """
    Analyze a PDF consultation file.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = analyze(
            text="",
            patient_name=patient_name,
            pdf_path=tmp_path,
        )
        os.unlink(tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-patient")
def add_patient(request: AddPatientRequest):
    """Register a new patient."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO patients (patient_id, name, age, gender) VALUES (?, ?, ?, ?)",
        (request.patient_id, request.name, request.age, request.gender)
    )
    conn.commit()
    conn.close()
    return {"message": f"Patient {request.name} registered", "patient_id": request.patient_id}


@app.post("/add-session")
def add_session(request: SessionSaveRequest):
    """Save a session result to the database."""
    try:
        save_session_to_db(request.patient_id, request.session_data)
        return {"message": "Session saved", "session_id": request.session_data.get("session_id")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patient-history/{patient_id}")
def patient_history(patient_id: str):
    """Retrieve full session history. Searches by patient_id OR name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Try exact patient_id first
    patient = cur.execute(
        "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
    ).fetchone()

    # Fallback: search by name (case-insensitive)
    if not patient:
        patient = cur.execute(
            "SELECT * FROM patients WHERE LOWER(name) = LOWER(?)", (patient_id,)
        ).fetchone()

    # Fallback: search by name containing the query
    if not patient:
        patient = cur.execute(
            "SELECT * FROM patients WHERE LOWER(name) LIKE LOWER(?)",
            (f"%{patient_id}%",)
        ).fetchone()

    if not patient:
        conn.close()
        # Return helpful message with list of available patients
        all_patients = sqlite3.connect(DB_PATH)
        all_patients.row_factory = sqlite3.Row
        rows = all_patients.execute("SELECT patient_id, name FROM patients").fetchall()
        all_patients.close()
        available = [{"patient_id": r["patient_id"], "name": r["name"]} for r in rows]
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Patient '{patient_id}' not found.",
                "hint": "Use the exact patient_id shown after saving a session.",
                "available_patients": available
            }
        )

    sessions = cur.execute(
        "SELECT * FROM sessions WHERE patient_id = ? ORDER BY date DESC",
        (patient_id,)
    ).fetchall()
    conn.close()

    def row_to_dict(row):
        d = dict(row)
        for k in ["emotions", "symptoms", "mind_map"]:
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    pass
        return d

    return {
        "patient": dict(patient),
        "sessions": [row_to_dict(s) for s in sessions],
        "session_count": len(sessions),
    }


@app.get("/all-patients")
def all_patients():
    """List all registered patients with session counts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    patients = cur.execute("""
        SELECT p.*, COUNT(s.session_id) as session_count
        FROM patients p
        LEFT JOIN sessions s ON p.patient_id = s.patient_id
        GROUP BY p.patient_id
        ORDER BY p.created_at DESC
    """).fetchall()
    conn.close()
    return {"patients": [dict(p) for p in patients]}


@app.get("/progress/{patient_id}")
def patient_progress(patient_id: str):
    """Return severity trend across sessions for progress visualization."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    sessions = cur.execute(
        """SELECT session_id, date, severity_score, severity_label,
                  predicted_disorder, recommended_therapy
           FROM sessions WHERE patient_id = ?
           ORDER BY date ASC""",
        (patient_id,)
    ).fetchall()
    conn.close()

    if not sessions:
        raise HTTPException(status_code=404, detail="No sessions found for patient")

    data = [dict(s) for s in sessions]
    return {
        "patient_id": patient_id,
        "progress": data,
        "trend": "improving" if (
            len(data) > 1 and
            data[-1]["severity_score"] < data[0]["severity_score"]
        ) else "stable",
    }


# ── Feature 12: Dashboard stats ─────────────────────────────────
@app.get("/dashboard-stats")
def dashboard_stats():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    total_patients = cur.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    total_sessions = cur.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    # Most common disorder this month
    from datetime import datetime, timedelta
    month_start = (datetime.now().replace(day=1)).isoformat()
    disorder_row = cur.execute("""
        SELECT predicted_disorder, COUNT(*) as cnt
        FROM sessions WHERE date >= ? AND predicted_disorder != ''
        GROUP BY predicted_disorder ORDER BY cnt DESC LIMIT 1
    """, (month_start,)).fetchone()
    top_disorder = dict(disorder_row) if disorder_row else {"predicted_disorder": "N/A", "cnt": 0}

    # Average severity this month
    avg_sev = cur.execute("""
        SELECT AVG(severity_score) FROM sessions WHERE date >= ?
    """, (month_start,)).fetchone()[0]

    # Severity trend (last 10 sessions)
    trend_rows = cur.execute("""
        SELECT date, severity_score, severity_label, predicted_disorder
        FROM sessions ORDER BY date DESC LIMIT 10
    """).fetchall()
    trend = [dict(r) for r in reversed(trend_rows)]

    # Recent sessions
    recent = cur.execute("""
        SELECT s.session_id, s.date, s.predicted_disorder,
               s.severity_label, s.severity_score, p.name
        FROM sessions s JOIN patients p ON s.patient_id = p.patient_id
        ORDER BY s.date DESC LIMIT 5
    """).fetchall()

    # Disorder distribution
    disorders = cur.execute("""
        SELECT predicted_disorder, COUNT(*) as count
        FROM sessions WHERE predicted_disorder != ''
        GROUP BY predicted_disorder ORDER BY count DESC
    """).fetchall()

    conn.close()
    return {
        "total_patients":   total_patients,
        "total_sessions":   total_sessions,
        "top_disorder":     top_disorder,
        "avg_severity":     round(float(avg_sev), 2) if avg_sev else 0.0,
        "severity_trend":   trend,
        "recent_sessions":  [dict(r) for r in recent],
        "disorder_distribution": [dict(r) for r in disorders],
    }


# ── Feature 8: Multi-session longitudinal report ─────────────────
@app.get("/longitudinal-report/{patient_id}")
def longitudinal_report(patient_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    patient = cur.execute(
        "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
    ).fetchone()
    if not patient:
        conn.close()
        raise HTTPException(status_code=404, detail="Patient not found")

    sessions = cur.execute("""
        SELECT * FROM sessions WHERE patient_id = ?
        ORDER BY date ASC
    """, (patient_id,)).fetchall()
    conn.close()

    if not sessions:
        raise HTTPException(status_code=404, detail="No sessions found")

    sessions_list = []
    all_symptoms  = {}

    for s in sessions:
        d = dict(s)
        try:
            d["symptoms"] = json.loads(d.get("symptoms") or "[]")
        except Exception:
            d["symptoms"] = []
        try:
            d["emotions"] = json.loads(d.get("emotions") or "[]")
        except Exception:
            d["emotions"] = []
        sessions_list.append(d)

        # Feature 11: Count symptom frequency
        for sym in d["symptoms"]:
            key = sym.lower().strip()
            if key:
                all_symptoms[key] = all_symptoms.get(key, 0) + 1

    # Classify symptoms by frequency
    persistent_symptoms = {k: v for k, v in all_symptoms.items() if v >= len(sessions_list) * 0.6}
    recurring_symptoms  = {k: v for k, v in all_symptoms.items() if 0.3 <= v / len(sessions_list) < 0.6}
    occasional_symptoms = {k: v for k, v in all_symptoms.items() if v / len(sessions_list) < 0.3}

    # Severity trajectory
    first_sev = sessions_list[0].get("severity_score",  0)
    last_sev  = sessions_list[-1].get("severity_score", 0)
    trajectory = "improving" if last_sev < first_sev - 0.3 else                  "worsening" if last_sev > first_sev + 0.3 else "stable"

    # Therapies tried
    therapies_tried = list(dict.fromkeys(
        s.get("recommended_therapy", "") for s in sessions_list
        if s.get("recommended_therapy")
    ))

    # Disorders across sessions
    disorders_seen = list(dict.fromkeys(
        s.get("predicted_disorder", "") for s in sessions_list
        if s.get("predicted_disorder")
    ))

    return {
        "patient":           dict(patient),
        "total_sessions":    len(sessions_list),
        "sessions":          sessions_list,
        "trajectory":        trajectory,
        "severity_change":   round(last_sev - first_sev, 2),
        "first_severity":    first_sev,
        "last_severity":     last_sev,
        "therapies_tried":   therapies_tried,
        "disorders_seen":    disorders_seen,
        # Feature 11: Symptom frequency
        "symptom_frequency": all_symptoms,
        "persistent_symptoms":  persistent_symptoms,
        "recurring_symptoms":   recurring_symptoms,
        "occasional_symptoms":  occasional_symptoms,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
