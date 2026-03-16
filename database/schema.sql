-- Psychological NLP System Database Schema
-- SQLite

CREATE TABLE IF NOT EXISTS patients (
    patient_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    age          INTEGER,
    gender       TEXT CHECK(gender IN ('Male','Female','Other','Prefer not to say')),
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id           TEXT PRIMARY KEY,
    patient_id           TEXT NOT NULL,
    date                 TEXT NOT NULL,
    raw_text             TEXT,
    emotions             TEXT,          -- JSON array
    predicted_disorder   TEXT,
    symptoms             TEXT,          -- JSON array
    recommended_therapy  TEXT,
    severity_score       REAL,
    severity_label       TEXT CHECK(severity_label IN ('Low','Medium','High')),
    mind_map             TEXT,          -- JSON object {nodes, edges}
    report               TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_sessions_date    ON sessions(date);
CREATE INDEX IF NOT EXISTS idx_sessions_disorder ON sessions(predicted_disorder);