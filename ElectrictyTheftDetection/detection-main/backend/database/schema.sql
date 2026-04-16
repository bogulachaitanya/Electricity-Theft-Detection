-- Database Schema for Electricity Theft Detection System

-- Table: Consumers
CREATE TABLE IF NOT EXISTS consumers (
    cons_no TEXT PRIMARY KEY,
    name TEXT,
    address TEXT,
    dt_id TEXT,             -- Distribution Transformer ID
    feeder_id TEXT          -- Feeder Line ID
);

-- Table: Hourly Consumption (Optional, usually kept in CSV/Parquet for performance)
-- But we define a summary table for the dashboard
CREATE TABLE IF NOT EXISTS consumption_summary (
    cons_no TEXT,
    last_updated DATETIME,
    avg_daily_usage REAL,
    peak_usage REAL,
    FOREIGN KEY(cons_no) REFERENCES consumers(cons_no)
);

-- Table: Risk Assessments (Result of the ML Engine)
CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cons_no TEXT,
    assessment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    risk_score REAL,
    risk_tier TEXT,         -- Normal, Suspicious, High Risk, Theft
    top_rule_flagged TEXT,
    xgboost_prob REAL,
    lstm_error REAL,
    status TEXT DEFAULT 'Open', -- Open, Assigned, Verified, False Positive
    FOREIGN KEY(cons_no) REFERENCES consumers(cons_no)
);

-- Table: Investigation Logs
CREATE TABLE IF NOT EXISTS investigation_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id INTEGER,
    inspector_name TEXT,
    visit_date DATETIME,
    findings TEXT,          -- Meter Tampered, Bypass, Faulty Meter, Normal
    action_taken TEXT,      -- Replaced, Fined, None
    FOREIGN KEY(assessment_id) REFERENCES risk_assessments(assessment_id)
);
